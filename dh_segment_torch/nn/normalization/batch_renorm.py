import torch
from torch.nn import Parameter, init


class BatchRenorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        self.weight = Parameter(torch.tensor(num_features, dtype=torch.float), requires_grad=True)
        self.bias = Parameter(torch.tensor(num_features, dtype=torch.float), requires_grad=True)

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_std", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

        self.reset_parameters()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_std.fill_(1)
        self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def _check_input_dim(self, input):
        raise NotImplementedError

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(BatchRenorm, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    @property
    def rmax(self) -> int:
        return (2 / 35000 * self.num_batches_tracked + 25/35).clamp_(1.0, 3.0)

    @property
    def dmax(self) -> int:
        return (5 / 20000 * self.num_batches_tracked - 25/20).clamp_(0.0, 5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(x)

        if self.training:
            self.num_batches_tracked = self.num_batches_tracked + 1
            if self.momentum is None:
                exponential_average_factor = 1.0 / float(self.num_batches_tracked)
            else:
                exponential_average_factor = self.momentum
            dims = tuple([i for i in range(x.dim()) if i != 1])
            batch_mean = x.mean(dims, keepdim=True)
            batch_std = x.std(dims, unbiased=False, keepdim=True) + self.eps

            r = (batch_std.detach() / self.running_std.view_as(batch_std)).clamp_(1 / self.rmax, self.rmax)

            d = ((batch_mean.detach() - self.running_mean.view_as(batch_std)) / self.running_std.view_as(batch_std)).clamp_(-self.dmax, self.dmax)

            x = (x - batch_mean) / batch_std * r + d

            self.running_mean += exponential_average_factor * (batch_mean.detach().view_as(self.running_mean) - self.running_mean)
            self.running_std += exponential_average_factor * (batch_std.detach().view_as(self.running_std) - self.running_std)
        else:
            return (x - self.running_mean) / self.running_std

        if self.affine:
            x = self.weight * x + self.bias
        return x


class BatchRenorm2d(BatchRenorm):
    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError("expected 4D input (got {x.dim()}D input)")