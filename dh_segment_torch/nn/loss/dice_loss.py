import torch


class Dice(torch.nn.Module):
    def __init__(self, smooth: float = 1.0, no_reduce: bool = False):
        super().__init__()
        self.smooth = smooth
        self.no_reduce = no_reduce

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(input)
        intersection = target*probs
        union = target + probs

        stack = torch.stack([intersection.sum(1), union.sum(1)], 1)
        if self.no_reduce:
            return stack
        else:
            return self.reduce_dice(stack)

    def reduce_dice(self, stack):
        if stack.ndim == 4:
            intersection = stack[:, 0].squeeze()
            union = stack[:, 1].squeeze()
            dims = (1, 2)
        else:
            intersection = stack[0]
            union = stack[1]
            dims = (0, 1)
        return (1 - (2 * (intersection.sum(dims) + self.smooth) / (union.sum(dims) + self.smooth))).mean()
