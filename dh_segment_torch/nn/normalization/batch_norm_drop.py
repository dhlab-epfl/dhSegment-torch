import torch

from dh_segment_torch.nn.normalization.batch_renorm import BatchRenorm2d


class BatchNorm2dDrop(torch.nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[0] == 1:
            return input
        else:
            return super().forward(input)


class BatchRenorm2dDrop(BatchRenorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[0] == 1:
            return input
        else:
            return super().forward(input)