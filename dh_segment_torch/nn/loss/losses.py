from typing import List, Optional, Callable

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.loss import dice_loss
from dh_segment_torch.utils.ops import cut_with_padding


class Loss(torch.nn.Module, Registrable):
    def __init__(
        self,
        loss_module: torch.nn.Module,
        reduce_function: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        super().__init__()
        self.ignore_padding = ignore_padding
        self.margin = margin
        self._loss_module = loss_module
        self._reduce_function = reduce_function

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = self._loss_module(input, target)
        if self.ignore_padding:
            if shapes is None:
                raise ValueError("Ignoring padding, thus shapes should be null.")
            return compute_with_shapes(
                loss, shapes, reduce=self._reduce_function, margin=self.margin
            )
        else:
            if self.margin > 0:
                loss = loss[..., self.margin:-self.margin, self.margin:-self.margin]
            return loss


@Loss.register("cross_entropy")
class CrossEntropyLoss(Loss):
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        """

        :param weights: Class weights, must be of size C
        :param size_average:
        :param ignore_index:
        :param reduction:
        :param ignore_padding:
        :param margin:
        """
        if ignore_padding:
            reduction = "none"
        loss = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(weights).to(torch.float)
            if weights is not None
            else weights,
            size_average=size_average,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        super().__init__(loss_module=loss, ignore_padding=ignore_padding, margin=margin)


@Loss.register("bce_with_logits")
class BCEWithLogitsLoss(Loss):
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        size_average: Optional[bool] = None,
        reduction: str = "mean",
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        """

        :param weights: Class weights, must be of size C
        :param size_average:
        :param reduction:
        :param ignore_padding:
        :param margin:
        """
        if ignore_padding:
            reduction = "none"
        loss = torch.nn.BCEWithLogitsLoss(
            size_average=size_average,
            reduction=reduction,
            pos_weight=torch.tensor(weights).to(torch.float)
            if weights is not None
            else weights,
        )
        super().__init__(loss_module=loss, ignore_padding=ignore_padding, margin=margin)


@Loss.register("dice")
class DiceLoss(Loss):
    def __init__(
        self, smooth: float = 1.0, ignore_padding: bool = False, margin: int = 0
    ):
        loss = dice_loss.Dice(smooth, no_reduce=ignore_padding)
        super().__init__(
            loss_module=loss,
            reduce_function=loss.reduce_dice,
            ignore_padding=ignore_padding,
            margin=margin,
        )


@Loss.register("combined_loss")
class CombinedLoss(Loss):
    def __init__(self, losses: List[Loss], weights: Optional[List[float]] = None):
        super().__init__(torch.nn.Identity)
        self.losses = losses
        self.weights = weights if weights is not None else [1.0] * len(losses)

        if len(self.losses) != len(self.weights):
            raise ValueError(
                "Should have the same number of losses and weights,"
                f"got {len(self.losses)} losses and {len(self.weights)} weights"
            )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        shapes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = torch.tensor(0.0).to(input.device)
        for loss, weight in zip(self.losses, self.weights):
            res += loss.forward(input, target, shapes) * weight
        return res / sum(self.weights)


def compute_with_shapes(input_tensor, shapes, reduce=torch.mean, margin=0):
    res = torch.tensor(0.0).to(input_tensor.device)
    for idx in range(shapes.shape[0]):
        shape = shapes[idx]
        res += reduce(cut_with_padding(input_tensor[idx], shape, margin))
    res = reduce(res)
    return res
