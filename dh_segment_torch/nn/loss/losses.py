from typing import List, Optional, Callable, Union

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.loss import dice_loss
from dh_segment_torch.nn.loss import topology_loss
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
                loss = loss[..., self.margin : -self.margin, self.margin : -self.margin]
            return loss


@Loss.register("cross_entropy")
class CrossEntropyLoss(Loss):
    """Cross entropy loss.

    C.f. :class:`torch.nn.CrossEntropyLoss` for this loss

    :param weights: C.f. pytorch doc.
    :param size_average: C.f. pytorch doc.
    :param ignore_index: C.f. pytorch doc.
    :param reduction: C.f. pytorch doc.
    :param ignore_padding: Whether to compute the loss ignoring the padding and margin.
    :param margin: The margin size, only used if the padding is ignored in the computation.
    """
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        size_average: Optional[bool] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        ignore_padding: bool = False,
        margin: int = 0,
    ):
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
    """Binary Cross Entropy Loss with logits.

     C.f. :class:`torch.nn.BCEWithLogitsLoss` for this loss

    :param weights: C.f. pytorch doc.
    :param size_average: C.f. pytorch doc.
    :param reduction: C.f. pytorch doc.
    :param ignore_padding: Whether to compute the loss ignoring the padding and margin.
    :param margin: The margin size, only used if the padding is ignored in the computation.
    """
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        size_average: Optional[bool] = None,
        reduction: str = "mean",
        ignore_padding: bool = False,
        margin: int = 0,
    ):
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
    r"""Dice Coefficient Loss.

    This loss penalizes the dissimilarity between the ground truth :math:`Y` and the prediction :math:`\hat{Y}` with a smoothing factor :math:`\alpha`, it is defined as:

    .. math::
        \text{loss}(Y, \hat{Y}) = 1-\frac{2 \left(\vert Y\cap \hat{Y} \vert + \alpha \right)}{\vert Y \vert + \vert \hat{Y} \vert + \alpha}

    It ressembles Intersection over Union (IoU), except that it counts true positives twice in the denominator.

    :param smooth: Smoothing ratio.
    :param ignore_padding: Whether to compute the loss ignoring the padding and margin.
    :param margin: The margin size, only used if the padding is ignored in the computation.
    """
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


@Loss.register("topology")
class TopologyLoss(Loss):
    r"""Topology loss.

    Implementation of the loss presented in  `arXiv:1712.02190 <https://arxiv.org/abs/1712.02190>`__.

    For a selection of labels, it takes the corresponding ground truth and prediction probabilities,
    compute for a selection of layers levels of an imagenet pre-trained VGG19 the result of each probabilities map and
    finally compute the sum of RMSE.

    :param layers_sel: Single layer or list of layers for the VGG19 selection, can be in range [1,4].
    :param labels_sel: List of labels index selected.
    :param multilabel: Whether the model is multilabel.
    :param ignore_padding: Cannot be used with this loss.
    :param margin: Not used.
    """
    def __init__(
        self,
        layers_sel: Union[int, List[int]],
        labels_sel: Optional[List[int]] = None,
        multilabel: bool = False,
        ignore_padding: bool = False,
        margin: int = 0,
    ):
        if ignore_padding:
            raise ValueError("Cannot compute topology loss and ignore padding")
        loss = topology_loss.TopologyLoss(layers_sel, labels_sel, multilabel)
        super().__init__(
            loss_module=loss, ignore_padding=False, margin=margin,
        )


@Loss.register("combined")
class CombinedLoss(Loss):
    r"""Weighted combination of losses.

    Combines several losses with pre-defined weights or equal weights by default.

    :param losses: List of :class:`losses <.Loss>`.
    :param weights: Optional weights list that should be the same size as the number of losses.
    """
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
