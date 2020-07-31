from typing import Optional, Union, List

import numpy as np
import torch


def batch_multilabel_confusion_matrix(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    n_classes: int,
    multilabel: bool,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    validate_input(y_true, y_pred, n_classes, multilabel)

    n_batches = y_pred.shape[0]

    if multilabel:
        true_and_pred = y_true.mul(y_pred)
        num_samples = y_pred.shape[2:].numel()
        tp_sum = batch_bincount(
            batch_nonzero(true_and_pred, n_batches),
            n_batches,
            weights,
            minlength=n_classes,
        )
        pred_sum = batch_bincount(
            batch_nonzero(y_pred, n_batches), n_batches, weights, minlength=n_classes
        )
        true_sum = batch_bincount(
            batch_nonzero(y_true, n_batches), n_batches, weights, minlength=n_classes
        )
    else:
        true_and_pred = [
            y_pred[batch_idx][y_pred[batch_idx] == y_true[batch_idx]]
            for batch_idx in range(n_batches)
        ]
        num_samples = y_pred.shape[1:].numel()
        tp_sum = batch_bincount(true_and_pred, n_batches, weights, minlength=n_classes)[
            :, 1:
        ]
        pred_sum = batch_bincount(
            y_pred.view(n_batches, -1), n_batches, weights, minlength=n_classes
        )[:, 1:]
        true_sum = batch_bincount(
            y_true.view(n_batches, -1), n_batches, weights, minlength=n_classes
        )[:, 1:]
        n_classes -= 1

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = num_samples - tp - fn - fp
    return torch.stack([tn, fp, fn, tp]).permute(1, 2, 0).reshape(-1, n_classes, 2, 2)


def validate_input(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, multilabel: bool
) -> None:
    if multilabel:
        assert y_true.shape[1] == n_classes

    else:
        assert y_pred.max() < n_classes
        assert y_true.max() < n_classes

    assert y_true.shape == y_pred.shape
    assert y_pred.ndim > 2


def batch_bincount(
    batch_input: Union[torch.Tensor, List[torch.Tensor]],
    n_batches: int,
    weights: Optional[torch.Tensor] = None,
    minlength=0,
) -> torch.Tensor:
    return torch.stack(
        [
            torch.bincount(batch_input[batch_idx], weights=weights, minlength=minlength)
            for batch_idx in range(n_batches)
        ]
    )


def batch_nonzero(batch_input: torch.Tensor, n_batches: int,) -> List[torch.Tensor]:
    non_zeros = torch.nonzero(batch_input, as_tuple=False)
    return [
        non_zeros[:, 1][non_zeros[:, 0] == batch_idx] for batch_idx in range(n_batches)
    ]


def nanaverage(a, axis=None, weights=None):
    a = np.asarray(a)

    if weights is None:
        return np.nanmean(a, axis=axis)
    else:
        a_mask = np.isnan(a)
        a_masked = a.copy()
        a_masked[a_mask] = 0
        weights_mask = weights == 0
        weights_masked = weights.copy()
        weights_masked[weights_mask] = 1

        count = np.sum(~a_mask * weights, axis=axis)
        total = np.sum(a_masked * weights, axis=axis)

        return total / count
