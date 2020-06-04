from typing import Tuple, Optional

import torch
import numpy as np


def batch_multilabel_confusion_matrix(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, multilabel: bool
) -> torch.Tensor:
    y_true, y_pred = validate_input(y_true, y_pred, n_classes, multilabel)
    n_classes = y_pred.shape[1]
    num_samples = y_pred.shape[2:].numel()

    true_and_pred = y_true.mul(y_pred)

    tp_sum = batch_bincount(true_and_pred, minlength=n_classes)
    pred_sum = batch_bincount(y_pred, minlength=n_classes)
    true_sum = batch_bincount(y_true, minlength=n_classes)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = num_samples - tp - fn - fp
    return torch.stack([tn, fp, fn, tp]).permute(1, 2, 0).reshape(-1, n_classes, 2, 2)


def validate_input(
    y_true: torch.Tensor, y_pred: torch.Tensor, n_classes: int, multilabel: bool
) -> Tuple[torch.sparse.LongTensor, torch.sparse.LongTensor]:
    if multilabel:
        assert y_true.shape[1] == n_classes
        y_true = one_hot_to_sparse(y_true)
        y_pred = one_hot_to_sparse(y_pred)
    else:
        y_true = multiclass_to_one_hot(y_true, n_classes)
        y_pred = multiclass_to_one_hot(y_pred, n_classes)

    assert y_true.shape == y_pred.shape
    assert y_pred.ndim > 2
    return y_true, y_pred


def one_hot_to_sparse(one_hot: torch.Tensor) -> torch.sparse.LongTensor:
    nonzeros = torch.nonzero(one_hot, as_tuple=False)
    return torch.sparse.LongTensor(
        nonzeros.t(), torch.ones(nonzeros.shape[0], dtype=torch.long), one_hot.shape
    ).coalesce()


def multiclass_to_one_hot(
    multiclass: torch.Tensor, n_classes: int
) -> torch.sparse.LongTensor:
    if not multiclass.dtype == torch.long:
        multiclass = multiclass.to(torch.long)
    nonzeros = torch.nonzero(multiclass, as_tuple=False)
    indices = torch.cat(
        [
            nonzeros[:, :1],
            (multiclass[multiclass > 0] - 1).unsqueeze(1),
            nonzeros[:, 1:],
        ],
        dim=-1,
    )
    shape = multiclass.shape[:1] + torch.Size([n_classes - 1]) + multiclass.shape[1:]

    one_hot = torch.sparse.LongTensor(
        indices.t(), torch.ones(indices.shape[0], dtype=torch.long), shape
    ).coalesce()

    return one_hot


def batch_bincount(
    batch_input: torch.Tensor, weights: Optional[torch.Tensor] = None, minlength=0
):
    if batch_input.ndim < 3:
        raise ValueError("Expected a tensor of shape (batch, classes, ...other dims)")
    return torch.cat(
        [
            torch.bincount(
                batch_input[idx].coalesce().indices()[0], weights, minlength
            ).unsqueeze(0)
            for idx in range(batch_input.shape[0])
        ],
        dim=0,
    )


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
