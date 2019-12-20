from typing import Any

import torch


def normalize_batch(batch: torch.Tensor, mean: Any, std: Any, inplace: bool = False):
    if not _is_image_batch(batch):
        raise TypeError("batch is not a batch of torch images")

    if not inplace:
        batch = batch.clone()

    dtype = batch.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=batch.device)
    std = torch.as_tensor(std, dtype=dtype, device=batch.device)
    batch.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return batch


def _is_image_batch(batch):
    return torch.is_tensor(batch) and batch.ndimension() == 4 and batch.shape[1] == 3