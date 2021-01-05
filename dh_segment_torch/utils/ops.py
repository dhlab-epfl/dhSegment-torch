from datetime import datetime
from itertools import islice
from typing import (
    Iterable,
    List,
    Union,
    Dict,
    Optional,
    TypeVar,
    Iterator,
)

import numpy as np

import torch

T = TypeVar("T")


def cut_with_padding(
    input_tensor: torch.Tensor, shape: torch.Tensor, margin: int = 0
) -> torch.Tensor:
    return input_tensor[
        ..., margin : shape[0].item() - margin, margin : shape[1].item() - margin
    ]


def detach_and_move_tensors(
    *tensors: torch.Tensor, device: Optional[str] = None, non_blocking: bool = True,
) -> Iterable[torch.Tensor]:
    tensors = [
        tensor.detach().to(
            device if device else tensor.device, non_blocking=non_blocking
        )
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in tensors
    ]
    if len(tensors) == 1:
        return tensors[0]
    return tensors


def batch_items(items: Iterable[T], batch_size: int = 1) -> Iterator[T]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break


def move_batch(
    batch: Dict[str, torch.Tensor], device: str, non_blocking: bool = True
) -> Dict[str, torch.Tensor]:
    return {k: t.to(device, non_blocking=non_blocking) for k, t in batch.items()}


def move_and_detach_batch(
    batch: Dict[str, torch.Tensor], device: str, non_blocking: bool = True
) -> Dict[str, torch.Tensor]:
    return {
        k: t.to(device, non_blocking=non_blocking).detach() for k, t in batch.items()
    }


def join_not_none(*items: Optional[str], join_str: str = "_"):
    return join_str.join([item for item in items if item and len(item) > 0])


def format_time(timestamp: float) -> str:
    timestamp = datetime.fromtimestamp(timestamp)
    return (
        f"{timestamp.year:04d}-{timestamp.month:02d}-{timestamp.day:02d}-"
        f"{timestamp.hour:02d}-{timestamp.minute:02d}-{timestamp.second:02d}"
    )


def should_run(iteration: int, every: int):
    return iteration >= every and iteration % every == 0


def normalize_dict(dict_: Union[Dict[str, T], List[T]]) -> Dict[str, T]:
    if not isinstance(dict_, Dict):
        dict_ = list_to_index_dict(dict_)
    return dict_


def list_to_index_dict(list_: Iterable[T]) -> Dict[str, T]:
    return {str(idx): item for idx, item in enumerate(list_)}


def is_int_array(array: np.array, eps: float = 1e-8):
    return np.all(np.abs(array - array.astype(np.int32)) < eps)
