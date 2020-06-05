import re
from datetime import datetime
from itertools import islice
from typing import Iterable, List, Tuple, Union, Dict, Any, Optional, TypeVar, Sequence, Iterator

import torch


def cut_with_padding(
    input_tensor: torch.Tensor, shape: torch.Tensor, margin: int = 0
) -> torch.Tensor:
    return input_tensor[
        ..., margin: shape[0].item() - margin, margin: shape[1].item() - margin
    ]


def detach_and_move_tensors(
    *tensors: torch.Tensor, device: Optional[str] = None
) -> Iterable[torch.Tensor]:
    return (
        tensor.detach().to(device if device else tensor.device)
        if isinstance(tensor, torch.Tensor)
        else tensor
        for tensor in tensors
    )


def make_params_groups(
    model_params: List[Tuple[str, torch.nn.Parameter]],
    param_groups: List[Tuple[Union[str, List[str]], Dict[str, Any]]] = None,
) -> Union[List[torch.nn.Parameter], List[Dict[str, Any]]]:
    if param_groups is None:
        new_params_groups: List[torch.nn.Parameter] = [
            param for _, param in model_params
        ]
    else:
        new_params_groups: List[Dict[str, Any]] = [
            {"params": list()} for _ in range(len(param_groups) + 1)
        ]
        for param_name, param in model_params:
            matched_group_index = -1
            matched_kwargs = {}
            for group_index, (regexes, kwargs) in enumerate(param_groups):
                if isinstance(regexes, str):
                    regexes = [regexes]
                for regex in regexes:
                    if re.search(regex, param_name):
                        if (
                            matched_group_index != -1
                            and matched_group_index != group_index
                        ):
                            raise ValueError(
                                f"Parameter {param_name} was matched in two groups"
                            )
                        matched_group_index = group_index
                        matched_kwargs = kwargs
            new_params_groups[matched_group_index]["params"].append(param)
            new_params_groups[matched_group_index].update(**matched_kwargs)
    return new_params_groups


T = TypeVar("T")


def batch_items(items: Iterable[T], batch_size: int = 1) -> Iterator[T]:
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if len(batch) > 0:
            yield batch
        else:
            break


def move_batch(batch: Dict[str, torch.Tensor], device: str, non_blocking: bool = True) -> Dict[str, torch.Tensor]:
    return {k: t.to(device, non_blocking=non_blocking) for k, t in batch.items()}


def move_and_detach_batch(batch: Dict[str, torch.Tensor], device: str, non_blocking: bool = True) -> Dict[str, torch.Tensor]:
    return {k: t.to(device, non_blocking=non_blocking).detach() for k, t in batch.items()}


def join_not_none(*items: Optional[str], join_str: str = "_"):
    return join_str.join([item for item in items if item and len(item) > 0])


def format_time(timestamp: float) -> str:
    timestamp = datetime.fromtimestamp(timestamp)
    return (
        f"{timestamp.year:04d}-{timestamp.month:02d}-{timestamp.day:02d}"
        f"{timestamp.hour:02d}-{timestamp.minute:02d}-{timestamp.second:02d}"
    )


def should_run(iteration: int, every: int):
    return iteration >= every and iteration % every == 0