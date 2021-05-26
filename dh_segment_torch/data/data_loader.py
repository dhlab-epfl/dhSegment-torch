from types import ModuleType
from typing import Callable, Optional, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils import data

from dh_segment_torch.config.registrable import Registrable


def compute_paddings(heights, widths) -> List[List[int]]:
    max_height = np.max(heights)
    max_width = np.max(widths)

    paddings_height = max_height - heights
    paddings_width = max_width - widths
    paddings_zeros = np.zeros(len(heights), dtype=int)

    paddings = np.stack(
        [paddings_zeros, paddings_width, paddings_zeros, paddings_height]
    ).T
    return list(map(list, paddings))


def collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    if not all(["shape" in x for x in examples]):
        for example in examples:
            example["shape"] = torch.tensor(example["image"].shape[1:])

    heights = np.array([x["shape"][0] for x in examples])
    widths = np.array([x["shape"][1] for x in examples])
    paddings = compute_paddings(heights, widths)
    images = []
    masks = []
    shapes_out = []

    for example, padding in zip(examples, paddings):
        image, shape = example["image"], example["shape"]
        images.append(F.pad(image, padding))
        shapes_out.append(shape)

        if "label" in example:
            label = example["label"]
            masks.append(F.pad(label, padding))

    if len(masks) > 0:
        return {
            "input": torch.stack(images, dim=0),
            "target": torch.stack(masks, dim=0),
            "shapes": torch.stack(shapes_out, dim=0),
        }
    else:
        return {
            "input": torch.stack(images, dim=0),
            "shapes": torch.stack(shapes_out, dim=0),
        }


class DataLoader(data.DataLoader, Registrable):
    default_implementation = "default"

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[data.Sampler] = None,
        batch_sampler: Optional[data.Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = collate_fn,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: int = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context: Optional[ModuleType] = None,
    ):
        if isinstance(dataset, data.IterableDataset):
            shuffle = False
            sampler = None
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
        )


DataLoader.register("default")(DataLoader)
