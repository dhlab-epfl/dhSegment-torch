from types import ModuleType
from typing import Callable, Optional

from torch.utils import data

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.dataset.utils import collate_fn


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
