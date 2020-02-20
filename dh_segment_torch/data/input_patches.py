import os
from abc import ABC
from itertools import cycle
from typing import List, Sequence, Iterator

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tsfm
from torch.utils.data.dataset import IterableDataset

from .input_dataset import InputFolderDataset, InputListCSVDataset, load_sample
from .transforms import SampleToPatches, transform_to_several


class PatchesDataset(IterableDataset, ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        pre_transform: tsfm.Compose = None,
        post_transform: tsfm.Compose = None,
        patch_size=(300, 300),
        batch_size=32,
        shuffle=False,
        prefetch_shuffle=5,
        drop_last=False,
        infinite_loop=False,
    ):
        self.dataframe = dataframe
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        self.patch_size = patch_size
        self.shuffle = shuffle
        self.prefetch_shuffle = prefetch_shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.infinite_loop = infinite_loop

    def __iter__(self):
        if self.shuffle:
            shuffled_dataframe = self.dataframe.sample(frac=1)
        else:
            shuffled_dataframe = self.dataframe

        if self.infinite_loop:
            wrapper = cycle
        else:
            wrapper = lambda x: x

        for samples in wrapper(
            batch_items(
                shuffled_dataframe[["images", "labels"]].values, self.prefetch_shuffle
            )
        ):
            samples = [
                load_sample({"image": image, "label": label})
                for image, label in samples
            ]

            if self.pre_transform is not None:
                samples = [self.pre_transform(sample) for sample in samples]

            patch_transform = SampleToPatches(self.patch_size)
            samples = [patch_transform(sample) for sample in samples]

            samples = {
                "images": np.array(
                    [patch for sample in samples for patch in sample["images"]]
                ),
                "labels": np.array(
                    [patch for sample in samples for patch in sample["labels"]]
                ),
                "shapes": np.array(
                    [patch for sample in samples for patch in sample["shapes"]]
                ),
            }

            if self.shuffle:
                indices = torch.randperm(len(samples["images"]))
            else:
                indices = range(len(samples["images"]))

            for perms in batch_items(indices, self.batch_size):
                if self.drop_last:
                    if len(perms) < self.batch_size:
                        continue
                selection = {
                    "images": samples["images"][perms],
                    "labels": samples["labels"][perms],
                    "shapes": samples["shapes"][perms],
                }
                if self.post_transform is not None:
                    selection = transform_to_several(self.post_transform)(selection)
                yield selection
            del samples


def batch_items(iterable: Sequence, batch_size: int = 1) -> Iterator:
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


class PatchesCSVDataset(PatchesDataset):
    def __init__(self, csv_filename: str, *args, **kwargs):
        super().__init__(
            pd.read_csv(csv_filename, header=None, names=["images", "labels"]),
            *args,
            **kwargs,
        )


class PatchesFolderDataset(PatchesDataset):
    def __init__(self, folder: str, *args, **kwargs):
        dataframe = InputFolderDataset(folder).dataframe
        super().__init__(dataframe, *args, **kwargs)


class PatchesListCSVDataset(PatchesDataset):
    def __init__(self, csv_list=List[str], *args, **kwargs):
        dataframe = InputListCSVDataset(csv_list).dataframe
        super().__init__(dataframe, *args, **kwargs)


def get_patches_dataset(
    data: str,
    pre_transform: tsfm.Compose = None,
    post_transform: tsfm.Compose = None,
    **kwargs,
):
    if os.path.isdir(data):
        return PatchesFolderDataset(data, pre_transform, post_transform, **kwargs)
    elif check_csv_file(data):
        return PatchesCSVDataset(data, pre_transform, post_transform, **kwargs)
    elif (
        isinstance(data, list)
        and len(data) > 0
        and all([check_csv_file(path) for path in data])
    ):
        return PatchesListCSVDataset(data, pre_transform, post_transform, **kwargs)
    else:
        raise TypeError(f"input_data {data} is neither a directory nor a csv file")


def check_csv_file(path):
    return os.path.isfile(path) and path.endswith("csv")


def patches_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset_size = len(dataset.dataframe)
    num_workers = worker_info.num_workers
    worker_id = worker_info.id

    # if num_workers < dataset_size:
    #     """
    #     Assign worker to image
    #     Assign slices of quadtree to each worker -> some worker will have small, other will have several
    #     E.g. 3 workers 1 image as follow: |1|2|
    #                                       |3|4|
    #     Then worker 1 get 1, worker 2 get 2, worker 3 get 3 and 4
    #     If 5 workers, then image is |1 |2 |3 |4 |
    #                                 |5 |6 |7 |8 |
    #                                 |9 |10|11|12|
    #                                 |13|14|15|16|
    #     Worker 1 gets 1,2,5,6, worker 2 [3,4,7,8], worker 3 [9,10,13,14], worker 4 [11,15], worker 5 [12,16]
    #     etc.
    #     """
    #     worker_per_image = num_workers // dataset_size
    #     image_id = worker_id % (dataset_size * worker_per_image)
    #     start = worker_id % dataset_size
    #     end = start + 1
    # else:
    items_per_worker = dataset_size // num_workers
    start = worker_id * items_per_worker
    end = min(start + items_per_worker, dataset_size)
    dataset.dataframe = dataset.dataframe.iloc[start:end]
