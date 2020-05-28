import os
from abc import ABC
from random import random
from typing import List, Sequence, Iterator, Tuple

import albumentations
import pandas as pd
import torch
import torchvision.transforms as tsfm
from skimage.util import view_as_windows
from torch.utils.data.dataset import IterableDataset

from .dataset import (
    InputFolderDataset,
    InputListCSVDataset,
    load_sample,
    sample_to_tensor,
)


class PatchesDataset(IterableDataset, ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        pre_transform: albumentations.Compose = None,
        post_transform: albumentations.Compose = None,
        patch_size=(300, 300),
        shuffle=False,
        prefetch_shuffle=5,
        drop_last=False,
        repeat_dataset=1,
        offsets_augment=False
    ):
        if repeat_dataset < 1:
            raise ValueError("Repeat dataset cannot be smaller than 1")
        self.dataframe = dataframe.loc[dataframe.index.repeat(repeat_dataset)].copy()
        self.pre_transform = pre_transform
        self.post_transform = post_transform

        if self.pre_transform:
            self.pre_transform.add_targets({"label": "mask"})
        if self.post_transform:
            self.post_transform.add_targets({"label": "mask"})

        self.patch_size = patch_size
        self.shuffle = shuffle
        self.prefetch_shuffle = prefetch_shuffle
        self.drop_last = drop_last
        self.repeat_dataset = repeat_dataset
        self.offsets_augment = offsets_augment

    def __iter__(self):

        if self.shuffle:
            shuffled_dataframe = self.dataframe.sample(frac=1)
        else:
            shuffled_dataframe = self.dataframe

        for samples in batch_items(
            shuffled_dataframe[["image", "label"]].values, self.prefetch_shuffle
        ):
            samples = [
                load_sample({"image": image, "label": label})
                for image, label in samples
            ]

            if self.pre_transform is not None:
                samples = [self.pre_transform(**sample) for sample in samples]

            samples = [
                sample_to_patche_samples(sample, self.patch_size, offset_augment=self.offsets_augment)
                for sample in samples
            ]

            paths = []
            idx = 0
            for patch_idx, sample in enumerate(samples):
                rows, cols = sample[0].shape[:2]
                for row in range(rows):
                    for col in range(cols):
                        paths.append((patch_idx, row, col))
                        idx += 1

            if self.shuffle:
                indices = torch.randperm(len(paths))
            else:
                indices = range(len(paths))
            for idx in indices:
                patch_idx, row, col = paths[idx]
                image = samples[patch_idx][0][row, col, 0]
                label = samples[patch_idx][1][row, col, 0]
                if self.post_transform is not None:
                    sample = self.post_transform(image=image, label=label)
                else:
                    sample = {'image': image, 'label': label}
                yield sample_to_tensor(sample)
            del samples

    @property
    def num_images(self):
        return len(self.dataframe)


def batch_items(iterable: Sequence, batch_size: int = 1) -> Iterator:
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


class PatchesCSVDataset(PatchesDataset):
    def __init__(self, csv_filename: str, *args, **kwargs):
        super().__init__(
            pd.read_csv(csv_filename, header=None, names=["image", "label"]),
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


def sample_to_patche_samples(sample, patch_shape: Tuple[int], overlap=(None, None), offset_augment=False):

    if offset_augment:
        offsets = (random.uniform(0, 1), random.uniform(0, 1))
        h, w = sample['image'].shape[:2]

        offset_h = round(offsets[0] * h // 2)
        offset_w = round(offsets[1] * w // 2)
    else:
        offset_h = 0
        offset_w = 0

    images = extract_patches(sample["image"][offset_h:, offset_w:], patch_shape, overlap)
    labels = extract_patches(sample["label"][offset_h:, offset_w:], patch_shape, overlap)

    return images, labels


def extract_patches(image, patch_shape=(300, 300), overlap=(None, None)):
    if len(image.shape) > 3:
        raise ValueError("Expected single image")

    patch_h, patch_w = patch_shape

    stride_h, stride_w = overlap
    if stride_h is None:
        stride_h = patch_h // 2
    if stride_w is None:
        stride_w = patch_w // 2

    window_shape = (patch_h, patch_w, image.shape[2]) if len(image.shape) == 3 else (patch_h, patch_w)
    step = (stride_h, stride_w, 1) if len(image.shape) == 3 else (stride_h, stride_w)
    patches = view_as_windows(image, window_shape, step)
    # patches = patches.reshape(-1, patch_h, patch_w, image.shape[2])
    return patches


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


# def patches_worker_init_fn(worker_id):
#     worker_info = torch.utils.data.get_worker_info()
#     dataset = worker_info.dataset
#     dataset_size = len(dataset.dataframe)
#     num_workers = worker_info.num_workers
#     worker_id = worker_info.id
#
#     # if num_workers < dataset_size:
#     #     """
#     #     Assign worker to image
#     #     Assign slices of quadtree to each worker -> some worker will have small, other will have several
#     #     E.g. 3 workers 1 image as follow: |1|2|
#     #                                       |3|4|
#     #     Then worker 1 get 1, worker 2 get 2, worker 3 get 3 and 4
#     #     If 5 workers, then image is |1 |2 |3 |4 |
#     #                                 |5 |6 |7 |8 |
#     #                                 |9 |10|11|12|
#     #                                 |13|14|15|16|
#     #     Worker 1 gets 1,2,5,6, worker 2 [3,4,7,8], worker 3 [9,10,13,14], worker 4 [11,15], worker 5 [12,16]
#     #     etc.
#     #     """
#     #     worker_per_image = num_workers // dataset_size
#     #     image_id = worker_id % (dataset_size * worker_per_image)
#     #     start = worker_id % dataset_size
#     #     end = start + 1
#     # else:
#     items_per_worker = dataset_size // num_workers
#     start = worker_id * items_per_worker
#     end = min(start + items_per_worker, dataset_size)
#     print(start, end)
#     dataset.dataframe = dataset.dataframe.iloc[start:end]

def patches_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset_size = len(dataset.dataframe)
    num_workers = worker_info.num_workers
    worker_id = worker_info.id

    if dataset_size > num_workers and worker_id >= num_workers - dataset_size % num_workers :
        offset = 1
    else:
        offset = 0

    items_per_worker = max(1, dataset_size // num_workers)
    start = (worker_id * items_per_worker + offset) % (dataset_size)
    end = (start + items_per_worker + offset) % (dataset_size+1)
    dataset.dataframe = dataset.dataframe.iloc[start:end]