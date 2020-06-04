import math
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from skimage.util import view_as_windows
from torch.utils.data.dataloader import get_worker_info

from dh_segment_torch.data.dataset.dataset import (
    Dataset,
    load_sample,
    sample_to_tensor,
    load_data_from_csv,
    load_data_from_csv_list,
    load_data_from_folder,
)
from dh_segment_torch.data.transform.albumentation import Compose
from dh_segment_torch.data.transform.assign_labels import Assign
from dh_segment_torch.utils.ops import batch_items


class PatchesDataset(IterableDataset, Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[str] = None,
        pre_patches_compose: Compose = None,
        post_patches_compose: Compose = None,
        assign_transform: Assign = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        patches_overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None,
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
    ):
        if repeat_dataset < 1:
            raise ValueError("Repeat dataset cannot be smaller than 1")

        data = data.loc[data.index.repeat(repeat_dataset)].copy()
        super().__init__(data, base_dir)

        self.pre_patches_compose = pre_patches_compose
        self.post_patches_compose = post_patches_compose

        if self.pre_patches_compose:
            self.pre_patches_compose.add_targets({"label": "mask"})
        if self.post_patches_compose:
            self.post_patches_compose.add_targets({"label": "mask"})

        self.assign_transform = assign_transform

        if isinstance(patches_size, int):
            patches_size = (patches_size, patches_size)

        self.patch_size = patches_size
        self.patches_overlap = patches_overlap
        self.shuffle = shuffle
        self.prefetch_shuffle = prefetch_shuffle
        self.drop_last = drop_last
        self.repeat_dataset = repeat_dataset
        self.offsets_augment = offsets_augment

    def __iter__(self):
        data = self.get_data_with_worker_info()
        if self.shuffle:
            shuffled_data = data.sample(frac=1)
        else:
            shuffled_data = data

        for samples in batch_items(
            shuffled_data[["image", "label"]].values, self.prefetch_shuffle
        ):
            samples = [
                load_sample({"image": image, "label": label})
                for image, label in samples
            ]

            if self.pre_patches_compose:
                samples = [self.pre_patches_compose(**sample) for sample in samples]

            if self.assign_transform:
                for sample in samples:
                    label = self.assign_transform.first_phase(sample["label"])
                    sample.update({"label": label})
            samples = [
                sample_to_patche_samples(
                    sample,
                    self.patch_size,
                    self.patches_overlap,
                    offset_augment=self.offsets_augment,
                )
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
                label = samples[patch_idx][1][row, col]

                sample = {"image": image, "label": label}
                if self.post_patches_compose:
                    sample = self.post_patches_compose(**sample)

                if self.assign_transform:
                    label = self.assign_transform.second_phase(sample["label"])
                    sample.update({"label": label})

                yield sample_to_tensor(sample)
            del samples

    def get_data_with_worker_info(self):
        worker_info = get_worker_info()
        if worker_info is None:
            return self.data
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            if num_workers > self.num_images:
                raise ValueError(
                    "Cannot have a number of worker larger than the number of images,"
                    "it will results in duplicates."
                )
            images_per_worker = int(math.ceil(self.num_images / float(num_workers)))
            start = worker_id * images_per_worker
            end = min(start + images_per_worker, self.num_images)
            return self.data.iloc[start:end]

    def set_shuffle(self, shuffle: bool):
        self.shuffle = shuffle

    @classmethod
    def from_csv(
        cls,
        csv_filename: Union[str, Path],
        base_dir: Union[str, Path] = None,
        pre_patches_compose: Compose = None,
        post_patches_compose: Compose = None,
        assign_transform: Assign = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        patches_overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None,
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
    ):
        data = load_data_from_csv(str(csv_filename))
        return cls(
            data,
            str(base_dir),
            pre_patches_compose,
            post_patches_compose,
            assign_transform,
            patches_size,
            patches_overlap,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
        )

    @classmethod
    def from_csv_list(
        cls,
        csv_list: List[Union[str, Path]],
        base_dir: Union[str, Path] = None,
        pre_patches_compose: Compose = None,
        post_patches_compose: Compose = None,
        assign_transform: Assign = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        patches_overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None,
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
    ):
        data = load_data_from_csv_list([str(csv) for csv in csv_list])
        return cls(
            data,
            str(base_dir),
            pre_patches_compose,
            post_patches_compose,
            assign_transform,
            patches_size,
            patches_overlap,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
        )

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        pre_patches_compose: Compose = None,
        post_patches_compose: Compose = None,
        assign_transform: Assign = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        patches_overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None,
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
    ):
        data = load_data_from_folder(str(folder))
        return cls(
            data,
            None,
            pre_patches_compose,
            post_patches_compose,
            assign_transform,
            patches_size,
            patches_overlap,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
        )


Dataset.register("patches_dataframe")(PatchesDataset)
Dataset.register("patches_csv", "from_csv")(PatchesDataset)
Dataset.register("patches_csv_list", "from_csv_list")(PatchesDataset)
Dataset.register("patches_folder", "from_folder")(PatchesDataset)


def sample_to_patche_samples(
    sample: Dict[str, np.ndarray],
    patch_shape: Tuple[int, int],
    overlap:  Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None,
    offset_augment: bool = False,
):

    if offset_augment:
        offsets = np.random.rand(2)
        h, w = sample["image"].shape[:2]

        offset_h = round(offsets[0] * h // 2)
        offset_w = round(offsets[1] * w // 2)
    else:
        offset_h = 0
        offset_w = 0

    images = extract_patches(
        sample["image"][offset_h:, offset_w:], patch_shape, overlap
    )
    labels = extract_patches(
        sample["label"][offset_h:, offset_w:], patch_shape, overlap
    )

    return images, labels


def extract_patches(
    image: np.ndarray,
    patch_shape: Tuple[int, int] = (300, 300),
        overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ] = None
):
    if len(image.shape) > 3:
        raise ValueError("Expected single image")

    patch_h, patch_w = patch_shape

    stride_h, stride_w = normalize_overlap(overlap)

    window_shape = (
        (patch_h, patch_w, image.shape[2]) if image.ndim == 3 else (patch_h, patch_w)
    )
    step = (stride_h, stride_w, 1) if image.ndim == 3 else (stride_h, stride_w)
    patches = view_as_windows(image, window_shape, step)
    # patches = patches.reshape(-1, patch_h, patch_w, image.shape[2])
    return patches


def normalize_overlap(overlap: Union[
            Optional[int],
            Optional[float],
            Tuple[Optional[Union[float, int]], Optional[Union[float, int]]],
        ], patch_shape: int) -> Tuple[int, int]:

    if isinstance(overlap, Tuple):
        return normalize_single_overlap(overlap[0], patch_shape), normalize_single_overlap(overlap[1], patch_shape)
    else:
        overlap = normalize_single_overlap(overlap, patch_shape)
        return overlap, overlap


def normalize_single_overlap(overlap: Optional[Union[int, float]], patch_shape: int) -> int:
    if overlap is None:
        return patch_shape // 2
    elif isinstance(overlap, int):
        return overlap
    elif isinstance(overlap, float):
        if 0 < overlap < 1:
            return round(patch_shape * overlap)
        else:
            return round(overlap)
    else:
        raise ValueError(f"Stride was not None, float, int, but {type(overlap)}")