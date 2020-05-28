from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, List

import numpy as np
import pandas as pd
import torch
from skimage.util import view_as_windows

from dh_segment_torch.data.dataset.dataset import (
    Dataset,
    load_sample,
    sample_to_tensor,
    load_data_from_csv,
    load_data_from_csv_list,
    load_data_from_folder,
)
from dh_segment_torch.data.transform.albumentation import Compose


class PatchesDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[str] = None,
        pre_compose_transform: Compose = None,
        post_compose_transform: Compose = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
        add_shapes: bool = False,
    ):
        if repeat_dataset < 1:
            raise ValueError("Repeat dataset cannot be smaller than 1")

        data = data.loc[data.index.repeat(repeat_dataset)].copy()
        super().__init__(data, base_dir)

        self.pre_compose_transform = pre_compose_transform
        self.post_compose_transform = post_compose_transform

        if self.pre_compose_transform:
            self.pre_compose_transform.add_targets({"label": "mask"})
        if self.post_compose_transform:
            self.post_compose_transform.add_targets({"label": "mask"})

        if isinstance(patches_size, int):
            patches_size = (patches_size, patches_size)

        self.patch_size = patches_size
        self.shuffle = shuffle
        self.prefetch_shuffle = prefetch_shuffle
        self.drop_last = drop_last
        self.repeat_dataset = repeat_dataset
        self.offsets_augment = offsets_augment
        self.add_shapes = add_shapes

    def __iter__(self):

        if self.shuffle:
            shuffled_data = self.data.sample(frac=1)
        else:
            shuffled_data = self.data

        for samples in batch_items(
            shuffled_data[["image", "label"]].values, self.prefetch_shuffle
        ):
            samples = [
                load_sample({"image": image, "label": label})
                for image, label in samples
            ]

            if self.pre_compose_transform is not None:
                samples = [self.pre_compose_transform(**sample) for sample in samples]

            samples = [
                sample_to_patche_samples(
                    sample, self.patch_size, offset_augment=self.offsets_augment
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
                label = samples[patch_idx][1][row, col, 0]
                if self.post_compose_transform is not None:
                    sample = self.post_compose_transform(image=image, label=label)
                else:
                    sample = {"image": image, "label": label}
                yield sample_to_tensor(sample, self.add_shapes)
            del samples

    @classmethod
    def from_csv(
        cls,
        csv_filename: Union[str, Path],
        base_dir: Union[str, Path] = None,
        pre_compose_transform: Compose = None,
        post_compose_transform: Compose = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
        add_shapes: bool = False,
    ):
        data = load_data_from_csv(str(csv_filename))
        return cls(
            data,
            str(base_dir),
            pre_compose_transform,
            post_compose_transform,
            patches_size,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
            add_shapes
        )

    @classmethod
    def from_csv_list(
        cls,
        csv_list: List[Union[str, Path]],
        base_dir: Union[str, Path] = None,
        pre_compose_transform: Compose = None,
        post_compose_transform: Compose = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
        add_shapes: bool = False,
    ):
        data = load_data_from_csv_list([str(csv) for csv in csv_list])
        return cls(
            data,
            str(base_dir),
            pre_compose_transform,
            post_compose_transform,
            patches_size,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
            add_shapes
        )

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        pre_compose_transform: Compose = None,
        post_compose_transform: Compose = None,
        patches_size: Union[int, Tuple[int, int]] = (300, 300),
        shuffle: bool = False,
        prefetch_shuffle: int = 5,
        drop_last: bool = False,
        repeat_dataset: int = 1,
        offsets_augment: bool = False,
        add_shapes: bool = False,
    ):
        data = load_data_from_folder(str(folder))
        return cls(
            data,
            None,
            pre_compose_transform,
            post_compose_transform,
            patches_size,
            shuffle,
            prefetch_shuffle,
            drop_last,
            repeat_dataset,
            offsets_augment,
            add_shapes
        )


Dataset.register("patches_dataframe")(PatchesDataset)
Dataset.register("patches_csv", "from_csv")(PatchesDataset)
Dataset.register("patches_csv_list", "from_csv_list")(PatchesDataset)
Dataset.register("patches_folder", "from_folder")(PatchesDataset)


def batch_items(iterable: Sequence, batch_size: int = 1) -> Iterator:
    l = len(iterable)
    for ndx in range(0, l, batch_size):
        yield iterable[ndx : min(ndx + batch_size, l)]


def sample_to_patche_samples(
    sample: Dict[str, np.ndarray],
    patch_shape: Tuple[int, int],
    overlap: Tuple[Optional[Union[float, int]], Optional[Union[float, int]]] = (
        None,
        None,
    ),
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
    overlap: Tuple[Optional[Union[float, int]], Optional[Union[float, int]]] = (
        None,
        None,
    ),
):
    if len(image.shape) > 3:
        raise ValueError("Expected single image")

    patch_h, patch_w = patch_shape

    stride_h, stride_w = overlap
    stride_h = normalize_stride(stride_h, patch_h)
    stride_w = normalize_stride(stride_w, patch_w)

    window_shape = (
        (patch_h, patch_w, image.shape[2]) if image.ndim == 3 else (patch_h, patch_w)
    )
    step = (stride_h, stride_w, 1) if image.ndim == 3 else (stride_h, stride_w)
    patches = view_as_windows(image, window_shape, step)
    # patches = patches.reshape(-1, patch_h, patch_w, image.shape[2])
    return patches


def normalize_stride(stride: Optional[Union[float, int]], patch_shape: int):
    if isinstance(stride, int):
        return stride
    elif stride is None:
        return patch_shape // 2
    elif isinstance(stride, float):
        if 0 < stride < 1:
            return round(patch_shape * stride)
        else:
            return round(stride)
    else:
        raise ValueError(f"Stride was not None, float, int, but {type(stride)}")

