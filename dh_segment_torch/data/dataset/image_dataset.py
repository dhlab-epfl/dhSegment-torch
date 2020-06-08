from pathlib import Path
from typing import Union, List, Optional

import pandas as pd
import torch

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


class ImageDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[str] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
    ):
        super().__init__(data, base_dir)
        self.compose = compose
        if self.compose:
            self.compose.add_targets({"label": "mask"})
        self.assign_transform = assign_transform

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "image": self.data.image.iloc[idx],
            "label": self.data.label.iloc[idx],
        }

        sample = load_sample(sample)

        if self.compose:
            sample = self.compose(**sample)
        if self.assign_transform:
            label = sample["label"]
            sample.update({"label": self.assign_transform.apply(label)})
        return sample_to_tensor(sample)

    @classmethod
    def from_csv(
        cls,
        csv_filename: Union[str, Path],
        base_dir: Union[str, Path] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
    ):
        data = load_data_from_csv(str(csv_filename))
        return cls(data, str(base_dir), compose, assign_transform)

    @classmethod
    def from_csv_list(
        cls,
        csv_list: List[Union[str, Path]],
        base_dir: Union[str, Path] = None,
        compose: Compose = None,
        assign_transform: Assign = None,
    ):
        data = load_data_from_csv_list([str(csv) for csv in csv_list])
        return cls(data, str(base_dir), compose, assign_transform)

    @classmethod
    def from_folder(
        cls,
        folder: Union[str, Path],
        compose: Compose = None,
        assign_transform: Assign = None,
    ):
        data = load_data_from_folder(str(folder))
        return cls(data, compose=compose, assign_transform=assign_transform)


Dataset.register("image_dataframe")(ImageDataset)
Dataset.register("image_csv", "from_csv")(ImageDataset)
Dataset.register("image_csv_list", "from_csv_list")(ImageDataset)
Dataset.register("image_folder", "from_folder")(ImageDataset)
