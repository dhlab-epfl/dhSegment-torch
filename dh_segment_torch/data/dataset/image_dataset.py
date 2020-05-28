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


class ImageDataset(Dataset):
    def __init__(self, data: pd.DataFrame, base_dir: Optional[str] = None, compose_transform: Compose = None, add_shapes: bool = True):
        super().__init__(data, base_dir)
        self.compose_transform = compose_transform
        if self.compose_transform:
            self.compose_transform.add_targets({"label": "mask"})
        self.add_shapes = add_shapes

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

        if self.compose_transform:
            sample = self.compose_transform(**sample)
        return sample_to_tensor(sample, self.add_shapes)

    @classmethod
    def from_csv(cls, csv_filename: Union[str, Path], base_dir: Union[str, Path] = None, compose_transform: Compose = None, add_shapes: bool = True):
        data = load_data_from_csv(str(csv_filename))
        return cls(data, str(base_dir), compose_transform, add_shapes)

    @classmethod
    def from_csv_list(cls, csv_list: List[Union[str, Path]], base_dir: Union[str, Path] = None, compose_transform: Compose = None, add_shapes: bool = True):
        data = load_data_from_csv_list([str(csv) for csv in csv_list])
        return cls(data, str(base_dir), compose_transform, add_shapes)

    @classmethod
    def from_folder(cls, folder: Union[str, Path], compose_transform: Compose = None, add_shapes: bool = True):
        data = load_data_from_folder(str(folder))
        return cls(data, compose_transform=compose_transform, add_shapes=add_shapes)


Dataset.register("image_dataframe")(ImageDataset)
Dataset.register("image_csv", "from_csv")(ImageDataset)
Dataset.register("image_csv_list", "from_csv_list")(ImageDataset)
Dataset.register("image_folder", "from_folder")(ImageDataset)
