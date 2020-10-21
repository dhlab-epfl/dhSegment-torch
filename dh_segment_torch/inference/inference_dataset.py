import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Union

import cv2
import torch
import pandas as pd
from dh_segment_torch.data.datasets.dataset import get_image_exts
from torchvision.transforms.functional import to_tensor as np_to_tensor


from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.transforms import Compose


class InferenceDataset(torch.utils.data.Dataset, Registrable):
    default_implementation = "paths"

    def __init__(
        self,
        image_paths: List[str],
        base_dir: Optional[Union[str, Path]] = None,
        pre_processing: Compose = None,
    ):

        if base_dir:
            base_dir = str(base_dir)
            image_paths = [os.path.join(base_dir, img_path) for img_path in image_paths]

        for img_path in image_paths:
            if not os.path.exists(img_path):
                raise FileNotFoundError(img_path)

        self.image_paths = image_paths
        self.pre_processing = pre_processing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = self.image_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.pre_processing:
            image = self.pre_processing(image=image)["image"]
        return {"image": np_to_tensor(image), "shape": torch.tensor(image.shape[:2]), 'path': image_path}

    @classmethod
    def from_folder(
        cls, folder: Union[str, Path], glob_pattern: str = None, pre_processing: Compose = None
    ):
        folder = str(folder)

        image_paths = []
        if glob_pattern:
            image_paths.extend(glob(os.path.join(folder, glob_pattern)))
        else:
            image_extensions = get_image_exts()
            for ext in image_extensions:
                image_paths.extend(glob(os.path.join(folder, f"*{ext}")))
        return cls(image_paths, pre_processing=pre_processing)

    @classmethod
    def from_csv(
        cls,
        csv_path: Union,
        base_dir: Optional[Union[str, Path]] = None,
        pre_processing: Compose = None,
    ):
        csv_path = str(csv_path)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(csv_path)
        image_paths = pd.read_csv(csv_path, usecols=[0], header=None)[0].values.tolist()

        return cls(image_paths, base_dir, pre_processing)


InferenceDataset.register("paths")(InferenceDataset)
InferenceDataset.register("folder", "from_folder")(InferenceDataset)
InferenceDataset.register("csv", "from_csv")(InferenceDataset)
