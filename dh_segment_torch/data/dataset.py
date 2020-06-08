import logging
import mimetypes
import os
from abc import ABC
from glob import glob
from typing import List, Optional, Dict, Any

import cv2
import pandas as pd
import torch
from torchvision.transforms.functional import to_tensor

from dh_segment_torch.config.registrable import Registrable

mimetypes.init()


class Dataset(torch.utils.data.Dataset, Registrable, ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        base_dir: Optional[str] = None,
        repeat_dataset: int = 1,
    ):
        self.data = data.applymap(lambda path: path.strip())
        if base_dir is not None:
            self.data = self.data.applymap(lambda path: os.path.join(base_dir, path))
        self.check_filenames_exist()
        if repeat_dataset < 1:
            raise ValueError(
                f"Repeat dataset cannot be smaller than 1, got {repeat_dataset}"
            )
        self.data = self.data.loc[self.data.index.repeat(repeat_dataset)].copy()

    @property
    def num_images(self):
        return len(self.data)

    def check_filenames_exist(self):

        # Checks that all image files can be found
        for img_filename in list(self.data.image.values):
            if not os.path.exists(img_filename):
                raise FileNotFoundError(img_filename)

        for label_filename in list(self.data.label.values):
            if not os.path.exists(label_filename):
                raise FileNotFoundError(label_filename)


def load_data_from_csv(csv_filename: str):
    return pd.read_csv(csv_filename, header=None, names=["image", "label"])


def load_data_from_csv_list(csv_list: List[str]):
    list_dataframes = list()
    for csv_file in csv_list:
        assert os.path.isfile(csv_file), f"{csv_file} does not exist"
        list_dataframes.append(
            pd.read_csv(csv_file, header=None, names=["image", "label"])
        )

    return pd.concat(list_dataframes, axis=0)


def load_data_from_folder(folder: str):
    image_dir = os.path.join(folder, "images")
    labels_dir = os.path.join(folder, "labels")
    check_dirs_exist(image_dir, labels_dir)
    input_data = compose_input_data(image_dir, labels_dir)
    return pd.DataFrame(data=input_data, columns=["image", "label"])


def load_sample(sample: dict) -> dict:
    """
    Loads the image and the label image. Returns the updated dictionary.

    :param sample: dictionary containing at least ``image`` and ``label`` keys.
    """
    image_filename, label_filename = sample["image"], sample["label"]

    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label image
    label_image = cv2.imread(label_filename)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    sample.update({"image": image, "label": label_image})
    return sample


def sample_to_tensor(sample: Dict[str, Any]):
    image, label = sample["image"], sample["label"]

    # If we have multilabel, we need to transpose
    if label.ndim == 3:
        label = label.transpose((2, 0, 1))

    sample.update(
        {
            "image": to_tensor(image),
            "label": torch.from_numpy(label),
            "shape": torch.tensor(image.shape[:2]),
        }
    )
    return sample


def get_image_exts():
    image_exts = [
        ext for ext, app in mimetypes.types_map.items() if app.startswith("image")
    ]
    image_exts = image_exts + [ext.upper() for ext in image_exts]
    return image_exts


def check_dirs_exist(image_dir: str, labels_dir: str):
    assert os.path.isdir(image_dir), f"Dataset creation: {image_dir} not found."
    assert os.path.isdir(labels_dir), f"Dataset creation: {labels_dir} not found."


def compose_input_data(image_dir: str, labels_dir: str):
    image_extensions = get_image_exts()
    image_files = list()
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(image_dir, f"*{ext}"), recursive=True))
    input_data = list()
    for image_filename in image_files:
        basename = ".".join(os.path.basename(image_filename).split(".")[:-1])
        label_filename_candidates = glob(os.path.join(labels_dir, basename + ".*"))

        if len(label_filename_candidates) == 0:
            logging.error(
                f"Did not found the corresponding label image of {image_filename} "
                f"in {labels_dir} directory"
            )
            continue
        elif len(label_filename_candidates) > 1:
            logging.warning(
                f"Found more than 1 label match for {image_filename}. "
                f"Taking the first one {label_filename_candidates[0]}"
            )

        label_filename = label_filename_candidates[0]

        input_data.append((image_filename, label_filename))
    return input_data
