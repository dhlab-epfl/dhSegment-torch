import logging
import mimetypes
import os
from abc import ABC
from glob import glob
from typing import List, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tsfm
from torchvision.transforms.functional import to_tensor
import torch.nn.functional as F


mimetypes.init()


class InputDataset(Dataset, ABC):
    """Abstract class for Dataset

    :ivar dataframe: a DataFrame containing the images and labels filenames (``image`` and ``labels``)
    :vartype dataframe: pandas.DataFrame
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a albumentations.Compose
    :vartype transform: albumentations.Compose
    """

    def __init__(
        self, dataframe: pd.DataFrame, transform: albumentations.Compose = None
    ):
        self.dataframe = dataframe
        self.transform = transform
        if self.transform:
            self.transform.add_targets({"label": "mask"})
        self.check_filenames_exist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            "image": self.dataframe.image.iloc[idx],
            "label": self.dataframe.label.iloc[idx],
        }

        sample = load_sample(sample)

        if self.transform:
            sample = self.transform(**sample)

        return sample_to_tensor(sample)

    def check_filenames_exist(self):
        # Checks that all image files can be found
        for img_filename in list(self.dataframe.image.values):
            if not os.path.exists(img_filename):
                raise FileNotFoundError(img_filename)

        for label_filename in list(self.dataframe.label.values):
            if not os.path.exists(label_filename):
                raise FileNotFoundError(label_filename)


class InputCSVDataset(InputDataset):
    """Class for Dataset using a csv file.

    :ivar csv_filename: filename of csv containinf the tuple (image filename, label filename)
    :vartype csv_filename: str
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a torchvision.transforms.Compose
    :vartype transform: torchvision.transforms.Compose
    """

    def __init__(self, csv_filename: str, transform: albumentations.Compose = None):
        super().__init__(
            dataframe=pd.read_csv(csv_filename, header=None, names=["image", "label"]),
            transform=transform,
        )


class InputFolderDataset(InputDataset):
    """Class for Dataset using an input folder.

    :ivar folder: folder containing ``image`` and ``label`` directories.
    :vartype folder: str
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a torchvision.transforms.Compose
    :vartype transform: torchvision.transforms.Compose
    """

    def __init__(self, folder: str, transform: albumentations.Compose = None):
        self.image_dir = os.path.join(folder, "images")
        self.labels_dir = os.path.join(folder, "labels")
        self.check_dir_exist()

        input_data = self.compose_input_data()
        super().__init__(
            dataframe=pd.DataFrame(data=input_data, columns=["image", "label"]),
            transform=transform,
        )

    def check_dir_exist(self):
        assert os.path.isdir(
            self.image_dir
        ), f"Dataset creation: {self.image_dir} not found."
        assert os.path.isdir(
            self.labels_dir
        ), f"Dataset creation: {self.labels_dir} not found."

    def compose_input_data(self):
        image_extensions = get_image_exts()
        image_files = list()
        for ext in image_extensions:
            image_files.extend(
                glob(os.path.join(self.image_dir, "**", ext), recursive=True)
            )

        input_data = list()
        for image_filename in image_files:
            basename = ".".join(os.path.basename(image_filename).split(".")[:-1])
            label_filename_candidates = glob(
                os.path.join(self.labels_dir, basename + ".*")
            )

            if len(label_filename_candidates) == 0:
                logging.error(
                    f"Did not found the corresponding label image of {image_filename} "
                    f"in {self.labels_dir} directory"
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


class InputListCSVDataset(InputDataset):
    """Class for Dataset using a list of csv files.

    :ivar csv_list: list of csv filenames
    :vartype csv_list: List[str]
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a torchvision.transforms.Compose
    :vartype transform: torchvision.transforms.Compose
    """

    def __init__(self, csv_list=List[str], transform: albumentations.Compose = None):

        list_dataframes = list()
        for csv_file in csv_list:
            assert os.path.isfile(csv_file), f"{csv_file} does not exist"
            list_dataframes.append(
                pd.read_csv(csv_file, header=None, names=["image", "label"])
            )

        dataframe = pd.concat(list_dataframes, axis=0)
        super().__init__(dataframe=dataframe, transform=transform)


def sample_to_tensor(sample):
    image, label = sample["image"], sample["label"]

    sample.update(
        {
            "image": to_tensor(image),
            "label": torch.from_numpy(label).permute((2, 0, 1)),
            "shape": torch.tensor(image.shape[:2]),
        }
    )
    return sample


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


def get_image_exts():
    image_exts = [
        ext for ext, app in mimetypes.types_map.items() if app.startswith("image")
    ]
    image_exts = image_exts + [ext.upper() for ext in image_exts]
    return image_exts


def get_dataset(data: str, transform: albumentations.Compose = None):
    if os.path.isdir(data):
        return InputFolderDataset(data, transform)
    elif check_csv_file(data):
        return InputCSVDataset(data, transform=transform)
    elif (
        isinstance(data, list)
        and len(data) > 0
        and all([check_csv_file(path) for path in data])
    ):
        return InputListCSVDataset(data, transform=transform)
    else:
        raise TypeError(f"input_data {data} is neither a directory nor a csv file")


def check_csv_file(path):
    return os.path.isfile(path) and path.endswith("csv")


# TODO paddings center ?
def compute_paddings(heights, widths):
    max_height = np.max(heights)
    max_width = np.max(widths)

    paddings_height = max_height - heights
    paddings_width = max_width - widths
    paddings_zeros = np.zeros(len(heights), dtype=int)

    paddings = np.stack(
        [paddings_zeros, paddings_width, paddings_zeros, paddings_height]
    ).T
    return list(map(tuple, paddings))


def collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    heights = np.array([x["shape"][0] for x in examples])
    widths = np.array([x["shape"][1] for x in examples])
    paddings = compute_paddings(heights, widths)
    images = []
    masks = []
    shapes_out = []
    for example, padding in zip(examples, paddings):

        image, label, shape = example["image"], example["label"], example["shape"]
        images.append(F.pad(image, padding))
        masks.append(F.pad(label, padding))
        shapes_out.append(shape)

    return {
        "images": torch.stack(images, dim=0),
        "labels": torch.stack(masks, dim=0),
        "shapes": torch.stack(shapes_out, dim=0),
    }
