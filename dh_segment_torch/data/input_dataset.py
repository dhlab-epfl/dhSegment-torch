#!/usr/bin/env python
import logging
import os
from abc import ABC
from glob import glob
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as tsfm


class InputDataset(Dataset, ABC):
    """Abstract class for Dataset

    :ivar dataframe: a DataFrame containing the images and labels filenames (``image`` and ``labels``)
    :vartype dataframe: pandas.DataFrame
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a torchvision.transforms.Compose
    :vartype transform: torchvision.transforms.Compose
    """
    def __init__(self,
                 dataframe: pd.DataFrame,
                 transform: tsfm.Compose = None):
        self.dataframe = dataframe
        self.transform = transform
        self.check_filenames_exist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self,
                    idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'image': self.dataframe.images.iloc[idx],
                  'label': self.dataframe.labels.iloc[idx]}
        # Load image
        sample = load_sample(sample)

        if self.transform:
            sample = self.transform(sample)

        return sample

    def check_filenames_exist(self):
        # Checks that all image files can be found
        for img_filename in list(self.dataframe.images.values):
            if not os.path.exists(img_filename):
                raise FileNotFoundError(img_filename)

        for label_filename in list(self.dataframe.labels.values):
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
    def __init__(self,
                 csv_filename: str,
                 transform: tsfm.Compose = None):
        super().__init__(dataframe=pd.read_csv(csv_filename, header=None, names=['images', 'labels']),
                         transform=transform)


class InputFolderDataset(InputDataset):
    """Class for Dataset using an input folder.

    :ivar folder: folder containing ``image`` and ``label`` directories.
    :vartype folder: str
    :ivar transform: the transforms to apply to the image and labels. it should be a list of transforms
    wrapped in a torchvision.transforms.Compose
    :vartype transform: torchvision.transforms.Compose
    """
    def __init__(self,
                 folder: str, 
                 transform: tsfm.Compose = None):
        self.image_dir = os.path.join(folder, 'images')
        self.labels_dir = os.path.join(folder, 'labels')
        self.check_dir_exist()

        input_data = self.compose_input_data()
        super().__init__(dataframe=pd.DataFrame(data=input_data, columns=['images', 'labels']),
                         transform=transform)

    def check_dir_exist(self):
        assert os.path.isdir(self.image_dir), f"Dataset creation: {self.image_dir} not found."
        assert os.path.isdir(self.labels_dir), f"Dataset creation: {self.labels_dir} not found."
        
    def compose_input_data(self):
        image_extensions = ['.jpg', '.png', '.jpeg', '.JPG', '.tif', '.TIF']
        image_files = list()
        for ext in image_extensions:
            image_files.extend(glob(os.path.join(self.image_dir, '**', ext), recursive=True))

        input_data = list()
        for image_filename in image_files:
            basename = os.path.basename(image_filename).split('.')[0]
            label_filename_candidates = glob(os.path.join(self.labels_dir, basename + '.*'))

            if len(label_filename_candidates) == 0:
                logging.error(f"Did not found the corresponding label image of {image_filename} "
                              f"in {self.labels_dir} directory")
                continue
            elif len(label_filename_candidates) > 1:
                logging.warning(f"Found more than 1 label match for {image_filename}. "
                                f"Taking the first one {label_filename_candidates[0]}")
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

    def __init__(self,
                 csv_list = List[str],
                 transform: tsfm.Compose = None):

        list_dataframes = list()
        for csv_file in csv_list:
            assert os.path.isfile(csv_file), f"{csv_file} does not exist"
            list_dataframes.append(pd.read_csv(csv_file, header=None, names=['images', 'labels']))

        dataframe = pd.concat(list_dataframes, axis=0)
        super().__init__(dataframe=dataframe,
                         transform=transform)




def load_sample(sample: dict) -> dict:
    """
    Loads the image and the label image. Returns the updated dictionary.

    :param sample: dictionary containing at least ``image`` and ``label`` keys.
    """
    image_filename, label_filename = sample['image'], sample['label']

    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label image
    label_image = cv2.imread(label_filename)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    sample.update({'image': image, 'label': label_image, 'shape': image.shape[:2]})
    return sample


def get_dataset(data: str, transform: tsfm.Compose = None):
    if os.path.isdir(data):
        return InputFolderDataset(data, transform)
    elif check_csv_file(data):
        return InputCSVDataset(data, transform=transform)
    elif isinstance(data, list) and len(data) > 0 and all([check_csv_file(path) for path in data]):
        return InputListCSVDataset(data, transform=transform)
    else:
        raise TypeError(f'input_data {data} is neither a directory nor a csv file')

def check_csv_file(path):
    return os.path.isfile(path) and path.endswith('csv')


# TODO paddings center ?
def compute_paddings(heights, widths):
    max_height = np.max(heights)
    max_width = np.max(widths)

    paddings_height = max_height - heights
    paddings_width = max_width - widths
    paddings_zeros = np.zeros(len(heights), dtype=int)

    paddings = np.stack([paddings_zeros, paddings_width, paddings_zeros, paddings_height]).T
    return list(map(tuple, paddings))


def collate_fn(examples):
    if not isinstance(examples, list):
        examples = [examples]
    heights = np.array([x['shape'][0] for x in examples])
    widths = np.array([x['shape'][1] for x in examples])
    paddings = compute_paddings(heights, widths)
    images = []
    masks = []
    shapes_out = []
    for example, padding in zip(examples, paddings):

        image, label, shape = example['image'], example['label'], example['shape']
        images.append(F.pad(image, padding))
        masks.append(F.pad(label, padding))
        shapes_out.append(shape)

    return {'images': torch.stack(images, dim=0),
            'labels': torch.stack(masks, dim=0),
            'shapes': torch.stack(shapes_out, dim=0)
            }


def patches_worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset_size = len(dataset.dataframe)
    num_workers = worker_info.num_workers
    worker_id = worker_info.id

    if num_workers > dataset_size:
        start = worker_id % dataset_size
        end = start + 1
    else:
        items_per_worker = dataset_size // num_workers
        start = worker_id * items_per_worker
        end = min(start + items_per_worker, dataset_size)
    dataset.dataframe = dataset.dataframe.iloc[start:end]