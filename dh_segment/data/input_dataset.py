#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tsfm
from typing import List
import pandas as pd
from glob import glob
import os
import cv2
import logging
from abc import ABC


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
