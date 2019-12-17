#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
from torchvision import transforms as tsfm
import pandas as pd
import os
import cv2
from dh_segment.utils.params_config import TrainingParams


class InputDataset(Dataset):

    def __init__(self):
        self.dataframe = None
        self.transform = None

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


class InputCSVDataset(Dataset):

    def __init__(self,
                 csv_filename: str,
                 parameters: TrainingParams,
                 transform: tsfm.Compose = None):

        self.dataframe = pd.read_csv(csv_filename, header=None, names=['images', 'labels'])
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


class InputFolderDataset(Dataset):

    def __init__(self):
        pass

    def __call__(self):
        pass


class InputListCSVDataset(Dataset):

    def __init__(self):
        pass

    def __call__(self):
        pass


def load_sample(sample: dict) -> dict:
    image_filename, label_filename = sample['image'], sample['label']

    image = cv2.imread(image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Load label image
    label_image = cv2.imread(label_filename)
    label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

    sample.update({'image': image, 'label': label_image, 'shape': image.shape[:2]})
    return sample


