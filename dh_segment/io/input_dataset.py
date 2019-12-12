#!/usr/bin/env python
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from dh_segment.utils.params_config import TrainingParams
from .transforms import SampleLoad


class InputCSVDataset(Dataset):

    def __init__(self,
                 csv_filename: str,
                 parameters: TrainingParams,
                 transform = None):

        self.dataframe = pd.read_csv(csv_filename, header=None, names=['images', 'labels'])
        self.parameters = parameters
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
        transform_load = SampleLoad()
        sample = transform_load(sample)

        # todo: Assign color to class id
        # call label_image_to_class and multilabel_image_to_class

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