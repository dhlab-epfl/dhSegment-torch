#!/usr/bin/env python
__author__ = "solivr"
__license__ = "GPL"

import cv2
import math
import torch
from torchvision import transforms
from ..utils.params_config import TrainingParams


class LoadSample(object):
    """
    todo: doc
    """
    def __call__(self,
                 sample: dict):
        image_filename, label_filename = sample['image'], sample['label']

        image = cv2.imread(image_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load label image
        label_image = cv2.imread(label_filename)
        label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)

        return {'image': image, 'label': label_image}


class CustomResize(object):
    """
    Resize according to number of pixels and keeps the same ratio for sample (image, label)
    """
    def __init__(self,
                 output_size: int):

        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self,
                  sample: dict):
        """

        :param sample:
        :return:
        """
        image, label_image = sample['image'], sample['label']

        # compute new size
        input_shape = image.shape
        # We want X/Y = x/y and we have size = x*y so :
        ratio = input_shape[1] / input_shape[0]
        new_height = int(math.sqrt(self.output_size / ratio))
        new_width = int(self.output_size / new_height)

        resized_image = cv2.resize(image, dsize=[new_width, new_height], interpolation=cv2.INTER_LINEAR)
        resized_label = cv2.resize(label_image, dsize=[new_width, new_height], interpolation=cv2.INTER_NEAREST)

        return {'image': resized_image, 'label': resized_label}


class SampleColorJitter(transforms.ColorJitter):
    """
    Wrapper for ``transforms.ColorJitter`` to use sample {image, label} as input and output
    """
    def __call__(self,
                 sample: dict):
        """
        todo: doc
        :param sample:
        :return:
        """
        image, label = sample['image'], sample['label']

        transform = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        return {'image': transform(image), 'label': label}


class SampleRandomVerticalFlip(object):
    """
    Wrapper for ``transforms.RandomVerticalFlip`` to use sample {image, label} as input and output
    """
    def __init__(self,
                 p: float = 0.5):
        self.p = p

    def __call__(self,
                 sample: dict):
        """
        todo: doc
        :param sample:
        :return:
        """
        image, label = sample['image'], sample['label']
        transform = transforms.RandomVerticalFlip(self.p)

        return {'image': transform(image), 'label': transform(label)}


class SampleRandomHorizontalFlip(object):
    """
    Wrapper for ``transforms.RandomVerticalFlip`` to use sample {image, label} as input and output
    """
    def __init__(self,
                 p: float = 0.5):
        self.p = p

    def __call__(self,
                 sample: dict):
        """
        todo: doc
        :param sample:
        :return:
        """
        image, label = sample['image'], sample['label']
        transform = transforms.RandomHorizontalFlip(self.p)

        return {'image': transform(image), 'label': transform(label)}


class ToTensor(object):
    """
    Convert ndarrays to Tensors for sample (image, label).
    """
    def __call__(self,
                 sample: dict):
        """

        :param sample:
        :return:
        """
        image, label = sample['image'], sample['label']

        # swap color axis to C x H x W
        image = image.transpose((2, 0, 1))

        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}


def make_transforms(parameters: TrainingParams):

    transform_list = list()

    # todo : scaling
    if parameters.data_augmentation_max_scaling > 0:
        pass

    # resize
    transform_list.append(CustomResize(parameters.input_resized_size))

    # todo: rotation
    if parameters.data_augmentation_max_rotation > 0:
        pass

    # todo: make patches
    if parameters.make_patches:
        pass

    if parameters.data_augmentation_flip_lr:
        transform_list.append(transforms.SampleRandomHorizontalFlip())

    if parameters.data_augmentation_flip_ud:
        transform_list.append(transforms.SampleRandomVerticalFlip())

    if parameters.data_augmentation_color:
        transform_list.append(SampleColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5))

    # to tensor
    transform_list.append(ToTensor())

    return transforms.Compose(transform_list)