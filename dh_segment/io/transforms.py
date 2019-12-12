#!/usr/bin/env python
import cv2
import math
from PIL import Image
import numpy as np
from typing import Tuple
import torch
from torchvision import transforms
from ..utils.params_config import TrainingParams


class SampleLoad(object):
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

        sample.update({'image': image, 'label': label_image, 'shape': image.shape[:2]})
        return sample


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

        sample.update({'image': resized_image, 'label': resized_label, 'shape': resized_image.shape[:2]})
        return sample


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

        sample.update({'image': transform(image)})
        return sample


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

        sample.update({'image': transform(image), 'label': transform(label)})
        return sample


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

        sample.update({'image': transform(image), 'label': transform(label)})
        return sample


class SampleStandardRandomRotation(transforms.RandomRotation):
    """
    todo: doc
    """
    def __call__(self,
                 sample: dict):
        image, label = sample['image'], sample['label']

        angle = self.get_params(self.degrees)

        rotated_image = transforms.functional.rotate(image, angle, self.resample, self.expand, self.center, self.fill)
        rotated_label = transforms.functional.rotate(label, angle, Image.NEAREST, self.expand, self.center, self.fill)

        sample.update({'image': rotated_image, 'label': rotated_label, 'shape': rotated_image.shape[:2]})
        return sample


class SampleCroppedRandomRotation(object):
    """
    todo: doc
    """
    def __init__(self):
        pass

    def __call__(self,
                 sample: dict):
        # todo
        pass


class SamplePatcher(object):
    """
    todo: doc
    """
    def __init__(self,
                 patch_shape: Tuple[int]):
        self.patch_shape = patch_shape

    def __call__(self,
                 sample: dict):
        pass


class SampleToTensor(object):
    """
    Convert ndarrays to Tensors for sample (image, label).
    """
    def __call__(self,
                 sample: dict):
        """

        :param sample:
        :return:
        """
        image, label, shape = sample['image'], sample['label'], sample['shape']

        # swap color axis to C x H x W
        image = image.transpose((2, 0, 1))
        label = label.transpose((2, 0, 1))

        sample.update({'image': torch.from_numpy(image),
                       'label': torch.from_numpy(label),
                       'shape': torch.from_numpy(shape)})
        return sample


def make_transforms(parameters: TrainingParams):
    """
    todo: doc
    :param parameters:
    :return:
    """

    transform_list = list()

    if parameters.data_augmentation_max_scaling > 0:
        scaled_size = parameters.data_augmentation_max_scaling * parameters.input_resized_size

        resized_size = np.maximum(parameters.minimum_input_size, scaled_size)
        resized_size = np.minimum(parameters.maximum_input_size, resized_size)
    else:
        resized_size = parameters.input_resized_size

    # resize
    transform_list.append(CustomResize(resized_size))

    # todo: cropped rotation
    if parameters.data_augmentation_max_rotation > 0:
        transform_list.append(SampleStandardRandomRotation())

    # todo: make patches
    if parameters.make_patches:
        raise NotImplementedError
        # transform_list.append(SamplePatcher())

    if parameters.data_augmentation_flip_lr:
        transform_list.append(transforms.SampleRandomHorizontalFlip())

    if parameters.data_augmentation_flip_ud:
        transform_list.append(transforms.SampleRandomVerticalFlip())

    if parameters.data_augmentation_color:
        transform_list.append(SampleColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5))

    # to tensor
    transform_list.append(SampleToTensor())

    return transforms.Compose(transform_list)