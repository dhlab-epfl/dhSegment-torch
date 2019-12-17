#!/usr/bin/env python
import cv2
import math
from PIL import Image
import numpy as np
from typing import Tuple
import logging
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from ..utils.params_config import DataParams, PredictionType


class AssignLabelClassification(object):
    """
    todo: doc
    """
    def __init__(self,
                 colors_array: np.ndarray):
        self.colors_array = colors_array

    def __call__(self,
                 sample: dict):

        image = sample['image']
        # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
        diff = image[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]

        return label_image


class AssignLabelMultilabel(object):
    """
    todo: doc
    """
    def __init__(self,
                 colors_array: np.ndarray,
                 code_array: np.ndarray):
        self.colors_array = colors_array
        self.codes_array = code_array

    def __call__(self,
                 sample: dict):
        image = sample['image']

        # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
        if len(image.shape) == 3:
            diff = image[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]
        else:
            raise NotImplementedError('Length is : {}'.format(len(image.shape)))

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]

        label_image = np.take(self.codes_array, label_image, axis=0) > 0  # [H, W, C]
        return label_image.transpose((2, 0, 1)) # [C, H, W]


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


class RandomResize(CustomResize):

    def __init__(self,
                 scaling: float,
                 output_size: int):
        super().__init__(output_size)
        self.range = [int(self.output_size / scaling), int(self.output_size * scaling)]

    def __call__(self,
                 sample: dict):
        self.output_size = np.random.randint(low=self.range[0], high=self.range[1])
        super()(sample)  # todo: verify syntax


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
        image = sample['image']

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


class SampleRandomRotation(object):
    """
    todo: doc
    """
    def __init__(self,
                 max_angle: int,
                 do_crop: bool = False):
        self.angle = max_angle
        self.do_crop = do_crop

    def __call__(self,
                 sample: dict):
        image, label = sample['image'], sample['label']
        rows, columns = image.shape[:2]

        angle = np.random.randint(-self.angle, self.angle)
        rot_matrix = cv2.getRotationMatrix2D(((columns - 1) / 2.0, (rows - 1) / 2.0), angle, 1)

        rotated_image = cv2.warpAffine(image, rot_matrix, (columns, rows), flags=cv2.INTER_LINEAR)
        rotated_label = cv2.warpAffine(label, rot_matrix, (columns, rows), flags=cv2.INTER_NEAREST)

        if self.do_crop:
            # todo: if crop not possible should we return rotated image or original image ?
            border_size = self.compute_border_to_crop(image.shape[:2])
            crop_image = self.crop(rotated_image, border_size)
            crop_label = self.crop(rotated_label, border_size)

            crop_shape = crop_image.shape[:2]

            sample.update({'image': crop_image, 'label': crop_label, 'shape': crop_shape})
            return sample

        sample.update({'image': rotated_image, 'label': rotated_label})
        return sample

    def compute_border_to_crop(self,
                               input_shape: Tuple[int, int]):
        """See https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
        """
        angle = abs(self.angle)
        h, w = input_shape[0], input_shape[1]
        if h > w:
            long_side, short_side = h, w
        else:
            long_side, short_side = w, h

        long_side = (long_side * math.cos(angle) - short_side * math.sin(angle)) / math.cos(2 * angle)
        short_side = (short_side - math.sin(angle) * long_side) / math.cos(angle)
        if h > w:
            h_output, w_output = long_side, short_side
        else:
            h_output, w_output = short_side, long_side

        return math.ceil((h - h_output) / 2), math.ceil((w - w_output) / 2)

    @staticmethod
    def crop(image: np.ndarray,
             border: Tuple[int, int]):

        rows, columns = image.shape[:2]
        if (border[0] < rows - border[0]) and (border[1] < columns - border[1]):
            return image[border[0]: rows - border[0], border[1]:columns - border[1], :]
        else:
            logging.error('Cropping image after rotation led to null image. Ignoring crop.')
            return image


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

        sample.update({'image': F.to_tensor(image),
                       'label': torch.from_numpy(label),
                       'shape': torch.tensor(shape)})
        return sample


def make_transforms(parameters: DataParams):
    """
    todo: doc
    :param parameters:
    :return:
    """

    transform_list = list()

    # todo : normalize

    # resize
    if parameters.data_augmentation_max_scaling > 1.0:
        transform_list.append(RandomResize(scaling=parameters.data_augmentation_max_scaling,
                                           output_size=parameters.input_resized_size))
    else:
        transform_list.append(CustomResize(output_size=parameters.input_resized_size))

    if parameters.data_augmentation_max_rotation > 0:
        transform_list.append(SampleRandomRotation(max_angle=parameters.data_augmentation_max_rotation,
                                                   do_crop=False))

    # todo: make patches
    if parameters.make_patches:
        raise NotImplementedError
        # transform_list.append(SamplePatcher())

    if parameters.data_augmentation_flip_lr:
        transform_list.append(transforms.SampleRandomHorizontalFlip()) # todo: this needs PIL image

    if parameters.data_augmentation_flip_ud:
        transform_list.append(transforms.SampleRandomVerticalFlip()) # todo: this needs PIL image

    if parameters.data_augmentation_color:
        transform_list.append(SampleColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5)) # todo: this needs PIL image

    # Assign class id to color
    # Attention: this should be the last operation before ToTensor transform
    if parameters.prediction_type == PredictionType.CLASSIFICATION:
        transform_list.append(AssignLabelClassification(parameters.color_codes))
    elif parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(AssignLabelMultilabel(parameters.color_codes, parameters.onehot_labels))

    # to tensor
    transform_list.append(SampleToTensor())

    return transforms.Compose(transform_list)