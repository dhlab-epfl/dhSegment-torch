#!/usr/bin/env python
import cv2
import math
import os
from PIL import Image
import numpy as np
from typing import Tuple
import torch
from torchvision import transforms
from ..utils.params_config import TrainingParams, PredictionType


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


class AssignLabel(object):
    """
    todo: doc
    """
    def __init__(self,
                 classes_file: str,
                 prediction_type: PredictionType):
        self.classes_file = classes_file
        self.prediction_type = prediction_type

    def __call__(self,
                 image: np.ndarray):
        assert len(image.shape) == 3, "Image must have [H, W, C] dimensions"

        if self.prediction_type == PredictionType.CLASSIFICATION:
            return self._assign_color_code_classification(image)
        elif self.prediction_type == PredictionType.MULTILABEL:
            return self._assign_color_code_multilabel(image)
        else:
            raise NotImplementedError

    def _assign_color_code_classification(self,
                                          image: np.ndarray) -> np.ndarray:
        classes_color_values, _ = self._get_classes_from_file()

        # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
        diff = image[:, :, None, :] - classes_color_values[None, None, :, :]  # [H,W,C,3]

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]
        return label_image

    def _assign_color_code_multilabel(self,
                                      image: np.ndarray) -> np.ndarray:

        classes_color_values, colors_labels = self._get_classes_from_file()

        # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
        if len(image.shape) == 3:
            diff = image[:, :, None, :] - classes_color_values[None, None, :, :]  # [H,W,C,3]
        else:
            raise NotImplementedError('Length is : {}'.format(len(image.shape)))

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        class_label = np.argmin(pixel_class_diff, axis=-1)  # [H,W]

        return np.take(colors_labels, class_label, axis=0) > 0

    def _get_classes_from_file(self) -> Tuple[np.ndarray, np.ndarray]:
        if not os.path.exists(self.classes_file):
            raise FileNotFoundError(self.classes_file)

        content = np.loadtxt(self.classes_file).astype(np.float32)

        if self.prediction_type == PredictionType.CLASSIFICATION:
            assert content.shape[1] == 3, "Color file should represent RGB values"
            return content, None
        elif self.prediction_type == PredictionType.MULTILABEL:
            assert content.shape[1] > 3, "The number of columns should be greater in multilabel framework"
            colors = content[:, :3]
            labels = content[:, 3:]
            return colors, labels.astype(np.int32)


class SampleAssignLabel(AssignLabel):
    """
    todo: doc
    """

    def __call__(self,
                 sample: dict):
        label = sample['label']
        label = super()(label)

        return sample.update({'label': label})


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
        range = [int(self.output_size / scaling), int(self.output_size * scaling)]
        self.output_size = np.random.randint(low=range[0], high=range[1])


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

    # resize
    if parameters.data_augmentation_max_scaling > 1.0:
        transform_list.append(RandomResize(parameters.data_augmentation_max_scaling,
                                           parameters.input_resized_size))
    else:
        transform_list.append(CustomResize(parameters.input_resized_size))

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