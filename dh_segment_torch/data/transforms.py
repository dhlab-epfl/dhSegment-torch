#!/usr/bin/env python
import cv2
import math
from PIL import Image
import numpy as np
import pandas as pd
from typing import Tuple
import logging
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from skimage.util import view_as_windows
from ..params import DataParams, PredictionType


class AssignLabelClassification(object):
    """
    Converts the RGB image to a HxW image with the corresponding class label for each pixel.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    """

    def __init__(self, colors_array: np.ndarray):
        self.colors_array = colors_array

    def __call__(self, sample: dict):

        label = sample["label"]
        # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
        diff = label[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]

        sample.update({"label": label_image})
        return sample


class AssignLabelMultilabel(object):
    """
    Converts the RGB image to a HxWxC image with onehot encoding in C dimension.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    :ivar code_array: list of onehot encoded labels with shape N x C, with C
    the number of classes and N the number of possible color codes.
    :vartype code_array: np.ndarray
    """

    def __init__(self, colors_array: np.ndarray, onehot_label_array: np.ndarray):
        self.colors_array = colors_array
        self.onehot_label_array = onehot_label_array

    def __call__(self, sample: dict):
        label = sample["label"]

        # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
        if len(label.shape) == 3:
            diff = (
                label[:, :, None, :] - self.colors_array[None, None, :, :]
            )  # [H,W,C,3]
        else:
            raise NotImplementedError("Length is : {}".format(len(label.shape)))

        pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]

        label_image = (
            np.take(self.onehot_label_array, label_image, axis=0) > 0
        )  # [H, W, C]
        label_image = label_image.astype(np.float32)
        sample.update({"label": label_image.transpose((2, 0, 1))})  # [C, H, W]

        return sample


class CustomResize(object):
    """
    Resize according to number of pixels and keeps the same ratio for sample (image, label).
    Needs numpy array as input.

    :ivar output_size: the size of the output image (in pixels)
    :vartype output_size: int
    """

    def __init__(self, output_size: int):

        assert isinstance(output_size, int)
        self.output_size = output_size

    def __call__(self, sample: dict):
        """

        :param sample:
        :return:
        """
        # TODO add to doc
        if self.output_size == -1:
            return sample

        image, label_image = sample["image"], sample["label"]

        # compute new size
        input_shape = image.shape
        # We want X/Y = x/y and we have size = x*y so :
        ratio = input_shape[1] / input_shape[0]
        new_height = int(math.sqrt(self.output_size / ratio))
        new_width = int(self.output_size / new_height)

        resized_image = cv2.resize(
            image, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR
        )
        resized_label = cv2.resize(
            label_image, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST
        )

        sample.update(
            {
                "image": resized_image,
                "label": resized_label,
                "shape": resized_image.shape[:2],
            }
        )
        return sample


class RandomResize(CustomResize):
    """
    Resizes the image using a random size (in pixels) keeping the original image ratio.

    :ivar scaling: scaling factor corresponding to the zoom in or zoom out. A scaling value of 2 means
    having a range of new size of [input_size/2, input_size*2]
    :vartpe scaling: float
    :ivar output_size: size of the output image (in pixels)
    :vartype output_size: int
    """

    def __init__(self, scaling: float, output_size: int):
        super().__init__(output_size)
        self.range = [int(self.output_size / scaling), int(self.output_size * scaling)]

    def __call__(self, sample: dict):
        self.output_size = np.random.randint(low=self.range[0], high=self.range[1])
        return super().__call__(sample)  # todo: verify syntax


class SampleColorJitter(transforms.ColorJitter):
    """
    Wrapper for ``transforms.ColorJitter`` to use sample {image, label} as input and output.
    Needs PIL Images as input.
    """

    def __call__(self, sample: dict):
        image = sample["image"]

        transform = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )

        sample.update({"image": transform(image)})
        return sample


class SampleRandomVerticalFlip(object):
    """
    Wrapper for ``transforms.RandomVerticalFlip`` to use sample {image, label} as input and output.
    Needs PIL Images as input.

    :ivar p: probability of vertival flip
    :vartype p: float
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]
        transform = F.vflip if np.random.random() < self.p else lambda x: x

        sample.update({"image": transform(image), "label": transform(label)})
        return sample


class SampleRandomHorizontalFlip(object):
    """
    Wrapper for ``transforms.RandomVerticalFlip`` to use sample {image, label} as input and output.
    Needs PIL Images as input.

    :ivar p: probability of horizontal flip
    :vartype p: float
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]
        transform = F.hflip if np.random.random() < self.p else lambda x: x

        sample.update({"image": transform(image), "label": transform(label)})
        return sample


class SampleRandomRotation(object):
    """
    Rotates the ``image`` and ``label`` with a random angle. if ``do_crop`` is set to True,
    the black borders resulting from the rotation are cropped and the size of the image ``shape`` is updated.
    Needs numpy array as input.

    :ivar max_angle: maximum angle of rotation (the range will be between [-angle, angle])
    :vartype max_angle: int
    :ivar do_crop: wether to crop the black borders or not
    :vartype do_crop: bool
    """

    def __init__(self, max_angle: int, do_crop: bool = False):
        self.angle = max_angle
        self.do_crop = do_crop

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]
        rows, columns = image.shape[:2]

        angle = np.random.randint(-self.angle, self.angle)
        rot_matrix = cv2.getRotationMatrix2D(
            ((columns - 1) / 2.0, (rows - 1) / 2.0), angle, 1
        )

        rotated_image = cv2.warpAffine(
            image, rot_matrix, (columns, rows), flags=cv2.INTER_LINEAR
        )
        rotated_label = cv2.warpAffine(
            label, rot_matrix, (columns, rows), flags=cv2.INTER_NEAREST
        )

        if self.do_crop:
            # todo: if crop not possible should we return rotated image or original image ?
            border_size = self.compute_border_to_crop(image.shape[:2])
            crop_image = self.crop(rotated_image, border_size)
            crop_label = self.crop(rotated_label, border_size)

            crop_shape = crop_image.shape[:2]

            sample.update(
                {"image": crop_image, "label": crop_label, "shape": crop_shape}
            )
            return sample

        sample.update({"image": rotated_image, "label": rotated_label})
        return sample

    def compute_border_to_crop(self, input_shape: Tuple[int, int]):
        """See https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders for formulae
        """
        angle = abs(self.angle)
        h, w = input_shape[0], input_shape[1]
        if h > w:
            long_side, short_side = h, w
        else:
            long_side, short_side = w, h

        long_side = (
            long_side * math.cos(angle) - short_side * math.sin(angle)
        ) / math.cos(2 * angle)
        short_side = (short_side - math.sin(angle) * long_side) / math.cos(angle)
        if h > w:
            h_output, w_output = long_side, short_side
        else:
            h_output, w_output = short_side, long_side

        return math.ceil((h - h_output) / 2), math.ceil((w - w_output) / 2)

    @staticmethod
    def crop(image: np.ndarray, border: Tuple[int, int]):
        rows, columns = image.shape[:2]
        if (border[0] < rows - border[0]) and (border[1] < columns - border[1]):
            return image[
                border[0] : rows - border[0], border[1] : columns - border[1], :
            ]
        else:
            logging.error(
                "Cropping image after rotation led to null image. Ignoring crop."
            )
            return image


def extract_patches(image, patch_shape=(300, 300), overlap=(None, None)):
    if len(image.shape) > 3:
        raise ValueError("Expected single image")

    patch_h, patch_w = patch_shape

    stride_h, stride_w = overlap
    if stride_h is None:
        stride_h = patch_h // 2
    if stride_w is None:
        stride_w = patch_w // 2

    window_shape = (patch_h, patch_w, image.shape[2])
    step = (stride_h, stride_w, 1)
    patches = view_as_windows(image, window_shape, step)
    patches = patches.reshape(-1, patch_h, patch_w, image.shape[2])
    return patches


class SampleToPatches(object):
    """
    Needs numpy array as input.
    """

    def __init__(self, patch_shape: Tuple[int]):
        self.patch_shape = patch_shape

    def __call__(self, sample: dict):
        images = extract_patches(sample["image"], self.patch_shape)
        labels = extract_patches(sample["label"], self.patch_shape)
        shapes = [self.patch_shape for _ in range(len(images))]

        sample = {"images": images, "labels": labels, "shapes": shapes}

        return sample


def transform_to_several(transform: object):
    def patched_transform(samples: dict):
        samples = pd.DataFrame(
            {
                "image": [image for image in samples["images"]],
                "label": [label for label in samples["labels"]],
                "shape": [shape for shape in samples["shapes"]],
            }
        ).to_dict("record")

        results = []
        for sample in samples:
            results.append(transform(sample))

        if torch.is_tensor(results[0]["image"]):
            stack = lambda x: torch.stack(x)
        else:
            stack = lambda x: np.stack(x)

        results = pd.DataFrame.from_dict(results)

        samples = {
            "images": stack(results["image"].values.tolist()),
            "labels": stack(results["label"].values.tolist()),
            "shapes": stack(results["shape"].values.tolist()),
        }
        return samples

    return patched_transform


class SampleToTensor(object):
    """
    Convert ndarrays to Tensors for sample {image, label, shape}.
    """

    def __call__(self, sample: dict):
        image, label, shape = sample["image"], sample["label"], sample["shape"]

        sample.update(
            {
                "image": F.to_tensor(image),
                "label": torch.from_numpy(label),
                "shape": torch.tensor(shape),
            }
        )
        return sample


class SampleNumpyToPIL(object):
    """
    Convert numpy array image to PIL image
    """

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]

        sample.update(
            {"image": Image.fromarray(image), "label": Image.fromarray(label)}
        )
        return sample


class SamplePILToNumpy(object):
    """
    Convert a PIL image to a numpy array
    """

    def __call__(self, sample: dict):
        image, label = sample["image"], sample["label"]

        sample.update(({"image": np.array(image), "label": np.array(label)}))
        return sample


def make_transforms(parameters: DataParams) -> transforms.Compose:
    """
    Create the transforms and concatenates them to form a list of transforms to apply to the ``sample`` dictionary.

    :param parameters: data parameters to generate the transforms
    :return: a list of transforms to apply wrapped in transform Compose
    """

    transform_list = list()

    # resize
    if parameters.data_augmentation_max_scaling > 1.0:
        transform_list.append(
            RandomResize(
                scaling=parameters.data_augmentation_max_scaling,
                output_size=parameters.input_resized_size,
            )
        )
    else:
        transform_list.append(CustomResize(output_size=parameters.input_resized_size))

    if parameters.data_augmentation_max_rotation > 0:
        transform_list.append(
            SampleRandomRotation(
                max_angle=parameters.data_augmentation_max_rotation, do_crop=False
            )
        )

    if parameters.make_patches:
        raise NotImplementedError
        # todo
        # transform_list.append(SamplePatcher())

    transform_list.append(SampleNumpyToPIL())

    if parameters.data_augmentation_horizontal_flip:
        transform_list.append(SampleRandomHorizontalFlip())

    if parameters.data_augmentation_vertical_flip:
        transform_list.append(SampleRandomVerticalFlip())

    if parameters.data_augmentation_color:
        transform_list.append(
            SampleColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5)
        )

    transform_list.append(SamplePILToNumpy())

    # Attention: this should be the last operation before ToTensor transform
    if parameters.prediction_type == PredictionType.CLASSIFICATION:
        transform_list.append(AssignLabelClassification(parameters.color_codes))
    elif parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(
            AssignLabelMultilabel(parameters.color_codes, parameters.onehot_labels)
        )

    # to tensor
    transform_list.append(SampleToTensor())

    return transforms.Compose(transform_list)


def make_global_transforms(parameters: DataParams, eval: bool = False):
    transform_list = list()

    # resize
    if (
        not eval
        and parameters.data_augmentation
        and parameters.data_augmentation_max_scaling > 1.0
    ):
        transform_list.append(
            RandomResize(
                scaling=parameters.data_augmentation_max_scaling,
                output_size=parameters.input_resized_size,
            )
        )
    else:
        transform_list.append(CustomResize(output_size=parameters.input_resized_size))

    if not eval and parameters.data_augmentation_max_rotation > 0:
        transform_list.append(
            SampleRandomRotation(
                max_angle=parameters.data_augmentation_max_rotation, do_crop=False
            )
        )

    return transforms.Compose(transform_list)


def make_local_transforms(parameters: DataParams, eval: bool = False):
    transform_list = list()

    if not eval:
        transform_list.append(SampleNumpyToPIL())

        if parameters.data_augmentation_horizontal_flip:
            transform_list.append(SampleRandomHorizontalFlip())

        if parameters.data_augmentation_vertical_flip:
            transform_list.append(SampleRandomVerticalFlip())

        if parameters.data_augmentation_color:
            transform_list.append(
                SampleColorJitter(brightness=1, contrast=1, saturation=1, hue=0.5)
            )

        transform_list.append(SamplePILToNumpy())

    # Attention: this should be the last operation before ToTensor transform
    if parameters.prediction_type == PredictionType.CLASSIFICATION:
        transform_list.append(AssignLabelClassification(parameters.color_codes))
    elif parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(
            AssignLabelMultilabel(parameters.color_codes, parameters.onehot_labels)
        )

    # to tensor
    transform_list.append(SampleToTensor())

    return transforms.Compose(transform_list)


def make_eval_transforms(parameters: DataParams) -> transforms.Compose:
    """
    Create the transforms for evaluation and concatenates them to form a list of
    transforms to apply to the ``sample`` dictionary.

    :param parameters: data parameters to generate the transforms
    :return: a list of transforms to apply wrapped in transform Compose
    """

    transform_list = list()

    transform_list.append(CustomResize(output_size=parameters.input_resized_size))

    if parameters.make_patches:
        raise NotImplementedError
        # todo
        # transform_list.append(SamplePatcher())

    # Attention: this should be the last operation before ToTensor transform
    if parameters.prediction_type == PredictionType.CLASSIFICATION:
        transform_list.append(AssignLabelClassification(parameters.color_codes))
    elif parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(
            AssignLabelMultilabel(parameters.color_codes, parameters.onehot_labels)
        )
    else:
        raise TypeError(f"Unsupported prediction type {parameters.prediction_type}")

    # to tensor
    transform_list.append(SampleToTensor())

    return transforms.Compose(transform_list)
