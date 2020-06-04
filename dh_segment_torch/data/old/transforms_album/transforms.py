from albumentations import (
    Compose,
    RandomScale,
    HorizontalFlip,
    VerticalFlip,
    RandomBrightnessContrast,
    HueSaturationValue,
)

from dh_segment_torch.data.old.transforms_album.assign_labels import (
    AssignLabelClassification,
    AssignLabel, AssignOneHot)
from dh_segment_torch.data.old.transforms_album.fixed_size_resize import FixedSizeResize
from dh_segment_torch.data.old.transforms_album.fixed_size_rotate_crop import (
    FixedSizeRotateCrop,
)
from dh_segment_torch.params import DataParams, PredictionType


def make_transforms(parameters: DataParams, eval: bool = False) -> Compose:
    """
    Create the transforms and concatenates them to form a list of transforms to apply to the ``sample`` dictionary.

    :param parameters: data parameters to generate the transforms
    :return: a list of transforms to apply wrapped in transform Compose
    """

    transform_list = make_global_transforms(parameters, eval).transforms.transforms
    transform_list += make_local_transforms(parameters, eval).transforms.transforms

    return Compose(transform_list)


def make_global_transforms(parameters: DataParams, eval: bool = False):
    transform_list = list()

    transform_list.append(
        FixedSizeResize(parameters.input_resized_size, always_apply=True)
    )


    # # resize
    # if (
    #     not eval
    #     and parameters.data_augmentation
    # ):
    #     if parameters.data_augmentation_max_scaling > 1.0:
    #         transform_list.append(RandomScale(parameters.data_augmentation_max_scaling-1))
    #
    #     if parameters.data_augmentation_max_rotation > 0:
    #         transform_list.append(FixedSizeRotateCrop(parameters.data_augmentation_max_rotation))


    if parameters.prediction_type == PredictionType.CLASSIFICATION:
        transform_list.append(AssignLabelClassification(parameters.color_codes))
    elif parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(
            AssignLabel(parameters.color_codes)
        )

    return Compose(transform_list)


def make_local_transforms(parameters: DataParams, eval: bool = False):
    transform_list = list()

    # resize
    if not eval and parameters.data_augmentation:
        if parameters.data_augmentation_max_scaling > 1.0:
            transform_list.append(
                RandomScale(parameters.data_augmentation_max_scaling - 1)
            )

        if parameters.data_augmentation_max_rotation > 0:
            transform_list.append(
                FixedSizeRotateCrop(parameters.data_augmentation_max_rotation)
            )

        if parameters.data_augmentation_horizontal_flip:
            transform_list.append(HorizontalFlip)

        if parameters.data_augmentation_vertical_flip:
            transform_list.append(VerticalFlip)

        if parameters.data_augmentation_color:
            transform_list.append(RandomBrightnessContrast())
            transform_list.append(HueSaturationValue())

    if parameters.prediction_type == PredictionType.MULTILABEL:
        transform_list.append(AssignOneHot(parameters.onehot_labels))
    # elif parameters.prediction_type == PredictionType.MULTILABEL:
    #     transform_list.append(
    #         AssignLabelMultilabel(parameters.color_codes, parameters.onehot_labels)
    #     )

    return Compose(transform_list)
