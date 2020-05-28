from typing import List

import numpy as np
from albumentations import DualTransform

from dh_segment_torch.data.transform import Transform


@Transform.register("assign_label_classification")
class AssignLabelClassification(DualTransform):
    """
   Converts the RGB image to a HxW image with the corresponding class label for each pixel.

   :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
   :vartype color_array: np.ndarray
   """

    def __init__(self, colors_array: List[List[int]]):
        super().__init__(always_apply=True, p=1)
        self.colors_array = np.array(colors_array)
        self.colors_array = (np.array(colors_array)*[1000*1000, 1000, 1]).sum(axis=-1)[None, :]#.astype(np.uint8)

    def apply(self, img, **kwargs):
        return img

    def apply_to_mask(self, img, **kwargs):
        # Convert label_image [H,W,3] to the classes [H,W],int32 according to the classes [C,3]
        diff = (img * [1000 * 1000, 1000, 1]).sum(axis=-1)[:, :, None] - self.colors_array  # [H, W, C]
        mask_image = np.argmin(np.abs(diff), axis=-1)  # [H,W]

        return mask_image


@Transform.register("assign_label_multilabel")
class AssignLabelMultilabel(DualTransform):
    """
    Converts the RGB image to a HxWxC image with onehot encoding in C dimension.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    :ivar code_array: list of onehot encoded labels with shape N x C, with C
    the number of classes and N the number of possible color codes.
    :vartype code_array: np.ndarray
    """

    def __init__(self, colors_array: List[List[int]], onehot_label_array: List[List[int]]):

        super().__init__(always_apply=True, p=1)
        self.colors_array = np.array(colors_array)
        self.colors_array = (np.array(colors_array)*[1000*1000, 1000, 1]).sum(axis=-1)[None, :]#.astype(np.uint8)
        self.onehot_label_array = np.array(onehot_label_array)#.astype(np.uint8)

    def apply(self, img, **kwargs):
        return img

    # def apply_to_mask(self, img, **kwargs):
    #
    #     # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
    #     if len(img.shape) == 3:
    #         diff = img[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]
    #     else:
    #         raise NotImplementedError("Length is : {}".format(len(img.shape)))
    #
    #     pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
    #     label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]
    #
    #     mask_image = (
    #         np.take(self.onehot_label_array, label_image, axis=0) > 0
    #     )  # [H, W, C]
    #     mask_image = mask_image.astype(np.float32)
    #
    #     return mask_image.transpose((2, 0, 1))

    def apply_to_mask(self, img, **kwargs):

        # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
        if len(img.shape) == 3:
            diff = (img*[1000*1000, 1000, 1]).sum(axis=-1)[:, :, None] - self.colors_array  # [H, W, C]
        else:
            raise NotImplementedError("Length is : {}".format(len(img.shape)))

        # pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(np.abs(diff), axis=-1)  # [H,W]

        mask_image = (
            np.take(self.onehot_label_array, label_image, axis=0) > 0
        )  # [H, W, C]
        mask_image = mask_image.astype(np.float32) # TODO check that is later converted to float32
        # mask_image = mask_image.astype(np.uint8)
        #
        return mask_image#.transpose((2, 0, 1))

@Transform.register("assign_label")
class AssignLabel(DualTransform):
    """
    Converts the RGB image to a HxWxC image with onehot encoding in C dimension.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    :ivar code_array: list of onehot encoded labels with shape N x C, with C
    the number of classes and N the number of possible color codes.
    :vartype code_array: np.ndarray
    """

    def __init__(self, colors_array: List[List[int]]):

        super().__init__(always_apply=True, p=1)
        self.colors_array = np.array(colors_array)
        self.colors_array = (np.array(colors_array)*[1000*1000, 1000, 1]).sum(axis=-1)[None, :]#.astype(np.uint8)

    def apply(self, img, **kwargs):
        return img

    # def apply_to_mask(self, img, **kwargs):
    #
    #     # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
    #     if len(img.shape) == 3:
    #         diff = img[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]
    #     else:
    #         raise NotImplementedError("Length is : {}".format(len(img.shape)))
    #
    #     pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
    #     label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]
    #
    #     mask_image = (
    #         np.take(self.onehot_label_array, label_image, axis=0) > 0
    #     )  # [H, W, C]
    #     mask_image = mask_image.astype(np.float32)
    #
    #     return mask_image.transpose((2, 0, 1))

    def apply_to_mask(self, img, **kwargs):

        # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
        if len(img.shape) == 3:
            diff = (img*[1000*1000, 1000, 1]).sum(axis=-1)[:, :, None] - self.colors_array  # [H, W, C]
        else:
            raise NotImplementedError("Length is : {}".format(len(img.shape)))

        # pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
        label_image = np.argmin(np.abs(diff), axis=-1)  # [H,W]

        label_image = label_image.astype(np.uint8)[:,:,None]
        #
        return label_image#.transpose((2, 0, 1))


@Transform.register("assign_onehot")
class AssignOneHot(DualTransform):

    def __init__(self, onehot_label_array: List[List[int]]):
        super().__init__(always_apply=True, p=1)
        self.onehot_label_array = np.array(onehot_label_array)#.astype(np.uint8)

    def apply(self, img, **kwargs):
        return img

    # def apply_to_mask(self, img, **kwargs):
    #
    #     # Convert label_image [H,W,3] to the classes [H,W,C],int32 according to the classes [C,3]
    #     if len(img.shape) == 3:
    #         diff = img[:, :, None, :] - self.colors_array[None, None, :, :]  # [H,W,C,3]
    #     else:
    #         raise NotImplementedError("Length is : {}".format(len(img.shape)))
    #
    #     pixel_class_diff = np.sum(np.square(diff), axis=-1)  # [H,W,C]
    #     label_image = np.argmin(pixel_class_diff, axis=-1)  # [H,W]
    #
    #     mask_image = (
    #         np.take(self.onehot_label_array, label_image, axis=0) > 0
    #     )  # [H, W, C]
    #     mask_image = mask_image.astype(np.float32)
    #
    #     return mask_image.transpose((2, 0, 1))

    def apply_to_mask(self, img, **kwargs):


        mask_image = (
            np.take(self.onehot_label_array, img[:,:,0], axis=0) > 0
        )  # [H, W, C]
        mask_image = mask_image.astype(np.float32) # TODO check that is later converted to float32
        # mask_image = mask_image.astype(np.uint8)
        #
        return mask_image#.transpose((2, 0, 1))