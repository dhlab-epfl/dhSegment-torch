from typing import List, Tuple

import numpy as np

from dh_segment_torch.config.registrable import Registrable


class Assign(Registrable):
    def first_phase(self, img):
        raise NotImplementedError

    def second_phase(self, img):
        raise NotImplementedError

    def apply(self, img):
        raise NotImplementedError


@Assign.register("assign_label")
class AssignLabel(Assign):
    """
    Converts the RGB image to a HxWxC image with onehot encoding in C dimension.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    :ivar code_array: list of onehot encoded labels with shape N x C, with C
    the number of classes and N the number of possible color codes.
    :vartype code_array: np.ndarray
    """

    def __init__(self, colors_array: List[Tuple[int, int, int]]):
        self.colors_array_flat = (np.array(colors_array) * [1000 * 1000, 1000, 1]).sum(
            axis=-1
        )[None, :]

    def first_phase(self, img):
        return self.apply(img)

    def second_phase(self, label):
        return label

    def apply(self, img):

        # Convert label_image [H,W] to the classes [H,W,C],int32 according to the classes [C]
        if len(img.shape) == 3:
            diff = (img * [1000 * 1000, 1000, 1]).sum(axis=-1)[
                :, :, None
            ] - self.colors_array_flat  # [H, W, C]
        else:
            raise NotImplementedError("Length is : {}".format(len(img.shape)))

        label_image = np.argmin(np.abs(diff), axis=-1).astype(np.int)

        return label_image


@Assign.register("assign_multilabel")
class AssignMultilabel(AssignLabel):
    """
    Converts the RGB image to a HxWxC image with onehot encoding in C dimension.

    :ivar color_array: the list of possible color codes with shape N x 3, with N the number of possible color codes.
    :vartype color_array: np.ndarray
    :ivar code_array: list of onehot encoded labels with shape N x C, with C
    the number of classes and N the number of possible color codes.
    :vartype code_array: np.ndarray
    """

    def __init__(
        self,
        colors_array: List[Tuple[int, int, int]],
        onehot_label_array: List[List[int]],
    ):
        super().__init__(colors_array)
        self.onehot_label_array = onehot_label_array

    def first_phase(self, img):
        return super().apply(img)

    def second_phase(self, label):
        mask_image = np.take(self.onehot_label_array, label, axis=0) > 0  # [H, W, C]
        mask_image = mask_image.astype(
            np.float32
        )  # TODO check that is later converted to float32

        return mask_image

    def apply(self, img):
        label = self.first_phase(img)
        return self.second_phase(label)
