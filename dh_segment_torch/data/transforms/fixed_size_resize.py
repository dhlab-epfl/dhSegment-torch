from typing import Union

import albumentations.augmentations.functional as F
import cv2
import numpy as np
from albumentations.core.transforms_interface import DualTransform

from dh_segment_torch.data.transform import Transform


@Transform.register("fixed_size_resize")
class FixedSizeResize(DualTransform):
    def __init__(
        self,
        output_size: Union[int, float],
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(FixedSizeResize, self).__init__(always_apply, p)
        self.output_size = output_size
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        if self.output_size == -1:
            return img
        input_shape = img.shape
        # We want X/Y = x/y and we have size = x*y so :
        ratio = input_shape[1] / input_shape[0]
        new_height = int(np.sqrt(self.output_size / ratio))
        new_width = int(self.output_size / new_height)

        return F.resize(
            img, height=new_height, width=new_width, interpolation=interpolation
        )

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        if self.output_size == -1:
            return keypoint
        height = params["rows"]
        width = params["cols"]
        ratio = width / height
        new_height = int(np.sqrt(self.output_size / ratio))
        new_width = int(self.output_size / new_height)

        scale_x = new_width / width
        scale_y = new_height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def get_transform_init_args_names(self):
        return ("output_size", "interpolation")
