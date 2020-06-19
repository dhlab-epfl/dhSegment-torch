from typing import Union, Optional

import albumentations.augmentations.functional as F
import cv2

from dh_segment_torch.data.transform import Transform
from dh_segment_torch.data.transforms.albumentation import DualTransform


@Transform.register("fixed_resize")
class FixedResize(DualTransform):
    def __init__(
        self,
        height: Optional[Union[int, float]] = None,
        width: Optional[Union[int, float]] = None,
        interpolation: int = cv2.INTER_LINEAR,
        always_apply: bool = True,
        p: float = 1,
    ):
        super(FixedResize, self).__init__(always_apply, p)
        if height is None and width is None:
            raise ValueError("Cannot have a fixed resizer without a fixed width or height")

        self.height = height
        self.width = width
        self.interpolation = interpolation

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        height, width = img.shape[:2]

        new_height, new_width = self._compute_new_hw(height, width)

        return F.resize(
            img, height=new_height, width=new_width, interpolation=interpolation
        )

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        height = params["rows"]
        width = params["cols"]
        new_height, new_width = self._compute_new_hw(height, width)

        scale_x = new_width / width
        scale_y = new_height / height
        return F.keypoint_scale(keypoint, scale_x, scale_y)

    def _compute_new_hw(self, height: int, width: int):
        ratio = width / height

        new_height = None
        new_width = None

        if self.height:
            new_height = int(self.height)
        if self.width:
            new_width = int(self.width)

        if new_height is None:
            assert new_width is not None
            new_height = int(new_width / ratio)

        if new_width is None:
            assert new_height is not None
            new_width = int(ratio * new_height)

        return new_height, new_width

    def get_transform_init_args_names(self):
        return ("height", "width", "interpolation")