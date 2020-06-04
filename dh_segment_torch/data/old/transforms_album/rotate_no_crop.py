import math

import albumentations.augmentations.functional as F
import cv2
import numpy as np
from albumentations.augmentations.functional import (
    _maybe_process_in_chunks,
    preserve_channel_dim,
)

from dh_segment_torch.data.transform.albumentation import Rotate


class RotateNoCrop(Rotate):
    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return rotate_no_crop(img, angle, interpolation, self.border_mode, self.value)

    def apply_to_mask(self, img, angle=0, **params):
        return rotate_no_crop(
            img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )

    def apply_to_bbox(self, bbox, angle=0, **params):
        height = params["rows"]
        width = params["cols"]

        new_height, new_width = get_rotated_size(height, width, angle)
        return F.bbox_rotate(bbox, angle, new_height, new_width)

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        height = params["rows"]
        width = params["cols"]

        return keypoint_rotate_no_crop(keypoint, angle, height, width)


@preserve_channel_dim
def rotate_no_crop(
    img,
    angle,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    height, width = img.shape[:2]

    image_center = (width / 2, height / 2)

    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    new_height, new_width = get_rotated_size(height, width, angle)

    matrix[0, 2] += new_width / 2 - image_center[0]
    matrix[1, 2] += new_height / 2 - image_center[1]

    warp_fn = _maybe_process_in_chunks(
        cv2.warpAffine,
        M=matrix,
        dsize=(new_width, new_height),
        flags=interpolation,
        borderMode=border_mode,
        borderValue=value,
    )

    return warp_fn(img)


def get_rotated_size(height, width, angle):
    image_center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(matrix[0, 0])
    abs_sin = abs(matrix[0, 1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    return new_height, new_width


def keypoint_rotate_no_crop(keypoint, angle, rows, cols, **params):
    """Rotate a keypoint by angle.
    Args:
        keypoint (tuple): A keypoint `(x, y, angle, scale)`.
        angle (float): Rotation angle.
        rows (int): Image height.
        cols (int): Image width.
    Returns:
        tuple: A keypoint `(x, y, angle, scale)`.
    """
    image_center = (cols / 2, rows / 2)
    matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)

    new_height, new_width = get_rotated_size(rows, cols, angle)

    matrix[0, 2] += new_width / 2 - image_center[0]
    matrix[1, 2] += new_height / 2 - image_center[1]

    #     matrix = cv2.getRotationMatrix2D(((cols - 1) * 0.5, (rows - 1) * 0.5), angle, 1.0)
    x, y, a, s = keypoint[:4]
    x, y = cv2.transform(np.array([[[x, y]]]), matrix).squeeze()
    return x, y, (a + math.radians(angle)) % (math.pi * 2), s
