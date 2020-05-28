import albumentations.augmentations.functional as F
import cv2

from dh_segment_torch.data.transform.albumentation import Rotate
from dh_segment_torch.data.transforms_album.rotate_no_crop import (
    get_rotated_size,
    rotate_no_crop,
    keypoint_rotate_no_crop,
)


class FixedSizeRotateCrop(Rotate):
    def apply(self, img, angle=0, interpolation=cv2.INTER_LINEAR, **params):
        return fixed_size_rotate_crop(
            img, angle, interpolation, self.border_mode, self.value
        )

    def apply_to_mask(self, img, angle=0, **params):
        return fixed_size_rotate_crop(
            img, angle, cv2.INTER_NEAREST, self.border_mode, self.mask_value
        )

    def apply_to_bbox(self, bbox, angle=0, **params):
        height = params["rows"]
        width = params["cols"]

        new_height, new_width = get_rotated_size(height, width, angle)
        bbox = F.bbox_rotate(bbox, angle, new_height, new_width, 0)

        crop_height, crop_width = rotatedRectWithMaxArea(height, width, angle)

        bbox = F.bbox_center_crop(bbox, crop_height, crop_width, new_height, new_width)

        resized_height, resized_width = get_resized_max_ratio(
            height, width, crop_height, crop_width
        )

        return F.bbox_center_crop(bbox, height, width, resized_height, resized_width)

    def apply_to_keypoint(self, keypoint, angle=0, **params):
        height = params["rows"]
        width = params["cols"]
        new_height, new_width = get_rotated_size(height, width, angle)

        keypoint = keypoint_rotate_no_crop(keypoint, angle, height, width)

        crop_height, crop_width = rotatedRectWithMaxArea(height, width, angle)

        keypoint = F.keypoint_center_crop(
            keypoint, crop_height, crop_width, new_height, new_width
        )

        resized_height, resized_width = get_resized_max_ratio(
            height, width, crop_height, crop_width
        )

        scale_x = resized_width / crop_width
        scale_y = resized_height / crop_height
        keypoint = F.keypoint_scale(keypoint, scale_x, scale_y)
        return F.keypoint_center_crop(
            keypoint, height, width, resized_height, resized_width
        )


def get_resized_max_ratio(original_height, original_width, new_height, new_width):
    ratio_h = original_height / new_height
    ratio_w = original_width / new_width

    scale_ratio = max(ratio_h, ratio_w)

    resize_height = round(scale_ratio * new_height) + 1
    resize_width = round(scale_ratio * new_width) + 1

    return resize_height, resize_width


def fixed_size_rotate_crop(
    img,
    angle,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
):
    original_height, original_width = img.shape[:2]
    rot_img = rotate_no_crop(img, angle, interpolation, border_mode, value)
    new_height, new_width = rotatedRectWithMaxArea(
        original_height, original_width, angle
    )

    cropped_img = F.center_crop(rot_img, new_height, new_width)

    resize_height, resize_width = get_resized_max_ratio(
        original_height, original_width, new_height, new_width
    )

    resized_img = F.resize(
        cropped_img,
        height=resize_height,
        width=resize_width,
        interpolation=interpolation,
    )

    return F.center_crop(resized_img, original_height, original_width)


def rotatedRectWithMaxArea(height, width, angle):
    """
    Source: https://stackoverflow.com/a/16778797
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    degrees), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """

    import math

    angle = math.radians(angle)

    h, w = height, width
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return round(hr), round(wr)
