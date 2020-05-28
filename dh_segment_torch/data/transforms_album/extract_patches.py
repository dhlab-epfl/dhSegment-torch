from typing import Tuple

from albumentations import DualTransform
from skimage.util import view_as_windows


class SampleToPatches(DualTransform):
    """
    Needs numpy array as input.
    """

    def __init__(self, patch_shape: Tuple[int]):
        super().__init__(always_apply=True)
        self.patch_shape = patch_shape

    def apply(self, img, **params):
        return extract_patches(img, self.patch_shape)

    def apply_to_mask(self, img, **params):
        return extract_patches(img, self.patch_shape)


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
