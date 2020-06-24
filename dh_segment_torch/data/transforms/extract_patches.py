from typing import Tuple, Union

from albumentations import DualTransform
from skimage.util import view_as_windows

from dh_segment_torch.data.transforms.transform import Transform


@Transform.register("sample_to_patches")
class SampleToPatches(DualTransform):
    """
    Needs numpy array as input.
    """

    def __init__(self, patch_shape: Union[int, Tuple[int, int]]):
        super().__init__(always_apply=True)
        if isinstance(patch_shape, int):
            patch_shape = (patch_shape, patch_shape)
        if patch_shape[0] <= 1 or patch_shape[1] <= 1:
            raise ValueError("Patch shapes should be > 1.")
        self.patch_shape = patch_shape

    def apply(self, img, **params):
        return extract_patches(img, self.patch_shape)

    def apply_to_mask(self, img, **params):
        return extract_patches(img, self.patch_shape)


def extract_patches(image, patch_shape=(300, 300), overlap=(None, None)):
    if len(image.shape) > 3:
        raise ValueError("Expected single image")

    if image.ndim != 2 and image.ndim != 3:
        raise ValueError("Excpected image to have at least two dimension")

    patch_h, patch_w = patch_shape

    stride_h, stride_w = overlap
    if stride_h is None:
        stride_h = patch_h // 2
    if stride_w is None:
        stride_w = patch_w // 2
    if image.ndim == 2:
        window_shape = (patch_h, patch_w)
        step = (stride_h, stride_w)
    else:
        window_shape = (patch_h, patch_w, image.shape[2])
        step = (stride_h, stride_w, 1)

    patches = view_as_windows(image, window_shape, step)
    patches = patches.reshape(-1, *window_shape)
    return patches
