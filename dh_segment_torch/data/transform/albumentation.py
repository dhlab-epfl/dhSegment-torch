from typing import Union, Tuple, Optional, List, Dict

from albumentations.augmentations import transforms
from albumentations.core import composition

from dh_segment_torch.data.transform.transform import Transform

"""
Wrapper around all albumentation Transforms and Compose
"""


@Transform.register("compose")
class Compose(composition.Compose, Transform):
    def __init__(self, transforms: List[Transform], additional_targets: Dict[str, str] = None, p: float = 1.0):
        super().__init__(transforms=transforms, additional_targets=additional_targets, p=p)


@Transform.register("blur")
class Blur(transforms.Blur, Transform):
    def __init__(self, blur_limit: Union[int, Tuple[int, int]] = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)


@Transform.register("clahe")
class CLAHE(transforms.CLAHE, Transform):
    def __init__(self, clip_limit: Union[float, Tuple[float, float]] = 4.0, tile_grid_size: Tuple[int, int] = (8, 8), always_apply: bool = False, p: float = 0.5):
        super().__init__(clip_limit=clip_limit, tile_grid_size=tile_grid_size, always_apply=always_apply, p=p)


@Transform.register("center_crop")
class CenterCrop(transforms.CenterCrop, Transform):
    def __init__(self, height: int, width: int, always_apply: bool = False, p: float = 1.0):
        super().__init__(height=height, width=width, always_apply=always_apply, p=p)


@Transform.register("channel_dropout")
class ChannelDropout(transforms.ChannelDropout, Transform):
    def __init__(self, channel_drop_range: Tuple[int, int] = (1, 1), fill_value: Tuple[int, float] = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(channel_drop_range=channel_drop_range, fill_value=fill_value, always_apply=always_apply, p=p)


@Transform.register("channel_shuffle")
class ChannelShuffle(transforms.ChannelShuffle, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("coarse_dropout")
class CoarseDropout(transforms.CoarseDropout, Transform):
    def __init__(self, max_holes: int = 8, max_height: int = 8, max_width: int = 8, min_holes: Optional[int] = None, min_height: Optional[int] = None, min_width: Optional[int] = None, fill_value: Union[int, float, List[int], List[float]] = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(max_holes=max_holes, max_height=max_height, max_width=max_width, min_holes=min_holes, min_height=min_height, min_width=min_width, fill_value=fill_value, always_apply=always_apply, p=p)


@Transform.register("crop")
class Crop(transforms.Crop, Transform):
    def __init__(self, x_min: int = 0, y_min: int = 0, x_max: int = 1024, y_max: int = 1024, always_apply: bool = False, p: float = 1.0):
        super().__init__(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max, always_apply=always_apply, p=p)


@Transform.register("crop_non_empty_mask_if_exists")
class CropNonEmptyMaskIfExists(transforms.CropNonEmptyMaskIfExists, Transform):
    def __init__(self, height: int, width: int, ignore_values: Optional[List[int]] = None, ignore_channels: Optional[List[int]] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(height=height, width=width, ignore_values=ignore_values, ignore_channels=ignore_channels, always_apply=always_apply, p=p)


@Transform.register("cutout")
class Cutout(transforms.Cutout, Transform):
    def __init__(self, num_holes: int = 8, max_h_size: int = 8, max_w_size: int = 8, fill_value: Union[int, float, List[int], List[float]] = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(num_holes=num_holes, max_h_size=max_h_size, max_w_size=max_w_size, fill_value=fill_value, always_apply=always_apply, p=p)


@Transform.register("downscale")
class Downscale(transforms.Downscale, Transform):
    def __init__(self, scale_min: float = 0.25, scale_max: float = 0.25, interpolation: int = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(scale_min=scale_min, scale_max=scale_max, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("dual_transform")
class DualTransform(transforms.DualTransform, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("elastic_transform")
class ElasticTransform(transforms.ElasticTransform, Transform):
    def __init__(self, alpha: float = 1, sigma: float = 50, alpha_affine: float = 50, interpolation: int = 1, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, approximate: bool = False, p: float = 0.5):
        super().__init__(alpha=alpha, sigma=sigma, alpha_affine=alpha_affine, interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, approximate=approximate, p=p)


@Transform.register("equalize")
class Equalize(transforms.Equalize, Transform):
    def __init__(self, mode: str = "cv", by_channels: bool = True, mask_params: List[str] = (), always_apply: bool = False, p: float = 0.5):
        super().__init__(mode=mode, by_channels=by_channels, mask_params=mask_params, always_apply=always_apply, p=p)


@Transform.register("fancy_pca")
class FancyPCA(transforms.FancyPCA, Transform):
    def __init__(self, alpha: float = 0.1, always_apply: bool = False, p: float = 0.5):
        super().__init__(alpha=alpha, always_apply=always_apply, p=p)


@Transform.register("flip")
class Flip(transforms.Flip, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("from_float")
class FromFloat(transforms.FromFloat, Transform):
    def __init__(self, dtype: str = "uint16", max_value: Optional[float] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(dtype=dtype, max_value=max_value, always_apply=always_apply, p=p)


@Transform.register("gauss_noise")
class GaussNoise(transforms.GaussNoise, Transform):
    def __init__(self, var_limit: Union[Tuple[float, float], float] = (10.0, 50.0), mean: float = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(var_limit=var_limit, mean=mean, always_apply=always_apply, p=p)


@Transform.register("gaussian_blur")
class GaussianBlur(transforms.GaussianBlur, Transform):
    def __init__(self, blur_limit: int = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)


@Transform.register("glass_blur")
class GlassBlur(transforms.GlassBlur, Transform):
    def __init__(self, sigma: float = 0.7, max_delta: int = 4, iterations: int = 2, always_apply: bool = False, mode: str = "fast", p: float = 0.5):
        super().__init__(sigma=sigma, max_delta=max_delta, iterations=iterations, always_apply=always_apply, mode=mode, p=p)


@Transform.register("grid_distortion")
class GridDistortion(transforms.GridDistortion, Transform):
    def __init__(self, num_steps: int = 5, distort_limit: Union[float, Tuple[float, float]] = 0.3, interpolation: int = 1, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(num_steps=num_steps, distort_limit=distort_limit, interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, p=p)


@Transform.register("grid_dropout")
class GridDropout(transforms.GridDropout, Transform):
    def __init__(self, ratio: float = 0.5, unit_size_min: Optional[int] = None, unit_size_max: Optional[int] = None, holes_number_x: Optional[int] = None, holes_number_y: Optional[int] = None, shift_x: int = 0, shift_y: int = 0, random_offset: bool = False, fill_value: int = 0, mask_fill_value: Optional[int] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(ratio=ratio, unit_size_min=unit_size_min, unit_size_max=unit_size_max, holes_number_x=holes_number_x, holes_number_y=holes_number_y, shift_x=shift_x, shift_y=shift_y, random_offset=random_offset, fill_value=fill_value, mask_fill_value=mask_fill_value, always_apply=always_apply, p=p)


@Transform.register("horizontal_flip")
class HorizontalFlip(transforms.HorizontalFlip, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("hue_saturation_value")
class HueSaturationValue(transforms.HueSaturationValue, Transform):
    def __init__(self, hue_shift_limit: Union[Tuple[int, int], int] = 20, sat_shift_limit: Union[Tuple[int, int], int] = 30, val_shift_limit: Union[Tuple[int, int], int] = 20, always_apply: bool = False, p: float = 0.5):
        super().__init__(hue_shift_limit=hue_shift_limit, sat_shift_limit=sat_shift_limit, val_shift_limit=val_shift_limit, always_apply=always_apply, p=p)


@Transform.register("iso_noise")
class ISONoise(transforms.ISONoise, Transform):
    def __init__(self, color_shift: Tuple[float, float] = (0.01, 0.05), intensity: Tuple[float, float] = (0.1, 0.5), always_apply: bool = False, p: float = 0.5):
        super().__init__(color_shift=color_shift, intensity=intensity, always_apply=always_apply, p=p)


@Transform.register("image_compression")
class ImageCompression(transforms.ImageCompression, Transform):
    def __init__(self, quality_lower: float = 99, quality_upper: float = 100, compression_type: int = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(quality_lower=quality_lower, quality_upper=quality_upper, compression_type=compression_type, always_apply=always_apply, p=p)


@Transform.register("image_only_transform")
class ImageOnlyTransform(transforms.ImageOnlyTransform, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("invert_img")
class InvertImg(transforms.InvertImg, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("jpeg_compression")
class JpegCompression(transforms.JpegCompression, Transform):
    def __init__(self, quality_lower: float = 99, quality_upper: float = 100, always_apply: bool = False, p: float = 0.5):
        super().__init__(quality_lower=quality_lower, quality_upper=quality_upper, always_apply=always_apply, p=p)


@Transform.register("lambda")
class Lambda(transforms.Lambda, Transform):
    def __init__(self, always_apply: bool = False, p: float = 1.0):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("longest_max_size")
class LongestMaxSize(transforms.LongestMaxSize, Transform):
    def __init__(self, max_size: int = 1024, interpolation: int = 1, always_apply: bool = False, p: float = 1):
        super().__init__(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("mask_dropout")
class MaskDropout(transforms.MaskDropout, Transform):
    def __init__(self, max_objects: int = 1, image_fill_value: int = 0, mask_fill_value: int = 0, always_apply: bool = False, p: float = 0.5):
        super().__init__(max_objects=max_objects, image_fill_value=image_fill_value, mask_fill_value=mask_fill_value, always_apply=always_apply, p=p)


@Transform.register("median_blur")
class MedianBlur(transforms.MedianBlur, Transform):
    def __init__(self, blur_limit: int = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)


@Transform.register("motion_blur")
class MotionBlur(transforms.MotionBlur, Transform):
    def __init__(self, blur_limit: int = 7, always_apply: bool = False, p: float = 0.5):
        super().__init__(blur_limit=blur_limit, always_apply=always_apply, p=p)


@Transform.register("multiplicative_noise")
class MultiplicativeNoise(transforms.MultiplicativeNoise, Transform):
    def __init__(self, multiplier: Union[Tuple[float, float], float] = (0.9, 1.1), per_channel: bool = False, elementwise: bool = False, always_apply: bool = False, p: bool = 0.5):
        super().__init__(multiplier=multiplier, per_channel=per_channel, elementwise=elementwise, always_apply=always_apply, p=p)


@Transform.register("no_op")
class NoOp(transforms.NoOp, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("normalize")
class Normalize(transforms.Normalize, Transform):
    def __init__(self, mean: Tuple[float, List[float]] = (0.485, 0.456, 0.406), std: Tuple[float, List[float]] = (0.229, 0.224, 0.225), max_pixel_value: float = 255.0, always_apply: bool = False, p: float = 1.0):
        super().__init__(mean=mean, std=std, max_pixel_value=max_pixel_value, always_apply=always_apply, p=p)


@Transform.register("optical_distortion")
class OpticalDistortion(transforms.OpticalDistortion, Transform):
    def __init__(self, distort_limit: Union[float, Tuple[float, float]] = 0.05, shift_limit: Union[float, Tuple[float, float]] = 0.05, interpolation: int = 1, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(distort_limit=distort_limit, shift_limit=shift_limit, interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, p=p)


@Transform.register("pad_if_needed")
class PadIfNeeded(transforms.PadIfNeeded, Transform):
    def __init__(self, min_height: int = 1024, min_width: int = 1024, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(min_height=min_height, min_width=min_width, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, p=p)


@Transform.register("posterize")
class Posterize(transforms.Posterize, Transform):
    def __init__(self, num_bits: Tuple[int, int] = 4, always_apply: bool = False, p: float = 0.5):
        super().__init__(num_bits=num_bits, always_apply=always_apply, p=p)


@Transform.register("rgb_shift")
class RGBShift(transforms.RGBShift, Transform):
    def __init__(self, r_shift_limit: Union[Tuple[int, int], int] = 20, g_shift_limit: Union[Tuple[int, int], int] = 20, b_shift_limit: Union[Tuple[int, int], int] = 20, always_apply: bool = False, p: float = 0.5):
        super().__init__(r_shift_limit=r_shift_limit, g_shift_limit=g_shift_limit, b_shift_limit=b_shift_limit, always_apply=always_apply, p=p)


@Transform.register("random_brightness")
class RandomBrightness(transforms.RandomBrightness, Transform):
    def __init__(self, limit: Union[Tuple[float, float], float] = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(limit=limit, always_apply=always_apply, p=p)


@Transform.register("random_brightness_contrast")
class RandomBrightnessContrast(transforms.RandomBrightnessContrast, Transform):
    def __init__(self, brightness_limit: Union[Tuple[float, float], float] = 0.2, contrast_limit: Union[Tuple[float, float], float] = 0.2, brightness_by_max: bool = True, always_apply: bool = False, p: float = 0.5):
        super().__init__(brightness_limit=brightness_limit, contrast_limit=contrast_limit, brightness_by_max=brightness_by_max, always_apply=always_apply, p=p)


@Transform.register("random_contrast")
class RandomContrast(transforms.RandomContrast, Transform):
    def __init__(self, limit: Union[Tuple[float, float], float] = 0.2, always_apply: bool = False, p: float = 0.5):
        super().__init__(limit=limit, always_apply=always_apply, p=p)


@Transform.register("random_crop")
class RandomCrop(transforms.RandomCrop, Transform):
    def __init__(self, height: int, width: int, always_apply: bool = False, p: float = 1.0):
        super().__init__(height=height, width=width, always_apply=always_apply, p=p)


@Transform.register("random_crop_near_b_box")
class RandomCropNearBBox(transforms.RandomCropNearBBox, Transform):
    def __init__(self, max_part_shift: float = 0.3, always_apply: bool = False, p: float = 1.0):
        super().__init__(max_part_shift=max_part_shift, always_apply=always_apply, p=p)


@Transform.register("random_fog")
class RandomFog(transforms.RandomFog, Transform):
    def __init__(self, fog_coef_lower: float = 0.3, fog_coef_upper: float = 1, alpha_coef: float = 0.08, always_apply: bool = False, p: float = 0.5):
        super().__init__(fog_coef_lower=fog_coef_lower, fog_coef_upper=fog_coef_upper, alpha_coef=alpha_coef, always_apply=always_apply, p=p)


@Transform.register("random_gamma")
class RandomGamma(transforms.RandomGamma, Transform):
    def __init__(self, gamma_limit: Union[float, Tuple[float, float]] = (80, 120), eps: Optional[float] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(gamma_limit=gamma_limit, eps=eps, always_apply=always_apply, p=p)


@Transform.register("random_grid_shuffle")
class RandomGridShuffle(transforms.RandomGridShuffle, Transform):
    def __init__(self, grid: Tuple[int, int] = (3, 3), always_apply: bool = False, p: float = 0.5):
        super().__init__(grid=grid, always_apply=always_apply, p=p)


@Transform.register("random_rain")
class RandomRain(transforms.RandomRain, Transform):
    def __init__(self, slant_lower: Tuple[int, int] = -10, slant_upper: Tuple[int, int] = 10, drop_length: int = 20, drop_width: int = 1, drop_color: List[Tuple[int, int, int]] = (200, 200, 200), blur_value: int = 7, brightness_coefficient: float = 0.7, rain_type: Optional[str] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(slant_lower=slant_lower, slant_upper=slant_upper, drop_length=drop_length, drop_width=drop_width, drop_color=drop_color, blur_value=blur_value, brightness_coefficient=brightness_coefficient, rain_type=rain_type, always_apply=always_apply, p=p)


@Transform.register("random_resized_crop")
class RandomResizedCrop(transforms.RandomResizedCrop, Transform):
    def __init__(self, height: int, width: int, scale: Tuple[float, float] = (0.08, 1.0), ratio: Tuple[float, float] = (0.75, 1.3333333333333333), interpolation: int = 1, always_apply: bool = False, p: float = 1.0):
        super().__init__(height=height, width=width, scale=scale, ratio=ratio, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("random_rotate")
class RandomRotate90(transforms.RandomRotate90, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("random_scale")
class RandomScale(transforms.RandomScale, Transform):
    def __init__(self, scale_limit: Union[Tuple[float, float], float] = 0.1, interpolation: int = 1, always_apply: bool = False, p: float = 0.5):
        super().__init__(scale_limit=scale_limit, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("random_shadow")
class RandomShadow(transforms.RandomShadow, Transform):
    def __init__(self, shadow_roi: Tuple[float, float, float, float] = (0, 0.5, 1, 1), num_shadows_lower: int = 1, num_shadows_upper: int = 2, shadow_dimension: int = 5, always_apply: bool = False, p: float = 0.5):
        super().__init__(shadow_roi=shadow_roi, num_shadows_lower=num_shadows_lower, num_shadows_upper=num_shadows_upper, shadow_dimension=shadow_dimension, always_apply=always_apply, p=p)


@Transform.register("random_sized_b_box_safe_crop")
class RandomSizedBBoxSafeCrop(transforms.RandomSizedBBoxSafeCrop, Transform):
    def __init__(self, height: int, width: int, erosion_rate: float = 0.0, interpolation: int = 1, always_apply: bool = False, p: float = 1.0):
        super().__init__(height=height, width=width, erosion_rate=erosion_rate, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("random_sized_crop")
class RandomSizedCrop(transforms.RandomSizedCrop, Transform):
    def __init__(self, min_max_height: Tuple[int, int], height: int, width: int, w2h_ratio: float = 1.0, interpolation: int = 1, always_apply: bool = False, p: float = 1.0):
        super().__init__(min_max_height=min_max_height, height=height, width=width, w2h_ratio=w2h_ratio, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("random_snow")
class RandomSnow(transforms.RandomSnow, Transform):
    def __init__(self, snow_point_lower: float = 0.1, snow_point_upper: float = 0.3, brightness_coeff: float = 2.5, always_apply: bool = False, p: float = 0.5):
        super().__init__(snow_point_lower=snow_point_lower, snow_point_upper=snow_point_upper, brightness_coeff=brightness_coeff, always_apply=always_apply, p=p)


@Transform.register("random_sun_flare")
class RandomSunFlare(transforms.RandomSunFlare, Transform):
    def __init__(self, flare_roi: Tuple[float, float, float, float] = (0, 0, 1, 0.5), angle_lower: float = 0, angle_upper: float = 1, num_flare_circles_lower: int = 6, num_flare_circles_upper: int = 10, src_radius: int = 400, src_color: Tuple[int, int, int] = (255, 255, 255), always_apply: bool = False, p: float = 0.5):
        super().__init__(flare_roi=flare_roi, angle_lower=angle_lower, angle_upper=angle_upper, num_flare_circles_lower=num_flare_circles_lower, num_flare_circles_upper=num_flare_circles_upper, src_radius=src_radius, src_color=src_color, always_apply=always_apply, p=p)


@Transform.register("resize")
class Resize(transforms.Resize, Transform):
    def __init__(self, height: int, width: int, interpolation: int = 1, always_apply: bool = False, p: float = 1):
        super().__init__(height=height, width=width, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("rotate")
class Rotate(transforms.Rotate, Transform):
    def __init__(self, limit: Union[Tuple[int, int], int] = 90, interpolation: int = 1, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(limit=limit, interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, p=p)


@Transform.register("shift_scale_rotate")
class ShiftScaleRotate(transforms.ShiftScaleRotate, Transform):
    def __init__(self, shift_limit: Union[Tuple[float, float], float] = 0.0625, scale_limit: Union[Tuple[float, float], float] = 0.1, rotate_limit: Union[Tuple[int, int], int] = 45, interpolation: int = 1, border_mode: int = 4, value: Optional[Union[int, float, List[int], List[float]]] = None, mask_value: Optional[Union[int, float, List[int], List[float]]] = None, always_apply: bool = False, p: float = 0.5):
        super().__init__(shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit, interpolation=interpolation, border_mode=border_mode, value=value, mask_value=mask_value, always_apply=always_apply, p=p)


@Transform.register("smallest_max_size")
class SmallestMaxSize(transforms.SmallestMaxSize, Transform):
    def __init__(self, max_size: int = 1024, interpolation: int = 1, always_apply: bool = False, p: float = 1):
        super().__init__(max_size=max_size, interpolation=interpolation, always_apply=always_apply, p=p)


@Transform.register("solarize")
class Solarize(transforms.Solarize, Transform):
    def __init__(self, threshold: Union[Tuple[float, float], float, Tuple[int, int], int] = 128, always_apply: bool = False, p: float = 0.5):
        super().__init__(threshold=threshold, always_apply=always_apply, p=p)


@Transform.register("to_float")
class ToFloat(transforms.ToFloat, Transform):
    def __init__(self, max_value: Optional[float] = None, always_apply: bool = False, p: float = 1.0):
        super().__init__(max_value=max_value, always_apply=always_apply, p=p)


@Transform.register("to_gray")
class ToGray(transforms.ToGray, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("to_sepia")
class ToSepia(transforms.ToSepia, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("transpose")
class Transpose(transforms.Transpose, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("vertical_flip")
class VerticalFlip(transforms.VerticalFlip, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("base_random_sized_crop")
class _BaseRandomSizedCrop(transforms._BaseRandomSizedCrop, Transform):
    def __init__(self, interpolation: int = 1, always_apply: bool = False, p: float = 1.0):
        super().__init__(interpolation=interpolation, always_apply=always_apply, p=p)
