from dh_segment_torch.data.transforms.albumentation import *
from dh_segment_torch.data.transforms.albumentation_imgaug import *
from dh_segment_torch.data.transforms.assign_labels import *
from dh_segment_torch.data.transforms.extract_patches import *
from dh_segment_torch.data.transforms.fixed_resize import *
from dh_segment_torch.data.transforms.fixed_size_resize import *
from dh_segment_torch.data.transforms.fixed_size_rotate_crop import *
from dh_segment_torch.data.transforms.rotate_no_crop import *
from dh_segment_torch.data.transforms.transform import *

_ASSIGN = [
    "Assign",
    "AssignLabel",
    "AssignMultilabel"
]

_TRANSFORM = [
    "Transform",
    "SampleToPatches",
    "FixedResize",
    "FixedSizeResize",
    "FixedSizeRotateCrop",
    "RotateNoCrop",
]


_ALBUMENTATION_TRANSFORMS = [
    "Blur",
    "CLAHE",
    "CenterCrop",
    "ChannelDropout",
    "ChannelShuffle",
    "CoarseDropout",
    "Compose",
    "Crop",
    "CropNonEmptyMaskIfExists",
    "Cutout",
    "Downscale",
    "DualTransform",
    "ElasticTransform",
    "Equalize",
    "FancyPCA",
    "Flip",
    "FromFloat",
    "GaussNoise",
    "GaussianBlur",
    "GlassBlur",
    "GridDistortion",
    "GridDropout",
    "HorizontalFlip",
    "HueSaturationValue",
    "ISONoise",
    "ImageCompression",
    "ImageOnlyTransform",
    "InvertImg",
    "JpegCompression",
    "Lambda",
    "LongestMaxSize",
    "MaskDropout",
    "MedianBlur",
    "MotionBlur",
    "MultiplicativeNoise",
    "NoOp",
    "Normalize",
    "OneOf",
    "OpticalDistortion",
    "PadIfNeeded",
    "Posterize",
    "RGBShift",
    "RandomBrightnessContrast",
    "RandomCrop",
    "RandomCropNearBBox",
    "RandomFog",
    "RandomGamma",
    "RandomGridShuffle",
    "RandomRain",
    "RandomResizedCrop",
    "RandomRotate90",
    "RandomScale",
    "RandomShadow",
    "RandomSizedBBoxSafeCrop",
    "RandomSizedCrop",
    "RandomSnow",
    "RandomSunFlare",
    "Resize",
    "Rotate",
    "ShiftScaleRotate",
    "SmallestMaxSize",
    "Solarize",
    "ToFloat",
    "ToGray",
    "ToSepia",
    "Transpose",
    "VerticalFlip",
    "_BaseRandomSizedCrop",
]

_IMG_AUG_TRANSFORMS = [
    "IAAAdditiveGaussianNoise",
    "IAAAffine",
    "IAACropAndPad",
    "IAAEmboss",
    "IAAFliplr",
    "IAAFlipud",
    "IAAPerspective",
    "IAAPiecewiseAffine",
    "IAASharpen",
    "IAASuperpixels",
]


_ALL_TRANSFORMS = _TRANSFORM + _ALBUMENTATION_TRANSFORMS + _IMG_AUG_TRANSFORMS

__all__ = _ALL_TRANSFORMS + _ASSIGN
