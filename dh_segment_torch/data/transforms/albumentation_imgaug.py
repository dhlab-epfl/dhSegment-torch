from typing import Tuple, Optional

from albumentations.imgaug import transforms as iaa

from dh_segment_torch.data.transforms.transform import Transform


@Transform.register("iaa_additive_gaussian_noise")
class IAAAdditiveGaussianNoise(iaa.IAAAdditiveGaussianNoise, Transform):
    def __init__(
        self,
        loc: int = 0,
        scale: Tuple[float, float] = (2.55, 12.75),
        per_channel: bool = False,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            loc=loc,
            scale=scale,
            per_channel=per_channel,
            always_apply=always_apply,
            p=p,
        )


@Transform.register("iaa_affine")
class IAAAffine(iaa.IAAAffine, Transform):
    def __init__(
        self,
        scale: float = 1.0,
        translate_percent: Optional[float] = None,
        translate_px: Optional[int] = None,
        rotate: float = 0.0,
        shear: float = 0.0,
        order: int = 1,
        cval: int = 0,
        mode: str = "reflect",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            scale=scale,
            translate_percent=translate_percent,
            translate_px=translate_px,
            rotate=rotate,
            shear=shear,
            order=order,
            cval=cval,
            mode=mode,
            always_apply=always_apply,
            p=p,
        )


@Transform.register("iaa_crop_and_pad")
class IAACropAndPad(iaa.IAACropAndPad, Transform):
    def __init__(
        self,
        px: Optional[int] = None,
        percent: Optional[float] = None,
        pad_mode: str = "constant",
        pad_cval: int = 0,
        keep_size: bool = True,
        always_apply: bool = False,
        p: float = 1,
    ):
        super().__init__(
            px=px,
            percent=percent,
            pad_mode=pad_mode,
            pad_cval=pad_cval,
            keep_size=keep_size,
            always_apply=always_apply,
            p=p,
        )


@Transform.register("iaa_emboss")
class IAAEmboss(iaa.IAAEmboss, Transform):
    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        strength: Tuple[float, float] = (0.2, 0.7),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(alpha=alpha, strength=strength, always_apply=always_apply, p=p)


@Transform.register("iaa_fliplr")
class IAAFliplr(iaa.IAAFliplr, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("iaa_flipud")
class IAAFlipud(iaa.IAAFlipud, Transform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply=always_apply, p=p)


@Transform.register("iaa_perspective")
class IAAPerspective(iaa.IAAPerspective, Transform):
    def __init__(
        self,
        scale: Tuple[float, float] = (0.05, 0.1),
        keep_size: bool = True,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            scale=scale, keep_size=keep_size, always_apply=always_apply, p=p
        )


@Transform.register("iaa_piecewise_affine")
class IAAPiecewiseAffine(iaa.IAAPiecewiseAffine, Transform):
    def __init__(
        self,
        scale: Tuple[float, float] = (0.03, 0.05),
        nb_rows: int = 4,
        nb_cols: int = 4,
        order: int = 1,
        cval: int = 0,
        mode: str = "constant",
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            scale=scale,
            nb_rows=nb_rows,
            nb_cols=nb_cols,
            order=order,
            cval=cval,
            mode=mode,
            always_apply=always_apply,
            p=p,
        )


@Transform.register("iaa_sharpen")
class IAASharpen(iaa.IAASharpen, Transform):
    def __init__(
        self,
        alpha: Tuple[float, float] = (0.2, 0.5),
        lightness: Tuple[float, float] = (0.5, 1.0),
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            alpha=alpha, lightness=lightness, always_apply=always_apply, p=p
        )


@Transform.register("iaa_superpixels")
class IAASuperpixels(iaa.IAASuperpixels, Transform):
    def __init__(
        self,
        p_replace: float = 0.1,
        n_segments: int = 100,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(
            p_replace=p_replace, n_segments=n_segments, always_apply=always_apply, p=p
        )
