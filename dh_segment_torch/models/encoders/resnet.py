from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from pretrainedmodels import pretrained_settings as pretraining
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from dh_segment_torch.models.encoders.encoder import Encoder
from dh_segment_torch.nn.normalizations.normalization import Normalization

imagenet_mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

resnext50_32x4d_pretrained_settings = {
    "url": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    **imagenet_mean_std,
}

resnext101_32x8d_pretrained_settings = {
    "url": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    **imagenet_mean_std,
}

wide_resnet50_2_pretrained_settings = {
    "url": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    **imagenet_mean_std,
}

wide_resnet101_2_pretrained_settings = {
    "url": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
    **imagenet_mean_std,
}


class ResNetEncoder(Encoder, ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        output_dims: List[int],
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = None,
        pretrained: bool = False,
        progress: bool = False,
    ):
        ResNet.__init__(
            self,
            block,
            layers,
            groups=groups,
            width_per_group=width_per_group,
            norm_layer=normalization,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
        Encoder.__init__(self, output_dims, pretrained_settings, pretrained, progress)
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = super().normalize_if_pretrained(x)

        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

        features_maps = [x]
        for layer in layers[: self.blocks + 1]:
            x = layer(x)
            features_maps.append(x)

        return features_maps

    @classmethod
    def resnet18(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretraining["resnet18"]["imagenet"],
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNet-18 model.
        From `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return cls(
            block=BasicBlock,
            layers=[2, 2, 2, 2],
            output_dims=[3, 64, 64, 128, 256, 512][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet34(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretraining["resnet34"]["imagenet"],
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNet-34 model.
        From `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return cls(
            block=BasicBlock,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 64, 128, 256, 512][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet50(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretraining["resnet50"]["imagenet"],
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNet-50 model.
        From `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return cls(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet101(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretraining["resnet101"]["imagenet"],
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNet-101 model.
        From `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return cls(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnet152(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretraining["resnet152"]["imagenet"],
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNet-152 model.
        From `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
        """
        return cls(
            block=Bottleneck,
            layers=[3, 8, 36, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnext50_32x4d(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = resnext50_32x4d_pretrained_settings,
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNeXt-50 32x4d model.
        From `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_"""
        return cls(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            groups=32,
            width_per_group=4,
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def resnext101_32x8d(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = resnext101_32x8d_pretrained_settings,
        pretrained: bool = True,
        progress: bool = False,
    ):
        """ResNeXt-101 32x8d model.
        From `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_"""
        return cls(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            groups=32,
            width_per_group=8,
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def wide_resnet50_2(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = wide_resnet50_2_pretrained_settings,
        pretrained: bool = True,
        progress: bool = False,
    ):
        """Wide ResNet-50-2 model.
        From `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_"""
        return cls(
            block=Bottleneck,
            layers=[3, 4, 6, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            width_per_group=64 * 2,
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )

    @classmethod
    def wide_resnet101_2(
        cls,
        replace_stride_with_dilation: Optional[Tuple[bool, bool, bool]] = None,
        normalization: Normalization = None,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = wide_resnet101_2_pretrained_settings,
        pretrained: bool = True,
        progress: bool = False,
    ):
        """Wide ResNet-101-2 model.
        From `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_"""
        return cls(
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            output_dims=[3, 64, 256, 512, 1024, 2048][: blocks + 2],
            width_per_group=64*2,
            replace_stride_with_dilation=replace_stride_with_dilation,
            normalization=normalization,
            pretrained_settings=pretrained_settings,
            blocks=blocks,
            pretrained=pretrained,
            progress=progress,
        )


Encoder.register("resnet18", "resnet18")(ResNetEncoder)
Encoder.register("resnet34", "resnet34")(ResNetEncoder)
Encoder.register("resnet50", "resnet50")(ResNetEncoder)
Encoder.register("resnet101", "resnet101")(ResNetEncoder)
Encoder.register("resnet152", "resnet152")(ResNetEncoder)
Encoder.register("resnext50_32x4d", "resnext50_32x4d")(ResNetEncoder)
Encoder.register("resnext101_32x8d", "resnext101_32x8d")(ResNetEncoder)
Encoder.register("wide_resnet50_2", "wide_resnet50_2")(ResNetEncoder)
Encoder.register("wide_resnet101_2", "wide_resnet101_2")(ResNetEncoder)
