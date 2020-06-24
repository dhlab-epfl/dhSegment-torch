from typing import List, Dict, Optional, Union, Tuple

import torch
import torch.nn as nn
from pretrainedmodels import pretrained_settings as pretraining
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from dh_segment_torch.models.encoders.encoder import Encoder
from dh_segment_torch.nn.normalizations.normalization import Normalization


class ResNetEncoder(Encoder, ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: List[int],
        output_dims: List[int],
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
            norm_layer=normalization,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )
        Encoder.__init__(self, output_dims, pretrained_settings, pretrained, progress)
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = Encoder.forward(self, x)

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


Encoder.register("resnet18", "resnet18")(ResNetEncoder)
Encoder.register("resnet34", "resnet34")(ResNetEncoder)
Encoder.register("resnet50", "resnet50")(ResNetEncoder)
