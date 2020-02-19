from typing import List, Dict

import torch.nn as nn
from pretrainedmodels import pretrained_settings
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock

from .utils import normalize_batch


class ResNetEncoder(ResNet):

    def __init__(self, output_dims: List[int], pretrained_settings: Dict[str, object], blocks: int = 4,
                 pretrained: bool = True, progress: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks
        self.output_dims = output_dims  # Maybe refactor in upper class
        self.pretrained_settings = pretrained_settings
        self.pretrained = pretrained
        if pretrained:
            state_dict = load_state_dict_from_url(pretrained_settings['url'], progress=progress)
            self.load_state_dict(state_dict)

    def forward(self, x):
        if self.pretrained:
            x = normalize_batch(x, self.pretrained_settings['mean'], self.pretrained_settings['std'])

        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4
        ]

        features_maps = [x]
        for layer in layers[:self.blocks + 1]:
            x = layer(x)
            features_maps.append(x)

        return features_maps


class Resnet18Encoder(ResNetEncoder):
    def __init__(self, blocks: int = 4, pretrained: bool = True, progress: bool = False, **kwargs):
        super().__init__(output_dims=[3, 64, 64, 128, 256, 512][:blocks+2],
                         pretrained_settings=pretrained_settings['resnet18']['imagenet'],
                         blocks=blocks,
                         pretrained=pretrained,
                         progress=progress,
                         block=BasicBlock,
                         layers=[2, 2, 2, 2],
                         **kwargs
                         )


class Resnet34Encoder(ResNetEncoder):
    def __init__(self, blocks: int = 4, pretrained: bool = True, progress: bool = False, **kwargs):
        super().__init__(output_dims=[3, 64, 64, 128, 256, 512][:blocks+2],
                         pretrained_settings=pretrained_settings['resnet34']['imagenet'],
                         blocks=blocks,
                         pretrained=pretrained,
                         progress=progress,
                         block=BasicBlock,
                         layers=[3, 4, 6, 3],
                         **kwargs
                         )


class Resnet50Encoder(ResNetEncoder):
    def __init__(self, blocks: int = 4, pretrained: bool = True, progress: bool = False, **kwargs):
        super().__init__(output_dims=[3, 64, 256, 512, 1024, 2048][:blocks+2],
                         pretrained_settings=pretrained_settings['resnet50']['imagenet'],
                         blocks=blocks,
                         pretrained=pretrained,
                         progress=progress,
                         block=Bottleneck,
                         layers=[3, 4, 6, 3],
                         **kwargs
                         )



