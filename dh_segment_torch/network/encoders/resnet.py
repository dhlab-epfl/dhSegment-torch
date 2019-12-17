from typing import List, Dict

import torch.nn as nn
from pretrainedmodels import pretrained_settings
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import ResNet, Bottleneck, BasicBlock


class ResNetEncoder(ResNet):

    def __init__(self, output_dims: List[int], pretrained_settings: Dict[str, object], blocks: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks
        self.output_dims = output_dims  # Maybe refactor in upper class
        self.pretrained_settings = pretrained_settings

    def forward(self, x):
        # TODO check where we normalize the image, I do not like that the normalization is done in the encoder
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


def _resnet(block, layers, output_dims, blocks, pretrained_settings, pretrained, progress, **kwargs):
    model = ResNetEncoder(output_dims, pretrained_settings, blocks, block=block, layers=layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(pretrained_settings['url'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18Encoder(blocks: int = 4, pretrained: bool = True, progress: bool = False) -> nn.Module:
    return _resnet(BasicBlock, [2, 2, 2, 2],
                   [3, 64, 64, 128, 256, 512][:blocks+2], blocks,
                   pretrained_settings['resnet18']['imagenet'],
                   pretrained, progress
                   )


def resnet34Encoder(blocks: int = 4, pretrained: bool = True, progress: bool = False) -> nn.Module:
    return _resnet(BasicBlock, [3, 4, 6, 3],
                   [3, 64, 64, 128, 256, 512][:blocks+2], blocks,
                   pretrained_settings['resnet34']['imagenet'],
                   pretrained, progress
                   )


def resnet50Encoder(blocks: int = 4, pretrained: bool = True, progress: bool = False) -> nn.Module:
    return _resnet(Bottleneck, [3, 4, 6, 3],
                   [3, 64, 256, 512, 1024, 2048][:blocks+2], blocks,
                   pretrained_settings['resnet50']['imagenet'],
                   pretrained, progress
                   )



