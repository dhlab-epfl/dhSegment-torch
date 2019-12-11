from typing import List

import torch.nn as nn
from torchvision.models.resnet import ResNet


class ResNetEncoder(ResNet):

    def __init__(self, output_dims: List[int], blocks: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks
        self.output_dims = output_dims  # Maybe refactor in upper class

    def forward(self, x):
        # TODO check where we normalize the image, I do not like that the normalization is done in the encoder
        layers = [
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4
        ]

        features = [x]
        for layer in layers[:self.blocks + 1]:
            x = layer(x)
            features.append(x)

        return features
