from abc import ABC

import torch.nn as nn


class SegmentationModel(nn.Module):

    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        feature_maps = self.encoder(x)
        network_output = self.decoder(*feature_maps)

        return network_output

