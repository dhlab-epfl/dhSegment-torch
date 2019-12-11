from functools import partial
from typing import List, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, normalizer_fn: nn.Module = nn.Identity, **kwargs):
        super().__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        normalizer = normalizer_fn(in_channels)
        super(Conv2D, self).__init__(conv, normalizer)


class UpsampleConcat(nn.Module):
    def __init__(self, mode: str = 'nearest'):
        super().__init__()
        self.mode = mode

    def forward(self, x, x_skip):
        target_shape = x_skip.shape[-2:]
        x = F.interpolate(x, target_shape, mode=self.mode, align_corners=False)  # TODO check align corners
        x = torch.cat([x, x_skip], dim=1)
        return x


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels: Iterable[int], decoder_channels: Iterable[int], n_classes: int,
                 use_deconvolutions: bool = False,
                 max_channels: int = None, use_batchnorm: bool = True):
        super().__init__()

        self.use_deconvolutions = use_deconvolutions

        batch_norm_decay = 0.999

        if use_batchnorm:
            normalizer_fn = partial(nn.BatchNorm2d, momentum=1 - batch_norm_decay)
        else:
            normalizer_fn = nn.Identity

        self.reduce_dims = nn.ModuleList()
        encoder_channels_reduced = []
        for dim in reversed(encoder_channels):
            if max_channels and dim > max_channels:
                self.reduce_dims.append(Conv2D(dim, max_channels, 1, normalizer_fn))
                encoder_channels_reduced.append(max_channels)
            else:
                self.reduce_dims.append(nn.Identity())
                encoder_channels_reduced.append(dim)

        self.upsample_concat = UpsampleConcat('bilinear')  # Hardcoded

        self.decoder_convs = nn.ModuleList()

        prev_channels = encoder_channels_reduced[0]
        for enc_channels, dec_channels in zip(encoder_channels_reduced[1:], decoder_channels):
            conv = Conv2D(prev_channels + enc_channels, dec_channels, 3, normalizer_fn, padding=1)
            relu = nn.ReLU()

            self.decoder_convs.append(nn.Sequential(conv, relu))

            prev_channels = dec_channels

        self.logits = Conv2D(prev_channels, n_classes, 1)

    def forward(self, *features):
        features = list(reversed(features))
        x = features[0]
        x = self.reduce_dims[0](x)

        for i, x_skip in enumerate(features[1:]):
            x_skip = self.reduce_dims[i](x_skip)

            x = self.upsample_concat(x, x_skip)
            x = self.decoder_convs[i](x)

        x = self.logits(x)
        return x
