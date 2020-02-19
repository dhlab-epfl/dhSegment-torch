from functools import partial
from typing import List, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2DNormalize(nn.Sequential):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, groups=1,
                 normalizer_fn: nn.Module = nn.Identity):
        super().__init__()
        padding = (kernel_size - 1) // 2
        super(Conv2DNormalize, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            normalizer_fn(out_channels)
        )


class UpsampleConcat(nn.Module):
    def __init__(self, mode: str = 'nearest', use_deconv: bool = False, x_channels: int = None):
        super().__init__()
        self.mode = mode
        if use_deconv:
            self.deconv = nn.ConvTranspose2d(x_channels, x_channels, kernel_size=2, stride=2)
        else:
            self.deconv = nn.Identity()

    def forward(self, x, x_skip):
        x = self.deconv(x)
        target_shape = x_skip.shape[-2:]
        x = F.interpolate(x, target_shape, mode=self.mode, align_corners=False)  # TODO check align corners
        x = torch.cat([x, x_skip], dim=1)
        return x


def get_channels_reduce(channels, max_channels: int = None, normalizer_fn: nn.Module = nn.Identity) -> (nn.Module, int):
    """
    This functions creates a module that reduces the given input channels to the max number of channels
    or leave it untouched if it is smaller
    :param channels:
    :param max_channels:
    :param normalizer_fn:
    :return:
    """
    if max_channels and channels > max_channels:
        reduce = Conv2DNormalize(channels, max_channels, kernel_size=1, normalizer_fn=normalizer_fn)
    else:
        reduce = nn.Identity()
    return reduce, min(channels, max_channels) if max_channels else channels


class UnetDecoder(nn.Module):

    def __init__(self, encoder_channels: List[int], decoder_channels: List[int], n_classes: int,
                 use_deconvolutions: bool = False,
                 max_channels: int = None, use_batchnorm: bool = True):
        super().__init__()

        self.use_deconvolutions = use_deconvolutions

        batch_norm_decay = 0.999

        if use_batchnorm:
            normalizer_fn = partial(nn.BatchNorm2d, momentum=1 - batch_norm_decay)
        else:
            normalizer_fn = nn.Identity

        encoder_channels = list(reversed(encoder_channels))
        output_encoder_channels = encoder_channels[0]
        self.reduce_output_encoder, output_encoder_channels = get_channels_reduce(
            output_encoder_channels, max_channels, normalizer_fn)

        self.level_ops = nn.ModuleList()

        prev_channels = output_encoder_channels
        for enc_channels, dec_channels in zip(encoder_channels[1:], decoder_channels):
            ops = {}

            ops['reduce_dim'], enc_channels = get_channels_reduce(enc_channels, max_channels, normalizer_fn)

            ops['up_concat'] = UpsampleConcat('bilinear', use_deconvolutions, prev_channels)

            conv = Conv2DNormalize(prev_channels + enc_channels, dec_channels, normalizer_fn=normalizer_fn)
            relu = nn.ReLU(inplace=True)

            ops['decoder_conv'] = nn.Sequential(conv, relu)

            self.level_ops.append(nn.ModuleDict(ops))

            prev_channels = dec_channels

        self.logits = Conv2DNormalize(prev_channels, n_classes, 1)

    def forward(self, *features_maps):
        features_maps = list(reversed(features_maps))
        x = features_maps[0]
        x = self.reduce_output_encoder(x)

        for x_skip, level_op in zip(features_maps[1:], self.level_ops):
            x_skip = level_op['reduce_dim'](x_skip)
            x = level_op['up_concat'](x, x_skip)
            x = level_op['decoder_conv'](x)
        x = self.logits(x)
        return x
