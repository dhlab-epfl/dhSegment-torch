from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNorm2dDrop(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.shape[0] == 1:
            return input
        else:
            return super().forward(input)


class Conv2DNormalizeActivate(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        normalizer_fn: nn.Module = nn.Identity,
        activation_fn: nn.Module = nn.Identity,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        super(Conv2DNormalizeActivate, self).__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups
            ),
            activation_fn(),
            normalizer_fn(out_channels),
        )


class FPABlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        normalizer_fn: nn.Module = nn.Identity,
        upscale_mode: str = "bilinear",
    ):
        super().__init__()

        self.upscale_params = dict(
            mode=upscale_mode, align_corners=upscale_mode == "bilinear"
        )

        relu = partial(nn.ReLU, inplace=True)

        self.pooling_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2DNormalizeActivate(
                in_channels,
                out_channels,
                kernel_size=1,
                normalizer_fn=normalizer_fn,
                activation_fn=relu,
            ),
        )

        self.middle_branch = Conv2DNormalizeActivate(
            in_channels,
            out_channels=1,
            kernel_size=1,
            normalizer_fn=normalizer_fn,
            activation_fn=relu,
        )

        self.conv1_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2DNormalizeActivate(
                in_channels,
                out_channels=1,
                kernel_size=7,
                normalizer_fn=normalizer_fn,
                activation_fn=relu,
            ),
        )

        self.conv2_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2DNormalizeActivate(
                in_channels=1,
                out_channels=1,
                kernel_size=5,
                normalizer_fn=normalizer_fn,
                activation_fn=relu,
            ),
        )

        self.conv3_down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            Conv2DNormalizeActivate(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                normalizer_fn=normalizer_fn,
                activation_fn=relu,
            ),
            Conv2DNormalizeActivate(
                in_channels=1,
                out_channels=1,
                kernel_size=3,
                normalizer_fn=normalizer_fn,
                activation_fn=relu,
            ),
        )

        self.conv2_up = Conv2DNormalizeActivate(
            in_channels=1,
            out_channels=1,
            kernel_size=5,
            normalizer_fn=normalizer_fn,
            activation_fn=relu,
        )

        self.conv1_up = Conv2DNormalizeActivate(
            in_channels=1,
            out_channels=1,
            kernel_size=7,
            normalizer_fn=normalizer_fn,
            activation_fn=relu,
        )

    def forward(self, x):
        height, width = x.shape[-2:]

        pooling_branch = self.pooling_branch(x)
        pooling_branch = F.interpolate(
            pooling_branch, size=(height, width), **self.upscale_params
        )

        middle_branch = self.middle_branch(x)

        down1 = self.conv1_down(x)
        down2 = self.conv2_down(down1)
        down3 = self.conv3_down(down2)
        up3 = F.interpolate(
            down3, size=(height // 4, width // 4), **self.upscale_params
        )

        up2 = self.conv2_up(down2)
        res = up2 + up3
        res = F.interpolate(res, size=(height // 2, width // 2), **self.upscale_params)

        up1 = self.conv1_up(down1)
        res = res + up1
        res = F.interpolate(res, size=(height, width), **self.upscale_params)

        res = torch.mul(res, middle_branch)

        res = res + pooling_branch

        return res


class GAUBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        normalizer_fn: nn.Module = nn.Identity,
        upscale_mode: str = "bilinear",
        use_sigmoid: bool = False,
    ):
        super().__init__()

        self.upscale_params = dict(
            mode=upscale_mode, align_corners=upscale_mode == "bilinear"
        )

        relu = partial(nn.ReLU, inplace=True)
        activation_high = nn.Sigmoid if use_sigmoid else relu
        self.process_high = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2DNormalizeActivate(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                normalizer_fn=normalizer_fn,
                activation_fn=activation_high,
            ),
        )

        self.process_low = Conv2DNormalizeActivate(
            in_channels,
            out_channels,
            kernel_size=3,
            normalizer_fn=normalizer_fn,
            activation_fn=relu,
        )

    def forward(self, low, high):
        target_shape = low.shape[-2:]
        high_up = F.interpolate(high, target_shape, **self.upscale_params)
        low = self.process_low(low)
        high = self.process_high(high)
        res = torch.mul(low, high)
        res = res + high_up

        return res


class PanDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        n_classes: int,
        use_batchnorm: bool = True,
        upscale_mode: str = "bilinear",
        use_sigmoid: bool = False,
    ):
        super().__init__()

        batch_norm_decay = 0.999

        if use_batchnorm:
            normalizer_fn = partial(BatchNorm2dDrop, momentum=1 - batch_norm_decay)
        else:
            normalizer_fn = nn.Identity

        self.upscale_params = dict(
            mode=upscale_mode, align_corners=upscale_mode == "bilinear"
        )

        encoder_channels = list(reversed(encoder_channels))
        self.fpa = FPABlock(
            in_channels=encoder_channels[0],
            out_channels=decoder_channels[0],
            normalizer_fn=normalizer_fn,
            upscale_mode=upscale_mode,
        )
        self.gau1 = GAUBlock(
            in_channels=encoder_channels[1],
            out_channels=decoder_channels[1],
            normalizer_fn=normalizer_fn,
            upscale_mode=upscale_mode,
            use_sigmoid=use_sigmoid,
        )
        self.gau2 = GAUBlock(
            in_channels=encoder_channels[2],
            out_channels=decoder_channels[2],
            normalizer_fn=normalizer_fn,
            upscale_mode=upscale_mode,
            use_sigmoid=use_sigmoid,
        )
        self.gau3 = GAUBlock(
            in_channels=encoder_channels[3],
            out_channels=decoder_channels[3],
            normalizer_fn=normalizer_fn,
            upscale_mode=upscale_mode,
            use_sigmoid=use_sigmoid,
        )

        self.logits = Conv2DNormalizeActivate(
            decoder_channels[3], n_classes, kernel_size=1
        )

    def forward(self, *features_maps):
        features_maps = list(reversed(features_maps))

        x = self.fpa(features_maps[0])
        x = self.gau1(features_maps[1], x)
        x = self.gau2(features_maps[2], x)
        x = self.gau3(features_maps[3], x)

        x = self.logits(x)

        target_shape = features_maps[-1].shape[-2:]
        x = F.interpolate(x, target_shape, **self.upscale_params)

        return x
