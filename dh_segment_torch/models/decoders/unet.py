from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dh_segment_torch.models.decoders.decoder import Decoder
from dh_segment_torch.nn.activations import Activation
from dh_segment_torch.nn.normalizations.normalization import Normalization


@Decoder.register("unet")
class UnetDecoder(Decoder):
    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
        use_deconvolutions: bool = False,
        max_channels: Optional[int] = None,
        normalization: Normalization = None,
        activation: Activation = None,
    ):
        super().__init__(encoder_channels, decoder_channels, num_classes)

        self.use_deconvolutions = use_deconvolutions

        # TODO add value to default config batch_norm_decay = 0.999
        if normalization is None:
            normalization = Normalization.get_constructor("batch_norm_2d")(
                momentum=1 - 0.999
            )

        if activation is None:
            activation = Activation.get_constructor("leaky_relu")(inplace=True)

        encoder_channels = list(reversed(encoder_channels))
        output_encoder_channels = encoder_channels[0]
        self.reduce_output_encoder, output_encoder_channels = get_channels_reduce(
            output_encoder_channels, max_channels, normalization
        )

        self.level_ops = nn.ModuleList()

        prev_channels = output_encoder_channels
        for enc_channels, dec_channels in zip(encoder_channels[1:], decoder_channels):
            ops = {}

            ops["reduce_dim"], enc_channels = get_channels_reduce(
                enc_channels, max_channels, normalization
            )

            ops["up_concat"] = UpsampleConcat(
                "bilinear", use_deconvolutions, prev_channels
            )

            conv = Conv2DNormalize(
                prev_channels + enc_channels, dec_channels, normalization=normalization
            )

            ops["decoder_conv"] = nn.Sequential(conv, activation)

            self.level_ops.append(nn.ModuleDict(ops))

            prev_channels = dec_channels

        self.logits = Conv2DNormalize(prev_channels, num_classes, 1)

    def forward(self, *features_maps):
        features_maps = list(reversed(features_maps))
        x = features_maps[0]
        x = self.reduce_output_encoder(x)

        for x_skip, level_op in zip(features_maps[1:], self.level_ops):
            x_skip = level_op["reduce_dim"](x_skip)
            x = level_op["up_concat"](x, x_skip)
            x = level_op["decoder_conv"](x)
        x = self.logits(x)
        return x


class Conv2DNormalize(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        groups=1,
        normalization: Normalization = nn.Identity,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv2d = nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups
            )
        self.normalization = normalization(out_channels)

    def forward(self, x: torch.Tensor):
        return self.normalization(self.conv2d(x))


class UpsampleConcat(nn.Module):
    def __init__(
        self,
        upscale_mode: str = "nearest",
        use_deconv: bool = False,
        x_channels: int = None,
    ):
        super().__init__()
        if use_deconv:
            self.deconv = nn.ConvTranspose2d(
                x_channels, x_channels, kernel_size=2, stride=2
            )
        else:
            self.deconv = nn.Identity()

        self.upscale_params = dict(
            mode=upscale_mode, align_corners=upscale_mode == "bilinear"
        )

    def forward(self, x, x_skip):
        x = self.deconv(x)
        target_shape = x_skip.shape[-2:]
        x = F.interpolate(
            x, target_shape, **self.upscale_params
        )  # TODO check align corners
        x = torch.cat([x, x_skip], dim=1)
        return x


def get_channels_reduce(
    channels, max_channels: int = None, normalization: Normalization = nn.Identity,
) -> (nn.Module, int):
    """
    This functions creates a module that reduces the given input channels to the max number of channels
    or leave it untouched if it is smaller
    :param channels:
    :param max_channels:
    :param normalization:
    :return:
    """
    if max_channels and channels > max_channels:
        reduce = Conv2DNormalize(
            channels, max_channels, kernel_size=1, normalization=normalization
        )
    else:
        reduce = nn.Identity()
    return reduce, min(channels, max_channels) if max_channels else channels
