import torch

from dh_segment_torch.config.params import Params
from dh_segment_torch.models.decoders.decoder import Decoder
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class UnetTest(DhSegmentTestCase):
    def test_encoders_from_params(self):
        available_decoders = Decoder.get_available()
        assert "unet" in available_decoders
        channels = [3, 6, 12, 24, 48]

        xs = [
            torch.zeros(4, c, 512 // (i + 1), 512 // (i + 1)).normal_()
            for i, c in enumerate(channels)
        ]

        n_classes = 8
        params = Params(
            {
                "type": "unet",
                "encoder_channels": channels,
                "decoder_channels": list(reversed(channels)),
                "num_classes": n_classes,
            }
        )
        decoder = Decoder.from_params(params)
        assert decoder.logits.conv2d.out_channels == n_classes
        assert isinstance(decoder.level_ops[0]["up_concat"].deconv, torch.nn.Identity)
        assert (
            decoder.level_ops[0]["decoder_conv"][0].conv2d.in_channels
            == channels[-1] + channels[-2]
        )
        assert decoder.level_ops[0]["decoder_conv"][0].conv2d.out_channels == channels[-1]
        assert (
            decoder.level_ops[1]["decoder_conv"][0].conv2d.in_channels
            == channels[-1] + channels[-3]
        )
        assert decoder.level_ops[1]["decoder_conv"][0].conv2d.out_channels == channels[-2]
        assert (
            decoder.level_ops[2]["decoder_conv"][0].conv2d.in_channels
            == channels[-2] + channels[-4]
        )
        assert decoder.level_ops[2]["decoder_conv"][0].conv2d.out_channels == channels[-3]
        assert (
            decoder.level_ops[3]["decoder_conv"][0].conv2d.in_channels
            == channels[-3] + channels[-5]
        )
        assert decoder.level_ops[3]["decoder_conv"][0].conv2d.out_channels == channels[-4]
        decoder.forward(*xs)

        max_channels = 32
        params = Params(
            {
                "type": "unet",
                "encoder_channels": channels,
                "decoder_channels": list(reversed(channels)),
                "num_classes": n_classes,
                "use_deconvolutions": True,
                "max_channels": max_channels,
                "normalization": {"type": "group_norm", "num_groups": 2},
                "activation": {"type": "swish"},
            }
        )

        decoder = Decoder.from_params(params)
        assert (
            decoder.level_ops[0]["decoder_conv"][0].conv2d.in_channels
            == min(max_channels, channels[-1]) + channels[-2]
        )
        assert isinstance(
            decoder.level_ops[0]["up_concat"].deconv, torch.nn.ConvTranspose2d
        )
        assert decoder.level_ops[0]["decoder_conv"][1]._get_name() == "Swish"
        decoder.forward(*xs)
