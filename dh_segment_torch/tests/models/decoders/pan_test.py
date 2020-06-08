import pytest
import torch

from dh_segment_torch.config import ConfigurationError
from dh_segment_torch.config.params import Params
from dh_segment_torch.models.decoders.decoder import Decoder
from dh_segment_torch.nn.normalization.batch_renorm import BatchRenorm2d
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class PanTest(DhSegmentTestCase):
    def test_encoders_from_params(self):
        available_decoders = Decoder.get_available()
        assert "pan" in available_decoders
        channels = [3, 6, 12, 24, 48]
        num_channels = 16

        xs = [
            torch.zeros(4, c, 100 // (i + 1), 100 // (i + 1)).normal_()
            for i, c in enumerate(channels)
        ]

        n_classes = 8
        params = Params(
            {
                "type": "pan",
                "encoder_channels": channels,
                "decoder_channels_size": num_channels,
                "num_classes": n_classes,
            }
        )
        decoder = Decoder.from_params(params)
        assert decoder.fpa.pooling_branch[1][0].in_channels == channels[-1]
        assert decoder.fpa.pooling_branch[1][0].out_channels == num_channels
        assert decoder.gau3.process_high[1][0].in_channels == num_channels
        assert decoder.gau3.process_high[1][0].out_channels == num_channels
        assert decoder.gau3.process_low[0].in_channels == channels[-4]
        assert decoder.gau3.process_low[0].out_channels == num_channels
        assert decoder.logits[0].in_channels == num_channels
        assert decoder.logits[0].out_channels == n_classes
        decoder.forward(*xs)

        params = Params(
            {
                "type": "pan",
                "encoder_channels": channels,
                "decoder_channels_size": num_channels,
            }
        )

        with pytest.raises(ConfigurationError):
            Decoder.from_params(params)

        params = Params(
            {
                "type": "pan",
                "encoder_channels": channels,
                "decoder_channels_size": num_channels,
                "num_classes": n_classes,
                "normalization": {"type": "batch_renorm_2d"},
                "activation": {"type": "leaky_relu", "inplace": True},
                "gau_activation": {"type": "swish"},
                "upscale_mode": "nearest",
            }
        )

        decoder = Decoder.from_params(params)

        assert isinstance(decoder.gau3.process_low[2], BatchRenorm2d)
        assert isinstance(decoder.gau3.process_low[1], torch.nn.LeakyReLU)
        assert decoder.gau3.process_high[1][1]._get_name() == "Swish"
        decoder.forward(*xs)
