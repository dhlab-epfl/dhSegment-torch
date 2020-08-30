import torch

from dh_segment_torch.config import Params
from dh_segment_torch.models import Encoder
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class MobileNetTest(DhSegmentTestCase):
    def test_encoders_from_params(self):
        mobilenet_encoders = ["mobilenetv2"]
        available_encoders = Encoder.get_available()
        for encoder in mobilenet_encoders:
            assert encoder in available_encoders
        params = Params({"type": "mobilenetv2", "blocks": 3})
        encoder = Encoder.from_params(params)
        assert encoder.blocks == 3
        assert len(encoder.output_dims) == 5
        assert encoder.features[-1][0].out_channels == 1280
        x = torch.zeros((2, 3, 256, 256)).normal_()
        encoder.forward(x)

        params = Params({"type": "mobilenetv2", "blocks": 2, "pretrained": False})
        encoder = Encoder.from_params(params)
        assert encoder.blocks == 2
        assert len(encoder.output_dims) == 4
        assert encoder.features[-1][0].out_channels == 1280
        encoder.forward(x)
