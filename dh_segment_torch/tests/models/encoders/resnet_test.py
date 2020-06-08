import torch

from dh_segment_torch.config.params import Params
from dh_segment_torch.models.encoders.encoder import Encoder
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class ResnetTest(DhSegmentTestCase):
    def test_encoders_from_params(self):
        resnet_encoders = ["resnet50", "resnet34", "resnet18"]
        available_encoders = Encoder.get_available()
        for encoder in resnet_encoders:
            assert encoder in available_encoders
        params = Params({"type": "resnet50", "blocks": 3})
        encoder = Encoder.from_params(params)
        assert encoder.blocks == 3
        assert len(encoder.output_dims) == 5
        assert encoder.layer4[-1].conv1.in_channels == 2048
        x = torch.zeros((2, 3, 128, 128)).normal_()
        encoder.forward(x)

        params = Params({"type": "resnet34", "blocks": 2})
        encoder = Encoder.from_params(params)
        assert encoder.blocks == 2
        assert len(encoder.output_dims) == 4
        assert len(encoder.layer4) == 3
        assert encoder.layer4[-1].conv1.in_channels == 512
        encoder.forward(x)

        params = Params({"type": "resnet18"})
        encoder = Encoder.from_params(params)
        assert len(encoder.layer4) == 2
        assert encoder.layer4[-1].conv1.in_channels == 512
        encoder.forward(x)

        params = Params(
            {
                "type": "resnet50",
                "blocks": 3,
                "pretrained": False,
                "replace_stride_with_dilation": [False, True, True],
                "normalization": {"type": "identity"},
            }
        )
        encoder = Encoder.from_params(params)
        assert isinstance(encoder.layer4[0].bn1, torch.nn.Identity)
        assert encoder.layer4[1].conv2.dilation == (4, 4)
        encoder.forward(x)
