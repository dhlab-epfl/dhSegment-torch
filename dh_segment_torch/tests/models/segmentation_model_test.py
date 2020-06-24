import torch

from dh_segment_torch.config import Params
from dh_segment_torch.models import Model, ResNetEncoder, PanDecoder
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class SegmentationModelTest(DhSegmentTestCase):
    def test_encoders_from_params(self):

        blocks = 2
        num_channels = 32
        n_classes = 2
        params = Params(
            {
                "encoder": {
                    "type": "resnet50",
                    "blocks": blocks,
                    "pretrained": False,
                    "replace_stride_with_dilation": [False, True, True],
                    "normalization": {"type": "identity"},
                },
                "decoder": {
                    "type": "pan",
                    "decoder_channels_size": num_channels,
                    "normalization": {"type": "batch_renorm_2d"},
                    "activation": {"type": "leaky_relu", "inplace": True},
                    "gau_activation": {"type": "swish"},
                    "upscale_mode": "nearest",
                },
                "loss": {"type": "dice"},
                "num_classes": n_classes,
            }
        )
        x = torch.zeros((2, 3, 128, 128)).normal_()
        y = (torch.zeros((2, n_classes, 128, 128)).normal_() > 0.5).to(torch.float)
        model = Model.from_params(params)
        res = model.forward(x, y)
        assert res["loss"] > 0
        assert res["logits"].shape[1] == n_classes
        assert isinstance(model.encoder, ResNetEncoder)
        assert isinstance(model.decoder, PanDecoder)
