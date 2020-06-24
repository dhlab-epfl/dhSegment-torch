import pytest
import torch
from torch.nn import BatchNorm2d, GroupNorm, Identity

from dh_segment_torch.config import ConfigurationError, Params
from dh_segment_torch.models import Encoder
from dh_segment_torch.nn import Normalization
from dh_segment_torch.nn.normalizations.batch_renorm import BatchRenorm2d
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class NormalizationTest(DhSegmentTestCase):
    def test_normalization_building(self):
        all_norms = ["identity", "batch_norm_2d", "batch_renorm_2d", "group_norm"]
        in_features = 2
        x = torch.zeros((2, in_features, 4, 4)).normal_()
        for norm in all_norms:
            cls, constructor = Normalization.get(norm)
            if constructor.__name__ == "__init__":
                constructor = cls
            if norm == "group_norm":
                with pytest.raises(TypeError):
                    normalizer = constructor()(in_features)
                normalizer = constructor(num_groups=2)(in_features)
                normalizer = constructor(2)(in_features)
            else:
                normalizer = constructor()(in_features)
            normalizer.forward(x)

    def test_normalizations_with_resnet(self):
        params = Params({"type": "resnet50", "normalization": {"type": "identity"}})
        encoder = Encoder.from_params(params)
        assert isinstance(encoder.layer4[0].bn1, Identity)
        x = torch.zeros((2, 3, 4, 4)).normal_()
        encoder.forward(x)

        params = Params(
            {"type": "resnet50", "normalization": {"type": "batch_norm_2d"}}
        )
        encoder = Encoder.from_params(params)
        assert isinstance(encoder.layer4[0].bn1, BatchNorm2d)
        encoder.forward(x)

        params = Params(
            {"type": "resnet50", "normalization": {"type": "batch_renorm_2d"}}
        )
        encoder = Encoder.from_params(params)
        assert isinstance(encoder.layer4[0].bn1, BatchRenorm2d)
        encoder.forward(x)

        params = Params({"type": "resnet50", "normalization": {"type": "group_norm"}})
        with pytest.raises(ConfigurationError):
            Encoder.from_params(params)

        params = Params(
            {
                "type": "resnet50",
                "normalization": {"type": "group_norm", "num_groups": 8},
            }
        )
        encoder = Encoder.from_params(params)
        assert isinstance(encoder.layer4[0].bn1, GroupNorm)
        encoder.forward(x)
