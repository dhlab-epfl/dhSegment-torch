import pytest
import torch

from dh_segment_torch.config.params import Params
from dh_segment_torch.nn.activations import Activation
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class ActivationTest(DhSegmentTestCase):
    def test_all_activation_building(self):
        all_activations = Activation.get_available()
        in_features = 2
        x = torch.zeros((2, in_features, 4, 4)).normal_()
        for activation_name in all_activations:
            kwargs = {}
            if activation_name == 'threshold':
                kwargs = {'threshold': 2, 'value': 5}
            activation = Activation.get_constructor(activation_name)(**kwargs)

            activation(x)

    def test_none_default_args(self):
        params = {
            'type': 'hardtanh',
            'min_val': 'a'
        }

        with pytest.raises(TypeError):
            Activation.from_params(Params(params))

