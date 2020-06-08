from dh_segment_torch.config.params import Params
from dh_segment_torch.models.model import Model
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase
from dh_segment_torch.training.regularizers import Regularizer


class RegularizerTest(DhSegmentTestCase):
    def test_build_regularizers(self):
        model = Model.from_params(
            Params(
                {
                    "encoder": {},
                    "decoder": {"decoder_channels": [512, 256, 128, 64, 32],},
                    "num_classes": 4,
                }
            )
        )

        named_parameters = [x for x in model.named_parameters()]

        regularizers = Regularizer.get_available()

        for regularizer in regularizers:
            alpha = 10
            params = {"type": regularizer, "alpha": alpha}
            reg = Regularizer.from_params(Params(params), model_params=named_parameters)
            assert reg.param_groups[0]["alpha"] == alpha

    def test_make_param_groups(self):
        model = Model.from_params(
            Params(
                {
                    "encoder": {"type": "resnet50"},
                    "decoder": {
                        "type": "unet",
                        "decoder_channels": [512, 256, 128, 64, 32],
                    },
                    "num_classes": 4,
                }
            )
        )

        named_parameters = [x for x in model.named_parameters()]
        alpha_decoder = 0.01
        alpha_logits = 0.1
        alpha_encoder = 0.0001

        params = {
            "param_groups": [
                ["encoder", {"alpha": alpha_encoder}],
                [
                    ["decoder.logits.0.weight", "decoder.logits.0.bias"],
                    {"alpha": alpha_logits},
                ],
            ],
            "alpha": alpha_decoder,
        }

        reg = Regularizer.from_params(Params(params), model_params=named_parameters)

        assert reg.param_groups[0]["alpha"] == alpha_encoder
        assert len(reg.param_groups[0]["params"]) == 161
        assert reg.param_groups[2]["alpha"] == alpha_decoder
        assert len(reg.param_groups[2]["params"]) == 20
        assert reg.param_groups[1]["alpha"] == alpha_logits
        assert len(reg.param_groups[1]["params"]) == 2
