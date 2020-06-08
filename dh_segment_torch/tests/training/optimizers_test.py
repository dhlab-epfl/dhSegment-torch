from dh_segment_torch.config.params import Params
from dh_segment_torch.models.model import Model
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase
from dh_segment_torch.training.optimizers import Optimizer


class OptimizerTest(DhSegmentTestCase):
    def test_build_optimizers(self):
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

        optimizers = Optimizer.get_available()

        for optimizer in optimizers:
            lr = 10
            params = {"type": optimizer, "lr": lr}
            opt = Optimizer.from_params(Params(params), model_params=named_parameters)
            assert opt.state_dict()["param_groups"][0]["lr"] == lr

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
        lr_decoder = 0.01
        lr_logits = 0.1
        lr_encoder = 0.0001

        params = {
            "param_groups": [
                ["encoder", {"lr": lr_encoder}],
                [
                    ["decoder.logits.0.weight", "decoder.logits.0.bias"],
                    {"lr": lr_logits},
                ],
            ],
            "lr": lr_decoder,
        }

        opt = Optimizer.from_params(Params(params), model_params=named_parameters)

        assert opt.state_dict()["param_groups"][0]["lr"] == lr_encoder
        assert len(opt.state_dict()["param_groups"][0]["params"]) == 161
        assert opt.state_dict()["param_groups"][2]["lr"] == lr_decoder
        assert len(opt.state_dict()["param_groups"][2]["params"]) == 20
        assert opt.state_dict()["param_groups"][1]["lr"] == lr_logits
        assert len(opt.state_dict()["param_groups"][1]["params"]) == 2
