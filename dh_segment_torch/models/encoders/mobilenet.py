from typing import List, Dict, Optional, Union

from frozendict import frozendict
from torchvision.models.mobilenet import MobileNetV2

from dh_segment_torch.models.encoders.encoder import Encoder

pretrained_settings_mobilenet_v2 = frozendict(
    {
        "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
        "input_space": "RGB",
        "input_size": [3, 224, 224],
        "input_range": [0, 1],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "num_classes": 1000,
    }
)


class MobileNetV2Encoder(Encoder, MobileNetV2):
    def __init__(
        self,
        output_dims: List[int],
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = None,
        pretrained: bool = False,
        progress: bool = False,
        # width_mult: float = 1.0,
        # inverted_residual_setting: Optional[List[List[int]]] = None,
        # round_nearest: int = 8,
        # block: Optional[nn.Module] = None
    ):
        MobileNetV2.__init__(self)
        Encoder.__init__(self, output_dims, pretrained_settings, pretrained, progress)
        # MobileNetV2.__init__(width_mult=width_mult, inverted_residual_setting=inverted_residual_setting,
        # round_nearest=round_nearest, block=block)
        self.blocks = blocks

    def forward(self, x):
        x = super().normalize_if_pretrained(x)

        layers = [
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:],
        ]

        features_maps = [x]

        for layer in layers[: self.blocks + 1]:
            x = layer(x)
            features_maps.append(x)
        return features_maps

    @classmethod
    def paper_encoder(
        cls,
        blocks: int = 4,
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = pretrained_settings_mobilenet_v2,
        pretrained: bool = True,
        progress: bool = False,
    ):
        return cls(
            [3, 16, 24, 32, 96, 1280][: blocks + 2],
            blocks,
            pretrained_settings,
            pretrained,
            progress,
        )


Encoder.register("mobilenetv2", "paper_encoder")(MobileNetV2Encoder)
