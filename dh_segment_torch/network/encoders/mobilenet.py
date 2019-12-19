from typing import List, Dict

from torch.hub import load_state_dict_from_url
from torchvision.models.mobilenet import MobileNetV2
import torchvision.transforms.functional as F


class MobileNetV2Encoder(MobileNetV2):
    def __init__(self, output_dims: List[int], pretrained_settings: Dict[str, object], blocks: int = 4,
                 pretrained: bool = True, progress: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.blocks = blocks
        self.output_dims = output_dims  # Maybe refactor in upper class
        self.pretrained_settings = pretrained_settings
        self.pretrained = pretrained
        if pretrained:
            state_dict = load_state_dict_from_url(pretrained_settings['url'], progress=progress)
            self.load_state_dict(state_dict)

    def forward(self, x):
        if self.pretrained:
            x = F.normalize(x, self.pretrained_settings['mean'], self.pretrained_settings['std'])

        layers = [
            self.features[:2],
            self.features[2:4],
            self.features[4:7],
            self.features[7:14],
            self.features[14:]
        ]

        features_maps = [x]

        for layer in layers[:self.blocks + 1]:
            x = layer(x)
            features_maps.append(x)
        return features_maps


class MobileNetV2PaperEncoder(MobileNetV2Encoder):
    def __init__(self, blocks: int = 4, pretrained: bool = True, progress: bool = False):
        super().__init__(output_dims=[3, 16, 24, 32, 96, 1280][:blocks+2],
                         pretrained_settings=pretrained_settings_mobilenet_v2,
                         blocks=blocks,
                         pretrained=pretrained,
                         progress=progress)


pretrained_settings_mobilenet_v2 = {
    'url': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'input_space': 'RGB',
    'input_size': [3, 224, 224],
    'input_range': [0, 1],
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'num_classes': 1000
}