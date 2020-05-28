from typing import Optional, Dict, Union, List

import torch
from torch.hub import load_state_dict_from_url

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.models.encoders.utils import normalize_batch

import logging

logger = logging.getLogger(__name__)


class Encoder(torch.nn.Module, Registrable):
    default_implementation = 'resnet50'
    def __init__(
        self,
        output_dims: List[int],
        pretrained_settings: Optional[
            Dict[str, Union[str, int, float, List[Union[int, float]]]]
        ] = None,
        pretrained: bool = False,
        progress: bool = False,
    ):
        Registrable.__init__(self)
        self.output_dims = output_dims
        self.pretrained = pretrained
        self.pretrained_settings = pretrained_settings
        if pretrained:
            if pretrained_settings is not None:
                state_dict = load_state_dict_from_url(
                    pretrained_settings["url"], progress=progress
                )
                incompatible_keys = self.load_state_dict(state_dict, strict=False)
                if len(incompatible_keys[0]) > 0 or len(incompatible_keys[1]) > 0:
                    logger.warning(f"There was {len(incompatible_keys[0])} missing keys"
                                   f" and {len(incompatible_keys[1])} unexpected keys"
                                   "  when loading pretrained model")
                    print(incompatible_keys)
            else:
                raise ValueError("Cannot be pretrained if pretrained_settings is None")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pretrained:
            x = normalize_batch(
                x, self.pretrained_settings["mean"], self.pretrained_settings["std"]
            )
        return x
