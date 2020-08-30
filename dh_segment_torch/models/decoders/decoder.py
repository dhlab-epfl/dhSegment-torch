import logging
from typing import List

import torch

from dh_segment_torch.config.registrable import Registrable

logger = logging.getLogger(__name__)


class Decoder(torch.nn.Module, Registrable):
    default_implementation = "unet"

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int],
        num_classes: int,
    ):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        self.n_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
