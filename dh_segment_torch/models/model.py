from typing import Dict

import torch
import torch.nn as nn

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.models.decoders.decoder import Decoder
from dh_segment_torch.models.encoders.encoder import Encoder
from dh_segment_torch.nn.losses import Loss, BCEWithLogitsLoss


class Model(Registrable, nn.Module):
    default_implementation = "segmentation_model"

    def forward(self, *inputs) -> Dict[str, torch.Tensor]:
        raise NotImplemented("Subclasses should implement this fonction")


@Model.register("segmentation_model", "from_partial")
class SegmentationModel(Model):
    def __init__(self, encoder: Encoder, decoder: Decoder, loss: Loss = BCEWithLogitsLoss()):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.loss = loss

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        res = {}
        features_maps = self.encoder(input)
        logits = self.decoder(*features_maps)
        res['logits'] = logits
        loss = self.loss(logits, target)
        res['loss'] = loss
        return res

    @classmethod
    def from_partial(cls, encoder: Encoder, decoder: Lazy[Decoder], loss: Loss = BCEWithLogitsLoss()):
        decoder = decoder.construct(encoder_channels=encoder.output_dims)
        return cls(encoder, decoder, loss)
