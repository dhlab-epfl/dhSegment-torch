import torch

from dh_segment_torch.config import Registrable


class Normalization(Registrable):
    """
    Wrapper around pytorch's normalization function to be able to be instantiated through from_params
    """

    def __init__(self, torch_normalizer: torch.nn.Module, **kwargs):
        super().__init__()
        self._torch_normalizer = torch_normalizer
        self._kwargs = kwargs

    def __call__(self, num_features: int, **kwargs):
        kwargs.update(self._kwargs)  # We use in priority args from init
        return self._torch_normalizer(num_features, **kwargs)

    def __repr__(self):
        return f"Normalizer {self._torch_normalizer} with params: {self._kwargs}"