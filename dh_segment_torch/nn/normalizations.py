import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.normalization import batch_norm_drop, batch_renorm


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


@Normalization.register("identity")
class IdentityNormNormalization(Normalization):
    def __init__(self):
        super().__init__(torch.nn.Identity)


@Normalization.register("batch_norm_2d")
class BatchNorm2dNormalization(Normalization):
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(
            torch.nn.BatchNorm2d,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@Normalization.register("batch_norm_2d_drop")
class BatchNorm2dDropNormalization(Normalization):
    def __init__(
        self,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__(
            batch_norm_drop.BatchNorm2dDrop,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )


@Normalization.register("batch_renorm_2d")
class BatchRenorm2dNormalization(Normalization):
    def __init__(
        self, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True,
    ):
        super().__init__(
            batch_renorm.BatchRenorm2d, eps=eps, momentum=momentum, affine=affine,
        )


@Normalization.register("batch_renorm_2d_drop")
class BatchRenorm2dDropNormalization(Normalization):
    def __init__(
        self, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True,
    ):
        super().__init__(
            batch_norm_drop.BatchRenorm2dDrop,
            eps=eps,
            momentum=momentum,
            affine=affine,
        )


@Normalization.register("group_norm")
class GroupNormNormalization(Normalization):
    def __init__(self, num_groups: int, eps: float = 1e-05, affine: bool = True):
        super().__init__(
            torch.nn.GroupNorm, num_groups=num_groups, eps=eps, affine=affine
        )

    def __call__(self, num_features: int, **kwargs):
        kwargs.update(self._kwargs)
        return self._torch_normalizer(num_channels=num_features, **kwargs)
