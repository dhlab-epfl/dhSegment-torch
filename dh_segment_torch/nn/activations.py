from typing import Callable

import torch

from dh_segment_torch.config.registrable import Registrable


class Activation(torch.nn.Module, Registrable):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _ActivationLambda(torch.nn.Module):
    def __init__(
        self, activation_func: Callable[[torch.Tensor], torch.Tensor], name: str
    ):
        super().__init__()
        self._name = name
        self._activation_func = activation_func

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._activation_func(x)

    def _get_name(self):
        return self._name


Registrable._register[Activation] = {
    "linear": (lambda: _ActivationLambda(lambda x: x, "Linear"), None),  # type: ignore
    "mish": (  # type: ignore
        lambda: _ActivationLambda(
            lambda x: x * torch.tanh(torch.nn.functional.softplus(x)), "Mish"
        ),
        None,
    ),
    "swish": (lambda: _ActivationLambda(lambda x: x * torch.sigmoid(x), "Swish"), None),  # type: ignore
    "relu": (torch.nn.ReLU, None),
    "relu6": (torch.nn.ReLU6, None),
    "elu": (torch.nn.ELU, None),
    "prelu": (torch.nn.PReLU, None),
    "leaky_relu": (torch.nn.LeakyReLU, None),
    "threshold": (torch.nn.Threshold, None),
    "hardtanh": (torch.nn.Hardtanh, None),
    "sigmoid": (torch.nn.Sigmoid, None),
    "tanh": (torch.nn.Tanh, None),
    "log_sigmoid": (torch.nn.LogSigmoid, None),
    "softplus": (torch.nn.Softplus, None),
    "softshrink": (torch.nn.Softshrink, None),
    "softsign": (torch.nn.Softsign, None),
    "tanhshrink": (torch.nn.Tanhshrink, None),
    "selu": (torch.nn.SELU, None),
}
