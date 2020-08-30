from typing import List, Optional, Union, Callable, Tuple

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.param_group import ParamGroup, make_params_groups, check_param_groups


class Initializer(Registrable):
    def __init__(self, regexes: Union[List[str], str], initializer: Callable, **kwargs):
        self.regexes = [regexes] if isinstance(regexes, str) else regexes
        self.initializer = lambda param: initializer(param, **kwargs)

    def apply(self, params: List[torch.nn.Parameter]):
        for param in params:
            self.initializer(param)


@Initializer.register("uniform")
class UniformInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], a: float = 0.0, b: float = 1.0):
        super().__init__(regexes, torch.nn.init.uniform_, a=a, b=b)


@Initializer.register("normal")
class NormalInitializer(Initializer):
    def __init__(
        self, regexes: Union[List[str], str], mean: float = 0.0, std: float = 1.0
    ):
        super().__init__(regexes, torch.nn.init.normal_, mean=mean, std=std)


@Initializer.register("constant")
class ConstantInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], val: float = 0.0):
        super().__init__(regexes, torch.nn.init.constant_, val=val)


@Initializer.register("ones")
class OnesInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str]):
        super().__init__(regexes, torch.nn.init.ones_)


@Initializer.register("zeros")
class ZerosInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str]):
        super().__init__(regexes, torch.nn.init.zeros_)


@Initializer.register("eye")
class EyeInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str]):
        super().__init__(regexes, torch.nn.init.eye_)


@Initializer.register("dirac")
class DiracInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], groups: int = 1):
        super().__init__(regexes, torch.nn.init.dirac_, groups=groups)


@Initializer.register("xavier_uniform")
class XavierUniformInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], gain: float = 1.0):
        super().__init__(regexes, torch.nn.init.xavier_uniform_, gain=gain)


@Initializer.register("xavier_normal")
class XavierNormalInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], gain: float = 1.0):
        super().__init__(regexes, torch.nn.init.xavier_normal_, gain=gain)


@Initializer.register("kaiming_uniform")
class KaimingUniformInitializer(Initializer):
    def __init__(
        self,
        regexes: Union[List[str], str],
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
    ):
        super().__init__(
            regexes,
            torch.nn.init.kaiming_uniform_,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
        )


@Initializer.register("kaiming_normal")
class KaimingNormalInitializer(Initializer):
    def __init__(
        self,
        regexes: Union[List[str], str],
        a: float = 0.0,
        mode: str = "fan_in",
        nonlinearity: str = "leaky_relu",
    ):
        super().__init__(
            regexes,
            torch.nn.init.kaiming_normal_,
            a=a,
            mode=mode,
            nonlinearity=nonlinearity,
        )


@Initializer.register("orthogonal")
class OrthogonalInitializer(Initializer):
    def __init__(self, regexes: Union[List[str], str], gain: float = 1.0):
        super().__init__(regexes, torch.nn.init.orthogonal_, gain=gain)


@Initializer.register("sparse")
class SparseInitializer(Initializer):
    def __init__(
        self, regexes: Union[List[str], str], sparsity: float, std: float = 0.01
    ):
        super().__init__(regexes, torch.nn.init.sparse_, sparsity=sparsity, std=std)


class InitializerApplier(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        initializers: List[Initializer],
        exclude_regexes: Optional[List[str]] = None,
    ):
        self.initializers = [
            ParamGroup(params={"init": initializer.apply}, regexes=initializer.regexes) for initializer in initializers
        ]
        self.exclude_regexes = exclude_regexes

    def apply(self, parameters: List[Tuple[str, torch.nn.Parameter]]):
        param_groups = make_params_groups(
            parameters, self.initializers, self.exclude_regexes
        )
        if isinstance(param_groups[0], torch.nn.Parameter):
            param_groups = [{"params": param_groups}]
        param_groups = check_param_groups(param_groups)

        for param_group in param_groups:
            params = param_group["params"]
            initializer = param_group.get("init", None)
            if initializer:
                assert isinstance(initializer, Callable)
                initializer(params)


InitializerApplier.register("default")(InitializerApplier)
