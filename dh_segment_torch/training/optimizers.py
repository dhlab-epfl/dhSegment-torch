from typing import List, Tuple, Optional, Union, Dict

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.param_group import (
    ParamGroup,
    make_params_groups,
    normalize_param_groups,
)


class Optimizer(torch.optim.Optimizer, Registrable):
    default_implementation = "adam"
    param_groups_names = list()

    def _init_and_get_param_groups(
        self, model_params, param_groups
    ) -> Union[List[torch.Tensor], Dict[str, any]]:
        param_groups = normalize_param_groups(param_groups)
        self.param_groups_names = list(param_groups.keys())
        return make_params_groups(model_params, param_groups)


@Optimizer.register("adam")
class AdamOptimizer(torch.optim.Adam, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
        amsgrad: bool = False,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
        )


@Optimizer.register("adadelta")
class AdadeltaOptimizer(torch.optim.Adadelta, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-06,
        weight_decay: float = 0,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            rho,
            eps,
            weight_decay,
        )


@Optimizer.register("adagrad")
class AdagradOptimizer(torch.optim.Adagrad, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            lr_decay,
            weight_decay,
            initial_accumulator_value,
            eps,
        )


@Optimizer.register("adamw")
class AdamWOptimizer(torch.optim.AdamW, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
        )


@Optimizer.register("sparse_adam")
class SparseAdamOptimizer(torch.optim.SparseAdam, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups), lr, betas, eps,
        )


@Optimizer.register("adamax")
class AdamaxOptimizer(torch.optim.Adamax, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.002,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            betas,
            eps,
            weight_decay,
        )


@Optimizer.register("asgd")
class ASGDOptimizer(torch.optim.ASGD, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.01,
        lambd: float = 0.0001,
        alpha: float = 0.75,
        t0: float = 1000000.0,
        weight_decay: float = 0,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            lambd,
            alpha,
            t0,
            weight_decay,
        )


@Optimizer.register("LBFGS")
class LBFGSOptimizer(torch.optim.LBFGS, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-07,
        tolerance_change: float = 1e-09,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            max_iter,
            max_eval,
            tolerance_grad,
            tolerance_change,
            history_size,
            line_search_fn,
        )

@Optimizer.register("rmsprop")
class RMSpropOptimizer(torch.optim.RMSprop, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            alpha,
            eps,
            weight_decay,
            momentum,
            centered,
        )


@Optimizer.register("rpop")
class RpropOptimizer(torch.optim.Rprop, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.01,
        etas: Tuple[float, float] = (0.5, 1.2),
        step_sizes: Tuple[float, float] = (1e-06, 50),
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups), lr, etas, step_sizes
        )


@Optimizer.register("sgd")
class SGDOptimizer(torch.optim.SGD, Optimizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.Tensor]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        lr: float = 0.001,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ):
        super().__init__(
            super()._init_and_get_param_groups(model_params, param_groups),
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        )
