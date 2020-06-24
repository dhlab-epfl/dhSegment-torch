from typing import Union, Dict, List, Any, Tuple, Optional

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.nn.param_group import ParamGroup, make_params_groups, normalize_param_groups, \
    check_param_groups


class Regularizer(Registrable):
    def __init__(
        self,
        params: Union[List[torch.nn.Parameter], List[Dict[str, Any]]],
        defaults: Dict[str, Any],
        param_groups_names: Optional[List[str]] = None,
    ):
        self.defaults = defaults

        param_groups = params
        if len(param_groups) == 0:
            raise ValueError("Cannot regularized an empty params list")
        if isinstance(param_groups[0], torch.nn.Parameter):
            param_groups = [{"params": param_groups}]

        self.param_groups = check_param_groups(param_groups, defaults)
        if param_groups_names:
            assert len(param_groups_names)+1 == len(params)
            self.param_groups_names = param_groups_names

    def get_penalty(self):
        raise NotImplementedError



@Regularizer.register("l1")
class L1Regularizer(Regularizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.nn.Parameter]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        alpha: float = 0.01,
    ):
        param_groups = normalize_param_groups(param_groups)
        super().__init__(
            make_params_groups(model_params, param_groups),
            {"alpha": alpha},
            list(param_groups.keys()),
        )

    def get_penalty(self) -> torch.Tensor:
        penalty = None
        for param_group in self.param_groups:
            params = param_group["params"]
            alpha = param_group["alpha"]
            for param in params:
                if penalty is None:
                    penalty = alpha * torch.sum(torch.abs(param))
                else:
                    penalty += alpha * torch.sum(torch.abs(param))
        return penalty


@Regularizer.register("l2")
class L2Regularizer(Regularizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.nn.Parameter]],
        param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
        alpha: float = 0.01,
    ):
        param_groups = normalize_param_groups(param_groups)
        super().__init__(
            make_params_groups(model_params, param_groups),
            {"alpha": alpha},
            list(param_groups.keys()),
        )

    def get_penalty(self) -> torch.Tensor:
        penalty = None
        for param_group in self.param_groups:
            params = param_group["params"]
            alpha = param_group["alpha"]
            for param in params:
                if penalty is None:
                    penalty = alpha * torch.sum(torch.pow(param, 2))
                else:
                    penalty += alpha * torch.sum(torch.pow(param, 2))
        return penalty
