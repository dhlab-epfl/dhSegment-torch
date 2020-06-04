from typing import Union, Dict, List, Any, Tuple

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.utils.ops import make_params_groups


class Regularizer(Registrable):
    def __init__(
        self,
        params: Union[List[torch.nn.Parameter], List[Dict[str, Any]]],
        defaults: Dict[str, Any],
    ):
        self.defaults = defaults
        self.param_groups = []

        param_groups = params
        if len(param_groups) == 0:
            raise ValueError("Cannot regularized an empty params list")
        if isinstance(param_groups[0], torch.nn.Parameter):
            param_groups = [{"params": param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)

    def get_penalty(self):
        raise NotImplementedError

    def add_param_group(self, param_group: Dict[str, Any]):
        assert isinstance(param_group, dict)

        params = param_group["params"]
        if isinstance(params, torch.Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError("Params should not be a set")
        else:
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, torch.Tensor):
                raise TypeError("Regularizer can only regularized tensors")

            if not param.is_leaf:
                raise ValueError("Can't optimize a non-leaf Tensor")

        for name, default in self.defaults.items():
            param_group.setdefault(name, default)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("Some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


@Regularizer.register("l1")
class L1Regularizer(Regularizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.nn.Parameter]],
        param_groups: List[Tuple[Union[str, List[str]], Dict[str, Any]]] = None,
        alpha: float = 0.01,
    ):
        super().__init__(
            make_params_groups(model_params, param_groups), {"alpha": alpha}
        )

    def get_penalty(self) -> torch.Tensor:
        penalty = 0.0
        for param_group in self.param_groups:
            params = param_group["params"]
            alpha = param_group["alpha"]
            for param in params:
                penalty = penalty + alpha * torch.sum(torch.abs(param))
        return penalty


@Regularizer.register("l2")
class L2Regularizer(Regularizer):
    def __init__(
        self,
        model_params: List[Tuple[str, torch.nn.Parameter]],
        param_groups: List[Tuple[Union[str, List[str]], Dict[str, Any]]] = None,
        alpha: float = 0.01,
    ):
        super().__init__(
            make_params_groups(model_params, param_groups), {"alpha": alpha}
        )

    def get_penalty(self) -> torch.Tensor:
        penalty = 0.0
        for param_group in self.param_groups:
            params = param_group["params"]
            alpha = param_group["alpha"]
            for param in params:
                penalty = penalty + alpha * torch.sum(torch.pow(param, 2))
        return penalty
