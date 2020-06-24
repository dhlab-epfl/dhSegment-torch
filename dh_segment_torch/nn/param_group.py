import re
from dataclasses import dataclass
from typing import List, Dict, Any, Union, Optional, Tuple

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.utils.ops import normalize_dict


@dataclass
class ParamGroup(Registrable):
    default_implementation = "param_group"
    params: Dict[str, Any]
    regexes: Optional[Union[List[str], str]] = None


ParamGroup.register("param_group")(ParamGroup)


def normalize_param_groups(
    param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]]
) -> Dict[str, ParamGroup]:
    if param_groups is None:
        return {}
    is_dict = isinstance(param_groups, Dict)
    param_groups = normalize_dict(param_groups)
    for param_group_name, param_group in param_groups.items():
        if param_group.regexes is None:
            if not is_dict:
                raise ValueError("Cannot infer regex from dict name")
            param_group.regexes = [param_group_name]
    return param_groups


def make_params_groups(
    model_params: List[Tuple[str, torch.nn.Parameter]],
    param_groups: Optional[Union[Dict[str, ParamGroup], List[ParamGroup]]] = None,
    exclude_regexes: Optional[List[str]] = None,
) -> Union[List[torch.nn.Parameter], List[Dict[str, Any]]]:
    if param_groups is None:
        new_params_groups: List[torch.nn.Parameter] = [
            param for _, param in model_params
        ]
    else:
        exclude_regexes = (
            "|".join([f"({regex})" for regex in exclude_regexes])
            if exclude_regexes
            else None
        )
        if isinstance(param_groups, Dict):
            param_groups = list(param_groups.values())
        new_params_groups: List[Dict[str, Any]] = [
            {"params": list()} for _ in range(len(param_groups) + 1)
        ]
        for param_name, param in model_params:
            matched_group_index = -1
            matched_kwargs = {}
            for group_index, param_group in enumerate(param_groups):
                if exclude_regexes and re.search(exclude_regexes, param_name):
                    break
                if isinstance(param_group.regexes, str):
                    regexes = [param_group.regexes]
                else:
                    regexes = param_group.regexes
                for regex in regexes:
                    if re.search(regex, param_name):
                        if (
                            matched_group_index != -1
                            and matched_group_index != group_index
                        ):
                            raise ValueError(
                                f"Parameter {param_name} was matched in two groups"
                            )
                        matched_group_index = group_index
                        matched_kwargs = param_group.params
            new_params_groups[matched_group_index]["params"].append(param)
            new_params_groups[matched_group_index].update(**matched_kwargs)
    for group_index in range(len(param_groups)):
        if len(new_params_groups[group_index]["params"]) == 0:
            raise ValueError(f"Group at index {group_index} was zero.")
    return new_params_groups


def check_param_groups(
    param_groups: Optional[List[Dict[str, Any]]], defaults: Dict[str, Any] = None
) -> Optional[List[Dict[str, Any]]]:
    if param_groups is None:
        return param_groups
    all_params = []
    defaults = defaults if defaults else {}
    for param_group in param_groups:
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
                raise TypeError("Can only affect tensors")
            if not param.is_leaf:
                raise ValueError("Can't affect a non-leaf Tensor")

        for name, default in defaults.items():
            param_group.setdefault(name, default)

        param_set = set()
        for group in all_params:
            param_set.update(set(group["params"]))

        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("Some parameters appear in more than one parameter group")

        all_params.append(param_group)
    return all_params
