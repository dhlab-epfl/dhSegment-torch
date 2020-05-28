import collections
import inspect
import logging
from copy import deepcopy
from pathlib import Path
from typing import (
    Type,
    TypeVar,
    cast,
    Callable,
    Dict,
    Union,
    Any,
    Mapping,
    Tuple,
    Set,
    Iterable,
    List,
)

from dh_segment_torch.config import ConfigurationError, RegistrableError
from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.params import Params

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="FromParams")


class FromParams:
    """
    Mixin class to give a from_params method.
    Inspired by AllenNLP library
    """

    @classmethod
    def from_params(cls: Type[T], params: Params, **extras):
        from dh_segment_torch.config.registrable import Registrable

        params = normalize_params(params)

        if is_base_registrable(cls):
            if Registrable._register.get(cls) is None:
                raise ConfigurationError("Tried to construct an abstract registrable class that has nothing registered.")
            class_as_registrable = cast(Type[Registrable], cls)
            choices = class_as_registrable.get_available()

            subclass_type = params.pop("type", choices[0])

            subclass, constructor = class_as_registrable.get(subclass_type)
            if has_from_params(subclass):
                kwargs = create_kwargs(constructor, subclass, params, **extras)
            else:
                extras = get_extras(subclass, extras)
                kwargs = {**params, **extras}
            return class_as_registrable.get_constructor(subclass_type)(**kwargs)
        else:
            if cls.__init__ == object.__init__:
                kwargs: Dict[str, Any] = {}
                params.assert_empty(cls.__name__)
            else:
                kwargs = create_kwargs(cls.__init__, cls, params, **extras)
            return cls(**kwargs)


def create_kwargs(
    constructor: Callable[..., T], cls: Type[T], params: Params, **extras
) -> Dict[str, Any]:
    kwargs: Dict[str, any] = {}

    parameters = infer_params(constructor, cls)

    for param_name, param in parameters.items():
        if param_name == "self" or param.kind == param.VAR_KEYWORD:
            continue

        constructed_param = pop_construct_param(param_name, param, params, **extras)

        if constructed_param is not param.default:
            kwargs[param_name] = constructed_param
    params.assert_empty(cls.__name__)
    return kwargs


def pop_construct_param(param_name: str, param: inspect.Parameter, params: Params, **extras):
    if param_name in extras:
        if param_name not in params:
            return extras[param_name]
        else:
            logger.warning(
                f"Parameter {param_name} was found in extras, which is not required,"
                "but was also found in params. Using params value, but this can"
                "lead to unexpected results"
            )
    optional = param.default != param.empty
    popped_params = (
        params.pop(param_name, param.default)
        if optional
        else params.pop(param_name)
    )

    if popped_params is None:
        param_type = infer_type(param)
        origin, _ = get_origin_args(param_type)
        if origin == Lazy:
            return Lazy(lambda **kwargs: None)
        return None

    return construct_param(param_name, param, popped_params, **extras)


def construct_param(
    param_name: str, param: inspect.Parameter, params: Params, **extras
):
    """

    :param param_name:
    :param param:
    :param params:
    :param extras:
    :return:
    """

    param_type = infer_type(param)

    origin, args = get_origin_args(param.annotation)

    if is_optional(param) and params is None:
        return params

    if has_from_params(param_type):
        if params is param.default:
            return param.default
        elif params is not None:
            params = normalize_params(params)
            param_type_as_from_params = cast(Type[FromParams], param_type)
            sub_extras = get_extras(param_type_as_from_params, extras)
            return param_type_as_from_params.from_params(params, **sub_extras)
    elif origin == Lazy:
        if params is param.default:
            return Lazy(lambda **kwargs: param.default)
        sub_param_type_as_from_params = cast(Type[FromParams], args[0])
        sub_extras = get_extras(sub_param_type_as_from_params, extras)
        return Lazy(
            lambda **kwargs: sub_param_type_as_from_params.from_params(
                params=deepcopy(params), **{**sub_extras, **kwargs}
            )
        )

    elif param_type in {int, bool}:
        if type(params) in {int, bool}:
            return param_type(params)
        else:
            raise TypeError(f"Expected {param_name} to be a {param_type.__name__}.")
    elif param_type == float:
        if type(params) in {int, float}:
            return params
        else:
            raise TypeError(f"Expected {param_name} to be either a float or a int.")
    elif param_type == str:
        if type(params) == str or isinstance(params, Path):
            return str(params)
        else:
            raise TypeError(f"Expected {params} to be a string.")
    elif (
        origin in {collections.abc.Mapping, Mapping, Dict, dict}
        and len(args) == 2
        and can_construct(args[-1])
    ):
        if not isinstance(params, collections.abc.Mapping):
            raise TypeError(f"Expected {param_name} to be a mapping.")
        value_class = args[-1]
        value_class_as_param = inspect.Parameter(
            "dummy", kind=inspect.Parameter.VAR_KEYWORD, annotation=value_class
        )

        new_dict = {}
        for key_param, value_params in params.items():
            new_dict[key_param] = construct_param(
                f"{param_name}.{key_param}",
                value_class_as_param,
                value_params,
                **extras,
            )
        return new_dict
    elif origin in {Tuple, tuple} and all(can_construct(arg) for arg in args):
        if not isinstance(params, collections.abc.Sequence):
            raise TypeError(f"Expected {param_name} to be a sequence.")
        new_tuple = []
        for i, (value_class, value_params) in enumerate(zip(args, params)):
            value_class_as_param = inspect.Parameter(
                "dummy", kind=inspect.Parameter.VAR_KEYWORD, annotation=value_class
            )
            new_tuple.append(
                construct_param(
                    f"{param_name}.{i}", value_class_as_param, value_params, **extras
                )
            )
        return tuple(new_tuple)
    elif origin in {Set, set} and len(args) == 1 and can_construct(args[0]):
        if not isinstance(params, collections.abc.Set) and not isinstance(
            params, collections.abc.Sequence
        ):
            raise TypeError(f"Expected {param_name} to be a set or a sequence.")
        value_class = args[0]
        value_class_as_param = inspect.Parameter(
            "dummy", kind=inspect.Parameter.VAR_KEYWORD, annotation=value_class
        )

        new_set = set()

        for i, value_params in enumerate(params):
            new_set.add(
                construct_param(
                    f"{param_name}.{i}", value_class_as_param, value_params, **extras
                )
            )
        return new_set
    elif (
        origin in {collections.abc.Iterable, Iterable, List, list}
        and len(args) == 1
        and can_construct(args[0])
    ):
        if not isinstance(params, collections.abc.Sequence):
            raise TypeError(f"Expected {param_name} to be a sequence.")
        value_class = args[0]
        value_class_as_param = inspect.Parameter(
            "dummy", kind=inspect.Parameter.VAR_KEYWORD, annotation=value_class
        )

        new_list = []

        for i, value_params in enumerate(params):
            new_list.append(
                construct_param(
                    f"{param_name}.{i}", value_class_as_param, value_params, **extras
                )
            )
        return new_list
    elif origin == Union:
        backup_params = deepcopy(params)

        for value_class in args:
            value_class_as_param = inspect.Parameter(
                "dummy", kind=inspect.Parameter.VAR_KEYWORD, annotation=value_class
            )
            try:
                return construct_param(
                    param_name, value_class_as_param, params, **extras
                )
            except (
                AttributeError,
                ValueError,
                TypeError,
                ConfigurationError,
                RegistrableError,
            ):
                params = deepcopy(backup_params)
        raise ConfigurationError(
            f"Failed to construct {param_name} wit type {param_type}"
        )
    else:
        logger.warning(
            f"The params {str(params)} were not matched, returning them as is."
        )
        if isinstance(params, Params):
            return params.as_dict()
        else:
            return params


def can_construct(type_: Type) -> bool:
    if type_ in {int, bool, str, float}:
        return True
    elif has_from_params(type_):
        return True
    else:
        origin, args = get_origin_args(type_)
        if origin == Lazy:
            return True
        return all(can_construct(arg) for arg in args)


def infer_params(constructor: Callable[..., T], cls: Type[T]):
    signature = inspect.signature(constructor)
    parameters = dict(signature.parameters)

    has_kwargs = False
    for param in parameters.values():
        if param.kind == param.VAR_KEYWORD:
            has_kwargs = True

    if not has_kwargs:
        return parameters

    super_class = None
    for super_class_candidate in inspect.getmro(cls)[1:]:
        if issubclass(super_class_candidate, FromParams):
            super_class = super_class_candidate
            break
    if not super_class:
        raise RuntimeError(
            "Found a kwargs parameter with no superclass to account for them."
        )
    super_parameters = infer_params(super_class.__init__, super_class)
    return {**super_parameters, **parameters}


def infer_type(param: inspect.Parameter) -> type:
    if param.annotation == param.empty:
        # If the param did not have a type annotation, try to see if it had a not none default value
        if param.default != param.empty and param.default is not None:
            return type(param.default)
        else:
            raise RegistrableError(
                "Got a registrable that had params either not type hinted either w/o defaults"
            )
    else:
        origin, args = get_origin_args(param.annotation)
        if is_optional(param):
            return args[0]
        else:
            return param.annotation


def is_optional(param: inspect.Parameter) -> bool:
    origin, args = get_origin_args(param.annotation)
    return origin == Union and len(args) == 2 and args[1] == type(None)


def get_extras(cls: Type[T], extras: Dict[str, Any]) -> Dict[str, Any]:
    class_from_params = getattr(cls, "from_params", cls)

    signature = inspect.signature(class_from_params)
    parameters = dict(signature.parameters)
    if any(param.kind == param.VAR_KEYWORD for param in parameters.values()):
        sub_extras = extras
    else:
        sub_extras = {k: v for k, v in extras.items() if k in parameters}
    return sub_extras


def is_base_registrable(cls) -> bool:
    from dh_segment_torch.config.registrable import Registrable

    if not issubclass(cls, Registrable):
        return False
    method_resolution_order = inspect.getmro(cls)[1:]
    for base_class in method_resolution_order:
        if issubclass(base_class, Registrable) and base_class is not Registrable:
            return False
    return True


def normalize_params(params: Union[Params, str, Dict]):
    if isinstance(params, str):
        params = Params({"type": params})

    if isinstance(params, dict):
        params = Params(params)

    return params


def get_origin_args(type_: type):
    origin = getattr(type_, "__origin__", None)
    args = getattr(type_, "__args__", [])
    return origin, args


def has_from_params(type_: Type) -> bool:
    return hasattr(type_, "from_params")
