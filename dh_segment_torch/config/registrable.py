import inspect
import logging
from collections import defaultdict
from typing import TypeVar, Dict, Type, Tuple, Optional, Callable, List

from dh_segment_torch.config import ConfigurationError, RegistrableError
from dh_segment_torch.config.from_params import FromParams

T = TypeVar("T", bound="Registrable")

logger = logging.getLogger(__name__)


class Registrable(FromParams):
    _register: Dict[Type, Dict[str, Tuple[Type[T], Callable[..., T]]]] = defaultdict(
        dict
    )
    default_implementation: str = None

    @classmethod
    def register(
        cls: Type[T],
        name: str,
        constructor: Optional[str] = None,
        exist_ok: bool = False,
    ):
        register = Registrable._register[cls]

        def add_to_register(subclass: Type[T]):
            if name in register:
                if exist_ok:
                    logger.warning(
                        f"{name} was already registered in {cls.__name__}, but allowing it."
                    )
                else:
                    raise RegistrableError(
                        f"{name} was already registered in {cls.__name__}."
                    )
            if constructor is None:
                constructor_method = subclass.__init__
            elif inspect.ismethod(getattr(subclass, constructor, None)):
                constructor_method = getattr(subclass, constructor)
            else:
                raise ConfigurationError(
                    f"The specified constructor {constructor} for {name} is not a method."
                )
            register[name] = (subclass, constructor_method)
            return subclass

        return add_to_register

    @classmethod
    def get_available(cls) -> List[str]:
        available = list(Registrable._register[cls].keys())
        default = cls.default_implementation
        if default is None:
            return available
        elif default not in available:
            raise RegistrableError(
                f"default implementation {default} is not registered."
            )
        else:
            return [default] + [i for i in available if i != default]

    @classmethod
    def get(cls, key: str) -> Tuple[Type[T], Callable[..., T]]:
        if key in Registrable._register[cls]:
            registered_class, constructor = Registrable._register[cls][key]
            # In some cases to make code more readable, some constructor are null,
            # we set it to __init__ when we have the opportunity
            if constructor is None:
                Registrable._register[cls][key] = (
                    registered_class,
                    registered_class.__init__,
                )
            return Registrable._register[cls][key]
        else:
            raise KeyError(f"Missing key {key} in class {cls.__name__}")

    @classmethod
    def get_type(cls, key: Type[T]):
        reversed_register = {
            type_class: type_str
            for (type_str, (type_class, _)) in Registrable._register[cls].items()
        }
        if key in reversed_register:
            return reversed_register[key]
        else:
            raise KeyError(f"Could not find type {key} for class {cls.__name__}")

    @classmethod
    def get_constructor(cls, key: str) -> Callable[..., T]:
        registered_class, constructor = cls.get(key)
        if constructor.__name__ == "__init__":
            constructor = registered_class
        return constructor
