from typing import TypeVar, Generic, Callable, Optional

T = TypeVar("T")


class Lazy(Generic[T]):
    def __init__(self, constructor: Callable[..., T]):
        self._constructor = constructor

    def construct(self, **kwargs) -> Optional[T]:
        return self._constructor(**kwargs)
