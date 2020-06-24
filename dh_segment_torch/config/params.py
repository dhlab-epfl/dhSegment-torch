import copy
import json
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Dict, Any, Union

from dh_segment_torch.config.errors import ConfigurationError

try:
    from _jsonnet import evaluate_file, evaluate_snippet
except ImportError:

    def evaluate_file(filename: str, **kwargs):
        logger.warning(
            f"error loading _jsonnet (not supported on Windows). Loading the file as normal json"
        )
        with open(filename, "r") as infile:
            return infile.read()

    def evaluate_snippet(filename: str, expr: str, **kwargs):
        logger.warning(
            f"error loading _jsonnet (not supported on Windows). Loading the file as normal json"
        )
        return expr


logger = logging.getLogger(__name__)


class Params(MutableMapping):
    SENTINEL = object()

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params

    def pop(self, key: str, default: Any = SENTINEL) -> Any:
        if default is self.SENTINEL:
            try:
                value = self.params.pop(key)
            except KeyError:
                raise ConfigurationError(f"Could not find required key: {key}")
        else:
            value = self.params.pop(key, default)
        return _force_value_to_params(value)

    def get(self, key: str, default: Any = SENTINEL):
        if default is self.SENTINEL:
            try:
                value = self.params.get(key)
            except KeyError:
                raise ConfigurationError(f"Could not find required key: {key}")
        else:
            value = self.params.get(key, default)
        return _force_value_to_params(value)

    def as_dict(self):
        return self.params

    def copy(self) -> "Params":
        return copy.deepcopy(self)

    def assert_empty(self, class_name: str) -> None:
        if self.params:
            raise ConfigurationError(
                f"Could not exhaust all parameters for {class_name}, still got {self.params}."
            )

    def __getitem__(self, item):
        if item in self.params:
            return _force_value_to_params(self.params[item])
        else:
            raise KeyError

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]):
        params = json.loads(evaluate_file(str(file_path)))
        return cls(params)

    def to_file(self, file_path: Union[str, Path]):
        with open(str(file_path), "w") as outfile:
            json.dump(self.as_dict(), outfile, indent=4)

    def __repr__(self):
        return self.params.__repr__()

    def __str__(self):
        return self.params.__str__()


def _force_value_to_params(value: Any):
    if isinstance(value, dict):
        value = Params(value)
    elif isinstance(value, list) or isinstance(value, set):
        value = [_force_value_to_params(v) for v in value]
    return value
