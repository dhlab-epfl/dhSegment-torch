from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Union, Dict, List, Set, Tuple, Any

import pytest

from dh_segment_torch.config import ConfigurationError
from dh_segment_torch.config.from_params import FromParams
from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.params import Params
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class RegistrableNoFromParams(Registrable):
    pass


class CustomHashable:
    def __init__(self, x: int):
        self.x = x

    def __hash__(self):
        return hash(self.x)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.x == other.x


Registrable._register[RegistrableNoFromParams] = {"custom_hash": (CustomHashable, None)}


class Transform(Registrable, ABC):
    default_implementation = "mult_by_2_add_5"

    @abstractmethod
    def apply(self, input):
        return None


@Transform.register("mult_by_x")
class MultiplyByX(Transform):
    def __init__(self, x: Union[int, str]):
        self.x = x

    @classmethod
    def multiplyby2(cls):
        return cls(2)

    def apply(self, input):
        return input * self.x


Transform.register("mult_by_2", "multiplyby2")(MultiplyByX)


@Transform.register("mult_by_x_add_y")
class MultiplyByXAddY(MultiplyByX):
    def __init__(self, x: int, y: int):
        super().__init__(x)
        self.y = y

    @classmethod
    def multiplyby2add5(cls):
        return cls(2, 5)

    def apply(self, input):
        return super(MultiplyByXAddY, self).apply(input) + self.y


Transform.register("mult_by_2_add_5", "multiplyby2add5")(MultiplyByXAddY)


class Dataset(Registrable):
    default_implementation = "default"

    def __init__(
        self, data_path: Union[str, int], transforms: Dict[Union[str, int], Transform]
    ):
        self.data_path = data_path
        self.transforms = transforms

    def apply(self):
        res = 1
        for key, transform in self.transforms.items():
            res = transform.apply(res)
        return res

    @classmethod
    def from_single_transform(cls, data_path: Union[str, int], transform: Transform):
        return cls(data_path, {0: transform})

    @classmethod
    def from_transforms_list(
        cls, data_path: Union[str, int], transforms: List[Transform]
    ):
        return cls(data_path, dict(enumerate(transforms)))

    @classmethod
    def from_transforms_tuple(
        cls, data_path: Union[str, int], transforms: List[Transform]
    ):
        return cls(data_path, dict(enumerate(transforms)))

    @classmethod
    def from_set_param(
        cls, data_path: str, transform: Transform, set_arg: Set[RegistrableNoFromParams]
    ):
        class_constructed = cls(data_path, {0: transform})
        class_constructed.set_arg = set_arg
        return class_constructed


Dataset.register("default")(Dataset)
Dataset.register("from_single_transform", "from_single_transform")(Dataset)
Dataset.register("from_transforms_list", "from_transforms_list")(Dataset)
Dataset.register("from_transforms_tuple", "from_transforms_tuple")(Dataset)
Dataset.register("from_set_param", "from_set_param")(Dataset)


class FromParamsTest(DhSegmentTestCase):
    def test_basic_from_params(self):
        config_dict = {}
        transform = Transform.from_params(Params(config_dict))
        assert transform.apply(1) == 7

        config_dict["x"] = 4
        with pytest.raises(ConfigurationError):
            Transform.from_params(Params(config_dict))

        config_dict["type"] = "mult_by_x"
        transform = Transform.from_params(Params(config_dict))
        assert transform.apply(4) == 16

        config_dict["type"] = "mult_by_x_add_y"
        config_dict["x"] = 4
        with pytest.raises(ConfigurationError):
            Transform.from_params(Params(config_dict))
        config_dict["type"] = "mult_by_x_add_y"
        config_dict["x"] = 4
        config_dict["y"] = 2
        transform = Transform.from_params(Params(config_dict))
        assert transform.apply(4) == 18

        config_dict["type"] = "mult_by_x_add_y"
        config_dict["x"] = 4
        config_dict["y"] = "test"
        with pytest.raises(TypeError):
            Transform.from_params(Params(config_dict))

    def test_union(self):
        config_dict = {"type": "mult_by_x", "x": "y"}
        transform = Transform.from_params(Params(config_dict))
        assert transform.apply(4) == "yyyy"

    def test_simple_nesting(self):
        config_dict = {
            "type": "from_single_transform",
            "data_path": 10,
            "transform": {"type": "mult_by_x", "x": 4},
        }

        dataset = Dataset.from_params(Params(config_dict))
        assert dataset.data_path == 10
        assert len(dataset.transforms) == 1
        assert dataset.transforms[0].x == 4

    def test_complex_nesting(self):
        base_config_dict = {
            "data_path": "./",
            "transforms": {
                "t1": {"type": "mult_by_x", "x": 4},
                "t2": {"type": "mult_by_2_add_5"},
            },
        }

        config_dict = deepcopy(base_config_dict)
        dataset = Dataset.from_params(Params(config_dict))

        assert dataset.data_path == "./"
        assert len(dataset.transforms) == 2
        assert dataset.transforms["t2"].y == 5

        # We now test with a list of transforms

        # Default dataset expects to be  mapping
        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["transforms"] = list(config_dict["transforms"].values())
            Dataset.from_params(Params(config_dict))

        config_dict = deepcopy(base_config_dict)
        config_dict["transforms"] = list(config_dict["transforms"].values())
        config_dict["type"] = "from_transforms_list"
        dataset = Dataset.from_params(Params(config_dict))

        assert dataset.data_path == "./"
        assert len(dataset.transforms) == 2
        assert dataset.transforms[1].y == 5

        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["type"] = "from_transforms_list"
            Dataset.from_params(Params(config_dict))

        # With tuples
        # Default dataset expects to be  mapping
        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["transforms"] = tuple(config_dict["transforms"].values())
            Dataset.from_params(Params(config_dict))

        config_dict = deepcopy(base_config_dict)
        config_dict["transforms"] = tuple(config_dict["transforms"].values())
        config_dict["type"] = "from_transforms_tuple"
        dataset = Dataset.from_params(Params(config_dict))

        assert dataset.data_path == "./"
        assert len(dataset.transforms) == 2
        assert dataset.transforms[1].y == 5

        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["type"] = "from_transforms_tuple"
            Dataset.from_params(Params(config_dict))

        # With set
        # Default dataset expects to be  mapping
        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["transforms"] = set(config_dict["transforms"].values())
            Dataset.from_params(Params(config_dict))

        config_dict = deepcopy(base_config_dict)
        config_dict["type"] = "from_set_param"
        config_dict["transform"] = list(config_dict["transforms"].values())[0]
        del config_dict["transforms"]
        config_dict["set_arg"] = [{"x": 1}, {"x": 2}]
        dataset = Dataset.from_params(Params(config_dict))

        assert dataset.data_path == "./"
        assert len(dataset.transforms) == 1
        assert dataset.transforms[0].x == 4
        assert len(dataset.set_arg) == 2

        with pytest.raises(TypeError):
            config_dict = deepcopy(base_config_dict)
            config_dict["type"] = "from_set_param"
            config_dict["transform"] = list(config_dict["transforms"].values())[0]
            del config_dict["transforms"]
            config_dict["set_arg"] = {"k1": {"x": 1}, "k2": {"x": 2}}
            Dataset.from_params(Params(config_dict))

    def test_lazy(self):
        test_string = "this is a test"
        extra_string = "extra string"

        class ConstructedObject(FromParams):
            def __init__(self, string: str, extra: str):
                self.string = string
                self.extra = extra

        class Testing(FromParams):
            def __init__(self, lazy_object: Lazy[ConstructedObject]):
                first_time = lazy_object.construct(extra=extra_string)
                second_time = lazy_object.construct(extra=extra_string)
                assert first_time.string == test_string
                assert first_time.extra == extra_string
                assert second_time.string == test_string
                assert second_time.extra == extra_string

        Testing.from_params(Params({"lazy_object": {"string": test_string}}))

    def test_list_with_strings(self):
        params = {
            "type": "from_transforms_list",
            "data_path": "./",
            "transforms": [
                "mult_by_2_add_5",
                {"type": "mult_by_x", "x": 5},
                "mult_by_2",
            ],
        }

        dataset = Dataset.from_params(Params(params))
        assert len(dataset.transforms) == 3
        assert isinstance(dataset.transforms[0], MultiplyByXAddY)
        assert dataset.transforms[0].x == 2 and dataset.transforms[0].y == 5
        assert isinstance(dataset.transforms[1], MultiplyByX)
        assert dataset.transforms[1].x == 5
        assert isinstance(dataset.transforms[2], MultiplyByX)
        assert dataset.transforms[2].x == 2

    def test_union_str_list_str(self):
        class A(FromParams):
            def __init__(self, x: List[Tuple[Union[str, List[str]], Dict[str, Any]]]):
                self.x = x

        a = A.from_params(Params({"x": [["test", {}]]}))
        # print(a.x)
        # 1/0
