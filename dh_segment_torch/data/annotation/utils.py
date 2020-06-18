from collections import namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Any, TypeVar


@dataclass
class ImageSize:
    height: int
    width: int


@dataclass
class Color:
    r: int
    g: int
    b: int


Coordinates = Tuple[Union[int, float], Union[int, float]]


def convert_coord_to_image(coord: Coordinates, height: int, width: int) -> Tuple[int, int]:
    x, y = coord
    return int(round(x * width)), int(round(y * height))


def convert_coord_to_normalized(coord: Coordinates, height: int, width: int) -> Tuple[float, float]:
    x, y = coord
    return x / float(width), y / float(height)


T = TypeVar("T")
U = TypeVar("U")


def reverse_dict(dico: Dict[T, U]) -> Dict[U, T]:
    return {v: k for k, v in dico.items()}
