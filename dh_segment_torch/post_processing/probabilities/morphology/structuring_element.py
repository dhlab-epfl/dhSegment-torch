from typing import Tuple

import cv2
import numpy as np
from skimage import morphology

from dh_segment_torch.config import Registrable


class StructuringElement(Registrable):
    default_implementation = "rectangle"

    @property
    def element(self) -> np.array:
        raise NotImplementedError


class OpencvStructuringElement(StructuringElement):
    def __init__(
        self, shape: int, ksize: Tuple[int, int], anchor: Tuple[int, int] = (-1, -1)
    ):
        self.shape = shape
        self.ksize = ksize
        self.anchor = anchor

    @property
    def element(self) -> np.array:
        return cv2.getStructuringElement(self.shape, self.ksize, self.anchor)

    @classmethod
    def rectangle(cls, ksize: Tuple[int, int], anchor: Tuple[int, int] = (-1, -1)):
        cls(cv2.MORPH_RECT, ksize, anchor)

    @classmethod
    def cross(cls, ksize: Tuple[int, int], anchor: Tuple[int, int] = (-1, -1)):
        cls(cv2.MORPH_CROSS, ksize, anchor)

    @classmethod
    def ellipse(cls, ksize: Tuple[int, int], anchor: Tuple[int, int] = (-1, -1)):
        cls(cv2.MORPH_ELLIPSE, ksize, anchor)


OpencvStructuringElement.register("rectangle", "rectangle")(StructuringElement)
OpencvStructuringElement.register("cross", "cross")(StructuringElement)
OpencvStructuringElement.register("ellipse", "ellipse")(StructuringElement)


@StructuringElement.register("square")
class SquareStructuringElement(StructuringElement):
    def __init__(self, side: int):
        self.side = side

    @property
    def element(self) -> np.array:
        return morphology.square(self.side)


@StructuringElement.register("diamond")
class DiamondStructuringElement(StructuringElement):
    def __init__(self, radius: int):
        self.radius = radius

    @property
    def element(self) -> np.array:
        return morphology.diamond(self.radius)


@StructuringElement.register("disk")
class DiskStructuringElement(StructuringElement):
    def __init__(self, radius: int):
        self.radius = radius

    @property
    def element(self) -> np.array:
        return morphology.disk(self.radius)


@StructuringElement.register("octagon")
class OctagonStructuringElement(StructuringElement):
    def __init__(self, m: int, n: int):
        self.m = m
        self.n = n

    @property
    def element(self) -> np.array:
        return morphology.octagon(self.m, self.n)


@StructuringElement.register("star")
class StarStructuringElement(StructuringElement):
    def __init__(self, a: int):
        self.a = a

    @property
    def element(self) -> np.array:
        return morphology.star(self.a)
