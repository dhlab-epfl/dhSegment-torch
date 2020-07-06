from typing import Tuple, Optional

import cv2
import numpy as np
from skimage import morphology

from dh_segment_torch.post_processing.operation import Operation
from dh_segment_torch.post_processing.probabilities.morphology.structuring_element import (
    StructuringElement,
)
from dh_segment_torch.post_processing.probabilities.operation import ProbasIntOperation, ProbasOperation


class MorphologicalOperator(ProbasIntOperation):
    def __init__(
        self,
        op: int,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        self.op = op
        self.kernel = kernel
        self.anchor = anchor
        self.iterations = iterations

        if border_type == "constant":
            self.border_type = cv2.BORDER_CONSTANT
        elif border_type == "replicate":
            self.border_type = cv2.BORDER_REPLICATE
        elif border_type == "reflect":
            self.border_type = cv2.BORDER_REFLECT
        elif border_type == "wrap":
            self.border_type = cv2.BORDER_WRAP
        elif border_type == "reflect_101":
            self.border_type = cv2.BORDER_REFLECT101
        else:
            raise ValueError(f"Border type {border_type} is not supported.")

    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        return cv2.morphologyEx(
            input,
            op=self.op,
            kernel=self.kernel,
            anchor=self.anchor,
            iterations=self.iterations,
            borderType=self.border_type,
        )

    @classmethod
    def erode(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_ERODE, kernel, anchor, iterations, border_type)

    @classmethod
    def dilate(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_DILATE, kernel, anchor, iterations, border_type)

    @classmethod
    def open(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_OPEN, kernel, anchor, iterations, border_type)

    @classmethod
    def close(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_CLOSE, kernel, anchor, iterations, border_type)

    @classmethod
    def gradient(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_GRADIENT, kernel, anchor, iterations, border_type)

    @classmethod
    def top_hat(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_TOPHAT, kernel, anchor, iterations, border_type)

    @classmethod
    def black_hat(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_BLACKHAT, kernel, anchor, iterations, border_type)

    @classmethod
    def hit_miss(
        cls,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        return cls(cv2.MORPH_HITMISS, kernel, anchor, iterations, border_type)


Operation.register("erode", "erode")(MorphologicalOperator)
Operation.register("dilate", "dilate")(MorphologicalOperator)
Operation.register("open", "open")(MorphologicalOperator)
Operation.register("close", "close")(MorphologicalOperator)
Operation.register("gradient", "gradient")(MorphologicalOperator)
Operation.register("top_hat", "top_hat")(MorphologicalOperator)
Operation.register("black_hat", "black_hat")(MorphologicalOperator)
Operation.register("hit_miss", "hit_miss")(MorphologicalOperator)


@Operation.register("open_close")
class OpenClose(MorphologicalOperator):
    def __init__(
        self,
        kernel: StructuringElement,
        anchor: Tuple[int, int] = (-1, -1),
        iterations: int = -1,
        border_type: str = "constant",
    ):
        super().__init__(cv2.MORPH_OPEN, kernel, anchor, iterations, border_type)

    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        input = super().apply(input)
        return cv2.morphologyEx(
            input,
            op=cv2.MORPH_CLOSE,
            kernel=self.kernel,
            anchor=self.anchor,
            iterations=self.iterations,
            borderType=self.border_type,
        )


@Operation.register("skeletonize")
class Skeletonize(ProbasOperation):
    def __init__(self):
        pass

    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        return morphology.skeletonize(input)
