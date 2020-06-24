from typing import Tuple, Optional, Union, List

import cv2
import numpy as np

from dh_segment_torch.post_processing.operation import (
    ProbasOperation,
    Operation, ProbasIntOperation,
)


@Operation.register("filter_gaussian")
class GaussianFilter(ProbasOperation):
    def __init__(
        self,
        sigma: float,
        ksize: Optional[Union[int, Tuple[int, int]]] = None,
        sigma_y: float = 0.0,
        borderType: int = cv2.BORDER_DEFAULT,
        classes_sel: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(classes_sel)
        if ksize is None:
            ksize = int(sigma * 3) * 2 + 1
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        self.ksize = ksize
        self.sigma = sigma
        self.sigma_y = sigma_y
        self.borderType = borderType

    def apply(self, input: np.array) -> np.array:
        return cv2.GaussianBlur(
            input, self.ksize, self.sigma, self.sigma_y, self.borderType
        )


@Operation.register("filter_median")
class MedianFilter(ProbasIntOperation):
    def __init__(self, ksize: int, classes_sel: Optional[Union[int, List[int]]] = None):
        super().__init__(classes_sel)

        self.ksize = ksize

    def apply(self, input: np.array) -> np.array:
        return cv2.medianBlur(input, self.ksize)


@Operation.register("filter_bilateral")
class BilateralFilter(ProbasOperation):
    def __init__(
        self,
        d: int,
        sigmaColor: float,
        sigmaSpace: float,
        borderType: int = cv2.BORDER_DEFAULT,
        classes_sel: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(classes_sel)

        self.d = d
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace
        self.borderType = borderType

    def apply(self, input: np.array) -> np.array:
        return cv2.bilateralFilter(
            input, self.d, self.sigmaColor, self.sigmaSpace, self.borderType
        )
