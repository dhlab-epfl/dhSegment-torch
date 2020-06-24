from typing import Optional, Union, List

import cv2
import numpy as np
from scipy.ndimage import label

from dh_segment_torch.post_processing.operation import (
    ProbasIntOperation,
    Operation,
)


@Operation.register("threshold")
class Thresholding(ProbasIntOperation):
    def __init__(
        self,
        low_threshold: Union[int, float],
        high_threshold: Union[int, float] = 1.0,
        threshold_mode: str = "binary",
        classes_sel: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(classes_sel)

        if low_threshold < 0:
            self.low_threshold = 0
            self.high_threshold = normalize_threshold(high_threshold)
            self.threshold_mode = parse_threshold_mode(threshold_mode)
            self.threshold_mode += cv2.THRESH_OTSU
        else:
            self.low_threshold = normalize_threshold(low_threshold)
            self.high_threshold = normalize_threshold(high_threshold)

    def apply(self, input: np.array) -> np.array:
        return cv2.threshold(
            input, self.low_threshold, self.high_threshold, self.threshold_mode
        )


@Operation.register("threshold_adaptative")
class AdaptiveThresholding(ProbasIntOperation):
    def __init__(
        self,
        high_threshold: Union[int, float],
        adaptive_method: str,
        blockSize: int,
        C: float,
        threshold_mode: str = "binary",
        classes_sel: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(classes_sel)

        self.high_threshold = normalize_threshold(high_threshold)
        if adaptive_method == "mean":
            self.adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        elif adaptive_method == "gaussian":
            self.adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            raise ValueError(
                f"Adaptive method {adaptive_method} not supported."
                f'Supporting only "mean" and "gaussian"'
            )
        self.blockSize = blockSize
        self.C = C

        self.threshold_mode = parse_threshold_mode(threshold_mode)

    def apply(self, input: np.array) -> np.array:
        return cv2.adaptiveThreshold(
            input,
            self.high_threshold,
            self.adaptive_method,
            self.threshold_mode,
            self.blockSize,
            self.C,
        )


@Operation.register("threshold_hysteresis")
class HysteresisThresholding(ProbasIntOperation):
    def __init__(
        self,
        low_threshold: Union[int, float],
        high_threshold: Union[int, float],
        vertical_local_maxima: bool = False,
        horizontal_local_maxima: bool = False,
        classes_sel: Optional[Union[int, List[int]]] = None,
    ):
        super().__init__(classes_sel)

        self.low_threshold = normalize_threshold(low_threshold)
        self.high_threshold = normalize_threshold(high_threshold)
        self.vertical_local_maxima = vertical_local_maxima
        self.horizontal_local_maxima = horizontal_local_maxima

    def apply(self, input: np.array) -> np.array:
        low_mask = input > self.low_threshold
        if self.vertical_local_maxima:
            low_mask &= vertical_local_maxima(input)
        if self.horizontal_local_maxima:
            low_mask &= horizontal_local_maxima(input)
        # Connected components extraction
        label_components, count = label(low_mask, np.ones((3, 3)))
        # Keep components with high threshold elements
        good_labels = np.unique(
            label_components[low_mask & (input > self.high_threshold)]
        )
        label_masks = np.zeros((count + 1,), np.uint8)
        label_masks[good_labels] = 255
        return label_masks[label_components]


def parse_threshold_mode(threshold_mode: str) -> int:
    if threshold_mode == "binary":
        return cv2.THRESH_BINARY
    elif threshold_mode == "binary_inv":
        return cv2.THRESH_BINARY_INV
    else:
        raise ValueError(
            f"Threshold mode {threshold_mode} not supported, "
            f'only supporting "binary" and "binary_inv"'
        )


def normalize_threshold(threshold: Union[int, float]) -> int:
    if isinstance(threshold, float):
        if threshold <= 1.0:
            threshold = round(threshold * 255)
        else:
            threshold = round(threshold)
    return threshold


def vertical_local_maxima(probs: np.ndarray) -> np.ndarray:
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[1:-1] = (probs[1:-1] >= probs[:-2]) & (probs[2:] <= probs[1:-1])
    local_maxima = cv2.morphologyEx(
        local_maxima.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    return local_maxima > 0


def horizontal_local_maxima(probs: np.ndarray) -> np.ndarray:
    local_maxima = np.zeros_like(probs, dtype=bool)
    local_maxima[:, 1:-1] = (probs[:, 1:-1] >= probs[:, :-2]) & (
        probs[:, 2:] <= probs[:, 1:-1]
    )
    local_maxima = cv2.morphologyEx(
        local_maxima.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    return local_maxima > 0
