from typing import Union

import cv2
import numpy as np

from dh_segment_torch.post_processing.operation import Operation
from dh_segment_torch.post_processing.probabilities.operation import ProbasIntOperation
from dh_segment_torch.post_processing.utils import normalize_min_area


@Operation.register("filter_small_objects")
class FilterSmallObjects(ProbasIntOperation):
    def __init__(self, min_area: Union[int, float], connectivity: int = 1):
        self.min_area = min_area
        if connectivity == 1:
            self.connectivity = 4
        elif connectivity == 2:
            self.connectivity = 8
        else:
            raise ValueError(f"Expected connectivity to be 1 or 2, not {connectivity}.")

    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        if self.min_area == 0:
            return input
        min_area = normalize_min_area(self.min_area, input)

        _, ccs, = cv2.connectedComponents(input, connectivity=self.connectivity)
        component_sizes = np.bincount(ccs.ravel())

        too_small = component_sizes < min_area
        too_small_mask = too_small[ccs]

        out = input.copy()
        out[too_small_mask] = 0
        return out


@Operation.register("filter_small_holes")
class FilterSmallHoles(FilterSmallObjects):
    def apply(self, input: np.array, *args, **kwargs) -> np.array:
        out = 1 - input
        return 1 - super().apply(out)
