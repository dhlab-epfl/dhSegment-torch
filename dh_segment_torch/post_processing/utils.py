from typing import List, Any, Union

import numpy as np


def merge_lists(lists: List[List[Any]], recursive=False) -> List[Any]:
    lists = [item for list_ in lists for item in list_]
    if recursive and len(lists) > 0 and isinstance(lists[0], list):
        return merge_lists(lists)
    return lists


def normalize_min_area(min_area: Union[int, float], input: np.array):
    max_area = np.prod(input.shape)
    if isinstance(min_area, float):
        if 0 <= min_area <= 1:
            min_area = min_area * input.size
        min_area = int(round(min_area))
    return min(min_area, max_area)
