import math
from typing import List, Union

import cv2
import numpy as np
from shapely import geometry

from dh_segment_torch.post_processing.operation import (
    Operation,
    BinaryToGeometriesOperation,
)
from dh_segment_torch.post_processing.utils import normalize_min_area


@Operation.register("polygon_detection")
class PolygonDetection(BinaryToGeometriesOperation):
    def __init__(
        self, min_area: Union[int, float] = 0.0, max_polygons=math.inf,
    ):

        self.min_area = min_area
        self.max_polygons = max_polygons

    def apply(
        self, binary: np.array, *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        min_area = normalize_min_area(self.min_area, binary)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours is None:
            return []
        found_polygons = []

        for contour in contours:
            if len(contour) < 3:
                continue
            polygon = geometry.Polygon([point[0] for point in contour])
            if polygon.area > min_area:
                found_polygons.append(polygon)

        # sort by area
        found_polygons = sorted(
            found_polygons, key=lambda poly: poly.area, reverse=True
        )
        return found_polygons[: min(self.max_polygons, len(found_polygons))]
