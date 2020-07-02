from typing import Optional, Union, List

import numpy as np
from shapely import geometry

from dh_segment_torch.post_processing.operation import GeometriesToGeometriesOperation, Operation


@Operation.register("lines_filter")
class LinesFilter(GeometriesToGeometriesOperation):
    def __init__(self,
                 dist_thresh: int,
                 classes_sel: Optional[Union[int, List[int]]] = None):
        super().__init__(classes_sel)
        self.dist_thresh = dist_thresh

    def apply(self, geoms: List[geometry.base.BaseGeometry]) -> List[geometry.base.BaseGeometry]:
        """
        Filter a list of line coordinates by a threshold.
        Takes the list of line coordinates, ordered by rank of confidence (as output by opencv hough)
        If two lines coordinates are at a distance < threshold, then take the one with the highest confidence.
        This allows to filter clusters of lines.
        """

        to_remove = set()
        for idx1 in range(len(geoms)):
            if idx1 in to_remove:
                continue
            geom1 = geoms[idx1]
            r1 = idx1
            for idx2 in range(idx1 + 1, len(geoms)):
                if idx2 in to_remove:
                    continue
                geom2 = geoms[idx2]
                r2 = idx2
                dist = geom1.distance(geom2)

                if dist < self.dist_thresh:
                    if r1 < r2:
                        to_remove.add(idx2)
                    else:
                        to_remove.add(idx1)

        selection = set(list(range(len(geoms)))).difference(to_remove)

        geoms_filtered = []
        for index in selection:
            geoms_filtered.append(geoms[index])

        return geoms_filtered
