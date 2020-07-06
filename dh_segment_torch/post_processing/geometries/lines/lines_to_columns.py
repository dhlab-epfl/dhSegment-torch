import logging
from typing import List

import numpy as np
from shapely import geometry, affinity

from dh_segment_torch.post_processing.operation import Operation

logger = logging.getLogger(__name__)


class LinesToColumns(Operation):
    def __init__(self, reference_angle: int):
        self.reference_angle = reference_angle

    def apply(
        self, lines: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:

        if len(lines) < 2:
            logger.warning(f"Only {len(lines)} lines, cannot infer any columns.")
            return []
        _, _, xmax, ymax = geometry.MultiLineString(lines).bounds
        diag_max = np.sqrt(xmax ** 2 + ymax ** 2)
        reference_line = geometry.LineString([[0, 0], [0, diag_max]])
        reference_line = affinity.rotate(
            reference_line, self.reference_angle, origin=(0, 0)
        )

        sorted_lines = sorted(lines, key=lambda line: line.distance(reference_line))

        polys = []
        for line1, line2 in zip(sorted_lines[:-1], sorted_lines[1:]):
            (xmin1, ymin1), (xmax1, ymax1) = line1.coords
            (xmin2, ymin2), (xmax2, ymax2) = line2.coords

            polys.append(
                geometry.Polygon(
                    [(xmin1, ymin1), (xmax1, ymax1), (xmax2, ymax2), (xmin2, ymin2)]
                )
            )
        return polys

    @classmethod
    def vertical_columns(cls):
        return cls(0)

    @classmethod
    def horizontal_columns(cls):
        return cls(-90)


Operation.register("lines_to_columns")(LinesToColumns)
Operation.register("vertical_lines_to_columns", "vertical_columns")(LinesToColumns)
Operation.register("horizontal_lines_to_columns", "horizontal_columns")(LinesToColumns)
