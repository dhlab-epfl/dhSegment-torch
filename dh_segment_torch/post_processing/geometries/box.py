import math
from typing import List, Optional, Union

import cv2
import numpy as np
from shapely import geometry
from scipy.spatial import KDTree

from dh_segment_torch.post_processing.operation import BinaryToGeometriesOperation, Operation


@Operation.register("box_detection")
class BoxDetection(BinaryToGeometriesOperation):
    def __init__(self,
                 box_type: str = 'min_rectangle',
                 min_area: float= 0.0,
                 p_arc_length: float = 0.01,
                 max_boxes=math.inf,
                 classes_sel: Optional[Union[int, List[int]]] = None):
        super().__init__(classes_sel)

        if box_type not in {'min_rectangle', 'rectangle', 'quadrilateral'}:
            raise ValueError(f"Box type {box_type} is not supported.")

        self.box_type = box_type
        self.min_area = min_area
        self.p_arc_length = p_arc_length
        self.max_boxes = max_boxes

    def apply(self, binary: np.array) -> List[geometry.base.BaseGeometry]:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours is None:
            return []
        found_boxes = []

        h_img, w_img = binary.shape[:2]

        def validate_box(box: np.array) -> (np.array, float):
            """
            :param box: array of 4 coordinates with format [[x1,y1], ..., [x4,y4]]
            :return: (box, area)
            """
            polygon = geometry.Polygon([point for point in box])
            if polygon.area > self.min_area * binary.size:
                # Correct out of range corners
                box = np.maximum(box, 0)
                box = np.stack((np.minimum(box[:, 0], binary.shape[1]),
                                np.minimum(box[:, 1], binary.shape[0])), axis=1)

                # return box
                return geometry.Polygon([point for point in box])

        if self.box_type == 'quadrilateral':
            for c in contours:
                epsilon = self.p_arc_length * cv2.arcLength(c, True)
                cnt = cv2.approxPolyDP(c, epsilon, True)
                # box = np.vstack(simplify_douglas_peucker(cnt[:, 0, :], 4))

                # Find extreme points in Convex Hull
                hull_points = cv2.convexHull(cnt, returnPoints=True)
                # points = cnt
                points = hull_points
                if len(points) > 4:
                    # Find closes points to corner using nearest neighbors
                    tree = KDTree(points[:, 0, :])
                    _, ul = tree.query((0, 0))
                    _, ur = tree.query((w_img, 0))
                    _, dl = tree.query((0, h_img))
                    _, dr = tree.query((w_img, h_img))
                    box = np.vstack([points[ul, 0, :], points[ur, 0, :],
                                     points[dr, 0, :], points[dl, 0, :]])
                elif len(hull_points) == 4:
                    box = hull_points[:, 0, :]
                else:
                    continue
                # Todo : test if it looks like a rectangle (2 sides must be more or less parallel)
                # todo : (otherwise we may end with strange quadrilaterals)
                if len(box) != 4:
                    self.box_type = 'min_rectangle'
                    print('Quadrilateral has {} points. Switching to minimal rectangle mode'.format(len(box)))
                else:
                    # found_box = validate_box(box)
                    found_boxes.append(validate_box(box))
        if self.box_type == 'min_rectangle':
            for c in contours:
                rect = cv2.minAreaRect(c)
                box = np.int0(cv2.boxPoints(rect))
                found_boxes.append(validate_box(box))
        elif self.box_type == 'rectangle':
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
                found_boxes.append(validate_box(box))

        # sort by area
        found_boxes = [box for box in found_boxes if box is not None]
        found_boxes = sorted(found_boxes, key=lambda box: box.area, reverse=True)
        return found_boxes[:min(self.max_boxes, len(found_boxes))]