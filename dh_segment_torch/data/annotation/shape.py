from typing import List, Tuple, Optional

import cv2
import numpy as np
from shapely import geometry, ops, affinity

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.utils import (
    Coordinates,
    convert_coord_to_image,
    convert_coord_to_normalized,
)
from dh_segment_torch.data.annotation.image_size import ImageSize


class Shape(Registrable):
    def __init__(self, normalized_coords: bool = True):
        self.normalized_coords = normalized_coords

    def mask(self, mask_size: ImageSize) -> np.array:
        raise NotImplementedError

    def coords_to_image(
        self, coords: List[Coordinates], mask_size: ImageSize
    ) -> List[Tuple[int, int]]:
        return [self.coord_to_image(coord, mask_size) for coord in coords]

    def coord_to_image(
        self, coord: Coordinates, mask_size: ImageSize
    ) -> Tuple[int, int]:
        convert_height, convert_width = (
            (mask_size.height, mask_size.width) if self.normalized_coords else (1, 1)
        )
        return convert_coord_to_image(coord, convert_height, convert_width)

    def normalize_coords(self, image_size: ImageSize):
        raise NotImplementedError


@Shape.register("circle")
class Circle(Shape):
    def __init__(
        self, center: Coordinates, radius: int = 5, normalized_coords: bool = True
    ):
        super().__init__(normalized_coords)
        self.center = center
        self.radius = radius

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width))
        coordinate = self.coord_to_image(self.center, mask_size)
        mask = cv2.circle(mask, coordinate, self.radius, 1, thickness=-1)
        return mask.astype(bool)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.center = convert_coord_to_normalized(
            self.center, image_size.height, image_size.width
        )
        self.normalized_coords = True


@Shape.register("line_string")
class LineString(Shape):
    def __init__(
        self,
        coordinates: List[Coordinates],
        thickness: int = 1,
        normalized_coords: bool = True,
    ):
        super().__init__(normalized_coords)
        self.coordinates = coordinates
        self.thickness = thickness

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width))
        coordinates = self.coords_to_image(self.coordinates, mask_size)
        mask = cv2.polylines(
            mask, np.int32([coordinates]), False, 1, thickness=self.thickness
        )
        return mask.astype(bool)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.coordinates = [
            convert_coord_to_normalized(coords, image_size.height, image_size.width)
            for coords in self.coordinates
        ]
        self.normalized_coords = True


@Shape.register("ellipse")
class Ellipse(Shape):
    def __init__(
        self,
        center: Coordinates,
        radiuses: Coordinates,
        angle: float,
        normalized_coords: bool = True,
    ):
        """
        :param center:
        :param radius:
        :param angle: Angle in degrees
        """
        super().__init__(normalized_coords)
        self.center = center
        self.radiuses = radiuses
        self.angle = angle

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width))
        center = self.coord_to_image(self.center, mask_size)
        radiuses = self.coord_to_image(self.radiuses, mask_size)
        mask = cv2.ellipse(
            mask, center, radiuses, self.angle, 0, 360, color=1, thickness=-1
        )
        return mask.astype(bool)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.center = convert_coord_to_normalized(
            self.center, image_size.height, image_size.width
        )
        self.radiuses = convert_coord_to_normalized(
            self.radiuses, image_size.height, image_size.width
        )
        self.normalized_coords = True


@Shape.register("rectangle")
class Rectangle(Shape):
    def __init__(
        self, corners: Tuple[Coordinates, Coordinates], normalized_coords: bool = True
    ):
        super().__init__(normalized_coords)
        self.corner1, self.corner2 = corners

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width))
        corner1 = self.coord_to_image(self.corner1, mask_size)
        corner2 = self.coord_to_image(self.corner2, mask_size)
        mask = cv2.rectangle(mask, corner1, corner2, 1, thickness=-1)
        return mask.astype(bool)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.corner1 = convert_coord_to_normalized(
            self.corner1, image_size.height, image_size.width
        )
        self.corner2 = convert_coord_to_normalized(
            self.corner2, image_size.height, image_size.width
        )
        self.normalized_coords = True


@Shape.register("polygon")
class Polygon(Shape):
    def __init__(
        self,
        shell: List[Coordinates],
        holes: Optional[List[List[Coordinates]]] = None,
        fix: bool = True,
        force_valid: bool = False,
        normalized_coords: bool = True,
    ):
        super().__init__(normalized_coords)
        self.polygons = geometry.Polygon(shell, holes=holes)
        if fix:
            self.polygons = fix_poly(self.polygons)
        if isinstance(self.polygons, geometry.Polygon):
            self.polygons = geometry.MultiPolygon([self.polygons])
        if force_valid and not self.polygons.is_valid:
            raise ValueError("Poly is not valid")

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width))

        exteriors = [
            np.int32(self.coords_to_image(poly.exterior.coords, mask_size))
            for poly in self.polygons
        ]
        interiors = [
            np.int32(self.coords_to_image(hole.exterior.coords, mask_size))
            for poly in self.polygons
            for hole in poly.interiors
        ]
        mask = cv2.fillPoly(mask, exteriors, 1)
        mask = cv2.fillPoly(mask, interiors, 0)
        return mask.astype(bool)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.polygons = affinity.scale(
            self.polygons, 1 / image_size.width, 1 / image_size.height, origin=(0, 0)
        )
        self.normalized_coords = True


@Shape.register("multi_polygon")
class MultiPolygon(Shape):
    def __init__(
        self,
        polygons: List[Tuple[List[Coordinates], Optional[List[List[Coordinates]]]]],
        fix: bool = True,
        force_valid: bool = False,
        normalized_coords: bool = True,
    ):
        super().__init__(normalized_coords)
        self.polygons = [
            Polygon(shell, holes, fix, force_valid, normalized_coords)
            for shell, holes in polygons
        ]

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width), dtype=bool)
        for poly in self.polygons:
            mask |= poly.mask(mask_size)
        return mask

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.polygons = affinity.scale(
            self.polygons, 1 / image_size.width, 1 / image_size.height, origin=(0, 0)
        )
        self.normalized_coords = True


def fix_poly(poly: geometry.Polygon):
    if poly.is_valid:
        return poly
    area = poly.area
    fixed_poly = poly.buffer(1e-8).buffer(-1e-8)
    if abs(area - fixed_poly.area) < 1e-4:
        return fixed_poly
    return fix_bad_poly(poly)


def fix_bad_poly(poly: geometry.Polygon):
    holes = [fix_poly(geometry.Polygon(h.coords)) for h in poly.interiors]
    line_string = geometry.LineString(poly.exterior.coords)
    multi_line_string = ops.unary_union(line_string)
    polygons = list(ops.polygonize(multi_line_string))
    for idx in range(len(polygons)):
        for hole in holes:
            polygons[idx] = polygons[idx].difference(hole)
    polygons = geometry.MultiPolygon(polygons)
    if polygons.area > 0:
        return polygons
    return poly
