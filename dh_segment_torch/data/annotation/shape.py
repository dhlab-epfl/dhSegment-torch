from typing import List, Tuple, Optional

import cv2
import numpy as np
from shapely import geometry, ops, affinity

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.utils import (
    Coordinates,
    convert_coord_to_image,
    convert_coord_to_normalized,
    int_coords,
)


class Shape(Registrable):
    def __init__(self, normalized_coords: bool = True):
        self.normalized_coords = normalized_coords

    def mask(self, mask_size: ImageSize) -> np.array:
        raise NotImplementedError

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
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

    def expanded_coords(self, image_size: ImageSize):
        raise NotImplementedError

    def normalize_coords(self, image_size: ImageSize):
        raise NotImplementedError

    def _raise_not_normalized(self):
        if not self.normalized_coords:
            raise ValueError("Can only get expended coords of normalized shape.")


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

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        center = self.coord_to_image(self.center, mask_size)
        return geometry.Point(center).buffer(self.radius)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.center = convert_coord_to_normalized(
            self.center, image_size.height, image_size.width
        )
        self.normalized_coords = True

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        return self.coord_to_image(self.center, image_size), self.radius


@Shape.register("point")
class Point(Circle):
    def __init__(
        self, center: Coordinates, radius: int = 1, normalized_coords: bool = True
    ):
        super().__init__(center, radius, normalized_coords)

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        center = self.coord_to_image(self.center, mask_size)
        return geometry.Point(center)


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

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        coordinates = self.coords_to_image(self.coordinates, mask_size)
        return geometry.LineString(coordinates)

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.coordinates = [
            convert_coord_to_normalized(coords, image_size.height, image_size.width)
            for coords in self.coordinates
        ]
        self.normalized_coords = True

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        return self.coords_to_image(self.coordinates, image_size)


@Shape.register("line")
class Line(LineString):
    def __init__(
        self,
        start: Coordinates,
        end: Coordinates,
        thickness: int = 1,
        normalized_coords: bool = True,
    ):
        super().__init__([start, end], thickness, normalized_coords)


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

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        center = self.coord_to_image(self.center, mask_size)
        x_radius, y_radius = self.coord_to_image(self.radiuses, mask_size)
        circle = geometry.Point(center).buffer(1)
        ellipse = affinity.scale(circle, x_radius, y_radius)
        ellipse_rotated = affinity.rotate(
            ellipse, self.angle
        )  # TODO check rotation angle
        return ellipse_rotated

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

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        return (
            self.coord_to_image(self.center, image_size),
            self.coord_to_image(self.radiuses, image_size),
            self.angle,
        )


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

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        corner1 = self.coord_to_image(self.corner1, mask_size)
        corner2 = self.coord_to_image(self.corner2, mask_size)
        xmin = min(corner1[0], corner2[0])
        ymin = min(corner1[1], corner2[1])
        xmax = max(corner1[0], corner2[0])
        ymax = max(corner1[1], corner2[1])
        return geometry.box(xmin, ymin, xmax, ymax)

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

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        return (
            self.coord_to_image(self.corner1, image_size),
            self.coord_to_image(self.corner2, image_size),
        )


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

    def geometry(self, mask_size: ImageSize) -> geometry.base.BaseGeometry:
        return affinity.scale(
            self.polygons, mask_size.width, mask_size.height, origin=(0, 0)
        )

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.polygons = affinity.scale(
            self.polygons, 1 / image_size.width, 1 / image_size.height, origin=(0, 0)
        )
        self.normalized_coords = True

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        polys = affinity.scale(
            self.polygons, image_size.width, image_size.height, origin=(0, 0)
        )

        polys_coords = [
            (
                int_coords(poly.exterior.coords),
                [int_coords(interior.coords) for interior in poly.interiors],
            )
            for poly in polys
        ]
        return polys_coords


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

    def geometry(self, mask_size: ImageSize) -> geometry.MultiPolygon:
        return ops.unary_union([poly.geometry(mask_size) for poly in self.polygons])

    def normalize_coords(self, image_size: ImageSize):
        if self.normalized_coords:
            return
        self.polygons = affinity.scale(
            self.polygons, 1 / image_size.width, 1 / image_size.height, origin=(0, 0)
        )
        self.normalized_coords = True

    def expanded_coords(self, image_size: ImageSize):
        self._raise_not_normalized()
        polys_coords = [
            poly_coords
            for polys in self.polygons
            for poly_coords in polys.expanded_coords(image_size)
        ]
        return polys_coords


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
