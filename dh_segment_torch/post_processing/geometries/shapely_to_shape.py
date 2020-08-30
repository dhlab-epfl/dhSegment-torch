import math
from itertools import islice

import cv2
import numpy as np
from shapely import geometry, affinity

from dh_segment_torch.data.annotation.shape import (
    Point,
    Circle,
    Ellipse,
    LineString,
    Line,
    Rectangle,
    Polygon,
    MultiPolygon,
)
from dh_segment_torch.post_processing.operation import (
    GeometriesToShapesOperation,
    Operation,
)


@Operation.register("to_point")
class ToPoint(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> Point:
        (x,), (y,) = input.centroid.xy
        return Point((x, y), normalized_coords=False)


@Operation.register("to_circle")
class ToCircle(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> Circle:
        xmin, _, xmax, _ = input.bounds
        (x,), (y,) = input.centroid.xy
        radius = (xmax - xmin) / 2
        return Circle((x, y), radius, normalized_coords=False)


@Operation.register("to_linestring")
class ToLineString(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> LineString:
        return LineString(list(input.coords), normalized_coords=False)


@Operation.register("to_line")
class ToLine(GeometriesToShapesOperation):
    def apply_to_geom(self, input: geometry.base.BaseGeometry, *args, **kwargs) -> Line:
        start, end = list(input.coords)
        return Line(start, end, normalized_coords=False)


@Operation.register("to_ellipse")
class ToEllipse(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> Ellipse:
        _, _, angle = cv2.fitEllipse(np.array(list(input.exterior.coords)).astype(int))

        hull = input.convex_hull
        coords = hull.exterior.coords

        edges = (
            (pt2[0] - pt1[0], pt2[1] - pt1[1])
            for pt1, pt2 in zip(coords, islice(coords, 1, None))
        )

        def _transformed_rects():
            for dx, dy in edges:
                # compute the normalized direction vector of the edge vector
                length = math.sqrt(dx ** 2 + dy ** 2)
                ux, uy = dx / length, dy / length
                # compute the normalized perpendicular vector
                vx, vy = -uy, ux
                # transform hull from the original coordinate system to the coordinate system
                # defined by the edge and compute the axes-parallel bounding rectangle
                transf_rect = affinity.affine_transform(
                    hull, (ux, uy, vx, vy, 0, 0)
                ).envelope
                # yield the transformed rectangle and a matrix to transform it back
                # to the original coordinate system
                yield (transf_rect, (ux, vx, uy, vy, 0, 0))

        # check for the minimum area rectangle and return it
        transf_rect, inv_matrix = min(_transformed_rects(), key=lambda r: r[0].area)

        xmin, ymin, xmax, ymax = transf_rect.bounds
        (x,), (y,) = input.centroid.xy
        yradius = (xmax - xmin) / 2
        xradius = (ymax - ymin) / 2

        return Ellipse((x, y), (xradius, yradius), angle, normalized_coords=False)


@Operation.register("to_rectangle")
class ToRectangle(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.base.BaseGeometry, *args, **kwargs
    ) -> Rectangle:
        xmin, ymin, xmax, ymax = input.bounds

        return Rectangle(((xmin, ymin), (xmax, ymax)), normalized_coords=False)


@Operation.register("to_polygon")
class ToPolygon(GeometriesToShapesOperation):
    def apply_to_geom(self, input: geometry.Polygon, *args, **kwargs) -> Polygon:
        shell = list(input.exterior.coords)
        holes = [[list(hole.coords)] for hole in input.interiors]
        return Polygon(shell, holes, normalized_coords=False)


@Operation.register("to_multipolygon")
class ToMultiPolygon(GeometriesToShapesOperation):
    def apply_to_geom(
        self, input: geometry.MultiPolygon, *args, **kwargs
    ) -> MultiPolygon:
        output = []
        for poly in input:
            shell = list(poly.exterior.coords)
            holes = [[list(hole.coords)] for hole in poly.interiors]
            output.append((shell, holes))
        return MultiPolygon(output, normalized_coords=False)
