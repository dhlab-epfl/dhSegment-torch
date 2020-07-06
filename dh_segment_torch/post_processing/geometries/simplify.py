from typing import List

from shapely import geometry

from dh_segment_torch.post_processing.operation import Operation


@Operation.register("simplify_geometries")
class SimplifyGeometries(Operation):
    def __init__(self, tolerance: float, preserve_topology: bool = True):
        self.tolerance = tolerance
        self.preserve_topology = preserve_topology

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [
            geom.simplify(self.tolerance, self.preserve_topology) for geom in geometries
        ]


@Operation.register("convex_hull")
class ConvexHullGeometries(Operation):
    def __init__(self):
        pass

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [geom.convex_hull for geom in geometries]


@Operation.register("bounding_rect")
class BoudingRectGeometries(Operation):
    def __init__(self):
        pass

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [geom.envelope for geom in geometries]


@Operation.register("minimum_bounding_rect")
class MinimumBoudingRectGeometries(Operation):
    def __init__(self):
        pass

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [geom.minimum_rotated_rectangle for geom in geometries]
