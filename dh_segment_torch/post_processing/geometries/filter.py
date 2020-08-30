from typing import List, Union

from shapely import geometry

from dh_segment_torch.post_processing.operation import Operation


@Operation.register("filter_by_geometry_area")
class FilterByGeometryArea(Operation):
    def __init__(self, min_area: Union[int, float]):
        self.min_area = min_area

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [geom for geom in geometries if geom.area > self.min_area]


@Operation.register("filter_by_geometry_length")
class FilterByGeometryLength(Operation):
    def __init__(self, min_length: Union[int, float]):
        self.min_length = min_length

    def apply(
        self, geometries: List[geometry.base.BaseGeometry], *args, **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return [geom for geom in geometries if geom.length > self.min_length]


@Operation.register("filter_by_geometries_overlap")
class FilterByOverlappingGeometries(Operation):
    def __init__(self, min_overlap: float = 0.6):
        self.min_overlap = min_overlap

    def apply(
        self,
        geometries: List[geometry.base.BaseGeometry],
        filtering_geometries: List[geometry.base.BaseGeometry],
        *args,
        **kwargs
    ) -> List[geometry.base.BaseGeometry]:

        return filter_geometries_by_geometries(
            geometries, filtering_geometries, self.min_overlap
        )


def filter_geometries_by_geometries(
    geometries: List[geometry.base.BaseGeometry],
    filtering_geometries: List[geometry.base.BaseGeometry],
    min_overlap: float,
):
    filtered_geometries = []
    for geometry in geometries:
        if any(
            [
                geometries_overlap(geometry, filtering_geometry) > min_overlap
                for filtering_geometry in filtering_geometries
            ]
        ):
            filtered_geometries.append(geometry)
    return filtered_geometries


def geometries_overlap(
    reference: geometry.base.BaseGeometry, comparison: geometry.base.BaseGeometry
) -> float:
    return reference.intersection(comparison).area / reference.area


@Operation.register("mask_by_geometries")
class MaskByGeometries(Operation):
    def __init__(self):
        pass

    def apply(
        self,
        geometries: List[geometry.base.BaseGeometry],
        masking_geometries: List[geometry.base.BaseGeometry],
        *args,
        **kwargs
    ) -> List[geometry.base.BaseGeometry]:
        return mask_geometries_by_geometries(geometries, masking_geometries)


def mask_geometries_by_geometries(
    geometries: List[geometry.base.BaseGeometry],
    masking_geometries: List[geometry.base.BaseGeometry],
):
    masked_geometries = []

    for geometry in geometries:
        for mask in masking_geometries:
            geometry = geometry.intersection(mask)
        if geometry.length > 0.0:
            masked_geometries.append(geometry)
    return masked_geometries
