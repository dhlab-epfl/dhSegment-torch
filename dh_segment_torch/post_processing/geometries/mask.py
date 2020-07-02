from typing import List

from shapely import geometry

from dh_segment_torch.post_processing.operation import Operation


@Operation.register("geometries_filter")
class GeometriesFiltering(Operation):
    def __init__(self, filter_index: int, min_overlap: float = 0.6):
        self.filter_index = filter_index
        self.min_overlap = min_overlap

    def apply(
        self, geometries: List[List[geometry.base.BaseGeometry]]
    ) -> List[List[geometry.base.BaseGeometry]]:
        filtering_geometries = geometries[self.filter_index]

        return [
            filter_geometries_by_geometries(
                geoms, filtering_geometries, self.min_overlap
            )
            if idx != self.filter_index
            else geoms
            for idx, geoms in enumerate(geometries)
        ]


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


@Operation.register("geometries_mask")
class GeometriesMasking(Operation):
    def __init__(self, mask_index: int):
        self.mask_index = mask_index

    def apply(
        self, geometries: List[List[geometry.base.BaseGeometry]]
    ) -> List[List[geometry.base.BaseGeometry]]:
        masking_geometries = geometries[self.mask_index]

        return [
            mask_geometries_by_geometries(geoms, masking_geometries)
            if idx != self.mask_index
            else geoms
            for idx, geoms in enumerate(geometries)
        ]


def mask_geometries_by_geometries(
    geometries: List[geometry.base.BaseGeometry],
    masking_geometries: List[geometry.base.BaseGeometry],
):
    masked_geometries = []

    for geometry in geometries:
        for mask in masking_geometries:
            geometry = geometry.intersection(mask)
        masked_geometries.append(geometry)
    return masked_geometries
