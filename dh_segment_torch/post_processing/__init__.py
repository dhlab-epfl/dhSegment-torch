from dh_segment_torch.post_processing.annotation.assign_label import *
from dh_segment_torch.post_processing.annotation.to_annotation import *
from dh_segment_torch.post_processing.annotation.to_labels_annotations import *
from dh_segment_torch.post_processing.geometries.box import *
from dh_segment_torch.post_processing.geometries.filter import *
from dh_segment_torch.post_processing.geometries.lines.lines import *
from dh_segment_torch.post_processing.geometries.lines.lines_filter import *
from dh_segment_torch.post_processing.geometries.lines.lines_page import *
from dh_segment_torch.post_processing.geometries.lines.lines_to_columns import *
from dh_segment_torch.post_processing.geometries.shapely_to_shape import *
from dh_segment_torch.post_processing.operation import *
from dh_segment_torch.post_processing.post_processing_pipeline import *
from dh_segment_torch.post_processing.probabilities.filters import *
from dh_segment_torch.post_processing.probabilities.thresholding import *

_BASE = [
    "Operation",
    "NoOperation",
    "ClasswiseNoOperation",
    "SplitOperation",
    "ConcatLists",
    "MergeLists",
    "ExtractIndexOpration",
    "IntermediaryOutput",
    "MergeListsOperation",
    "PostProcessingPipeline",
    "DagPipeline",
    "OperationsInputs",
]

_FILTER = ["GaussianFilter", "MedianFilter", "BilateralFilter"]

_THRESHOLDING = ["Thresholding", "AdaptiveThresholding", "HysteresisThresholding"]

_MORPHOLOGY = [
    "StructuringElement",
    "SquareStructuringElement",
    "DiamondStructuringElement",
    "DiskStructuringElement",
    "OctagonStructuringElement",
    "StarStructuringElement",
    "FilterSmallObjects",
    "FilterSmallHoles",
    "MorphologicalOperator",
    "OpenClose",
    "Skeletonize"
]

_GEOMETRY = [
    "LinesPage",
    "LinesDetection",
    "LinesFilter",
    "LinesToColumns",
    "BoxDetection",
    "PolygonDetection",
    "MaskByGeometries",
    "FilterByOverlappingGeometries",
    "FilterByGeometryArea",
    "FilterByGeometryLength",
    "SimplifyGeometries",
    "ConvexHullGeometries",
    "BoudingRectGeometries",
    "MinimumBoudingRectGeometries",
]


_SHAPELY_TO_SHAPE = [
    "to_point",
    "to_circle",
    "to_linestring",
    "to_line",
    "to_ellipse",
    "to_rectangle",
    "to_polygon",
    "to_multipolygon",
]

_ANNOTATION = [
    "AssignLabel",
    "ToLabelsAnnotations",
    "NormalizeLabelsAnnotations",
    "ToAnnotation",
]


__all__ = (
    _BASE
    + _FILTER
    + _THRESHOLDING
    + _MORPHOLOGY
    + _GEOMETRY
    + _SHAPELY_TO_SHAPE
    + _ANNOTATION
)
