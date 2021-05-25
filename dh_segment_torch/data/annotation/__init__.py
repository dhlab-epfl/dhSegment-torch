from dh_segment_torch.data.annotation.readers.annotation_reader import *
from dh_segment_torch.data.annotation.readers.cvat_reader import *
from dh_segment_torch.data.annotation.readers.via.via2_reader import *
from dh_segment_torch.data.annotation.readers.via.via3_reader import *

from dh_segment_torch.data.annotation.writers.annotation_writer import *
from dh_segment_torch.data.annotation.writers.image_writer import *
from dh_segment_torch.data.annotation.writers.via.via2_writer import *
from dh_segment_torch.data.annotation.writers.via.via3_writer import *
from dh_segment_torch.data.annotation.writers.web_annotation.mirador_writer import *

from dh_segment_torch.data.annotation.annotation import *
from dh_segment_torch.data.annotation.annotation_iterator import *
from dh_segment_torch.data.annotation.annotation_painter import *
from dh_segment_torch.data.annotation.labels_annotations import *

from dh_segment_torch.data.annotation.shape import *


_READER = ["AnnotationReader", "CVATReader", "VIA2Reader", "VIA3Reader"]

_WRITER = ["AnnotationWriter", "ImageWriter", "VIA2Writer", "VIA3Writer", "MiradorWebAnnotationWriter"]

_ANNOTATION = [
    "Annotation",
    "AnnotationIterator",
    "AnnotationPainter",
    "LabelsAnnotations",
]

_SHAPE = [
    "Shape",
    "Circle",
    "Ellipse",
    "Line",
    "LineString",
    "MultiPolygon",
    "Point",
    "Polygon",
    "Rectangle",
]

__all__ = _READER + _WRITER + _ANNOTATION + _SHAPE
