from typing import Union, List, Optional, Tuple

import pandas as pd
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from lxml import etree

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.readers.annotation_reader import AnnotationReader
from dh_segment_torch.data.annotation.readers.utils import annotation_data_to_data
from dh_segment_torch.data.annotation.shape import (
    Rectangle,
    Polygon,
    Line,
    LineString,
    Point,
)
from dh_segment_torch.data.annotation.utils import Coordinates, append_image_dir


@AnnotationReader.register("cvat")
class CVATReader(AnnotationReader):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):

        self.point_radius = point_radius
        self.line_thickness = line_thickness
        super().__init__(file_path, images_dir, image_auth)

        self.elements_parsers = [
            ("box", self.parse_rectangle),
            ("polygon", self.parse_polygon),
            ("polyline", self.parse_polyline),
            ("points", self.parse_points),
        ]

    def _read_data(self, path: str, image_dir: Optional[str] = None) -> pd.DataFrame:
        tree = etree.parse(path)
        root = tree.getroot()
        images_trees = root.findall("image")

        annotations_data = []
        all_paths = []

        for image_tree in images_trees:
            image_name = image_tree.attrib["name"]
            image_height = int(image_tree.attrib["height"])
            image_width = int(image_tree.attrib["width"])
            image_size = ImageSize(image_height, image_width)
            path = append_image_dir(image_name, image_dir)
            all_paths.append(path)

            labels_annotations = LabelsAnnotations()
            for element, parser in self.elements_parsers:
                for elem_to_parse in image_tree.findall(element):
                    parsed_labels_shapes = parser(elem_to_parse)
                    for label, shape in parsed_labels_shapes:
                        if label not in labels_annotations:
                            labels_annotations[label] = []
                        labels_annotations[label].append(shape)
            if len(labels_annotations) == 0:
                labels_annotations = None
            annotations_data.append((path, image_size, labels_annotations))
        annotations_data = pd.DataFrame(
            annotations_data, columns=["path", "image_size", "labels_annotation"]
        )
        return annotations_data

    def _transform_data_row_to_annot(self, row: pd.Series) -> Annotation:
        return Annotation(row['path'], image_size=row['image_size'], labels_annotations=row['labels_annotation'])

    def parse_rectangle(self, box_tree):
        label = box_tree.attrib["label"]
        tl = round_int(box_tree.attrib["xtl"]), round_int(box_tree.attrib["ytl"])
        br = round_int(box_tree.attrib["xbr"]), round_int(box_tree.attrib["ybr"])
        return [(label, Rectangle((tl, br), normalized_coords=False))]

    def parse_polygon(self, polygon_tree):
        label = polygon_tree.attrib["label"]
        coords = get_coords(polygon_tree.attrib["points"])
        return [(label, Polygon(coords, normalized_coords=False))]

    def parse_polyline(self, polyline_tree):
        label = polyline_tree.attrib["label"]
        coords = get_coords(polyline_tree.attrib["points"])
        if len(coords) == 2:
            return [
                (
                    label,
                    Line(
                        coords[0],
                        coords[1],
                        self.line_thickness,
                        normalized_coords=False,
                    ),
                )
            ]
        else:
            return [
                (
                    label,
                    LineString(coords, self.line_thickness, normalized_coords=False),
                )
            ]

    def parse_points(self, points_tree):
        label = points_tree.attrib["label"]
        coords = get_coords(points_tree.attrib["points"])
        return [
            (label, Point(coord, self.point_radius, normalized_coords=False))
            for coord in coords
        ]


def round_int(number: str) -> int:
    return int(round(float(number)))


def get_coords(points: str) -> List[Coordinates]:
    tuples = points.split(";")
    coords = [tuple([round_int(i) for i in j.split(",")]) for j in tuples]
    return coords
