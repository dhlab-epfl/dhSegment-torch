import json
import os
from collections.abc import Mapping
import pandas as pd
from typing import List, Optional, Callable, Dict, Any

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_reader import AnnotationReader
from dh_segment_torch.data.annotation.readers.via.utils import data_to_annotations
from dh_segment_torch.data.annotation.readers.via.via_shapes_parser import parse_via2_shape


class VIA2Reader(AnnotationReader):
    def __init__(
        self,
        data_getter: Callable[[str], List[Any]],
        attrib_name: str,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):
        self.data_getter = data_getter
        self.attrib_name = attrib_name
        self.point_radius = point_radius
        self.line_thickness = line_thickness

    def read(self, path: str, image_dir: Optional[str] = None) -> List[Annotation]:
        with open(path, 'r') as infile:
            data_raw = self.data_getter(json.load(infile))
        data = []
        all_paths = []
        for item in data_raw:
            path = item['filename']
            if image_dir:
                path = os.path.join(image_dir, path)
            all_paths.append(path)
            for region in item['regions']:
                shape = parse_via2_shape(region['shape_attributes'])
                labels = []
                if self.attrib_name in region['region_attributes']:
                    label = region['region_attributes'][self.attrib_name]
                    if isinstance(label, str):
                        if len(label) > 0:
                            labels.append(label)
                    elif isinstance(label, Mapping):
                        for label_name, status in label.items():
                            if status:
                                labels.append(label_name)
                data.append((path, shape, labels))
        data = pd.DataFrame(data, columns=['path', 'shape', 'label'])

        return data_to_annotations(data, all_paths)

    @classmethod
    def from_annotation_file(
        cls, attrib_name: str, point_radius: int = 5, line_thickness: int = 2
    ):
        return cls(lambda x: list(x.values()), attrib_name, point_radius, line_thickness)

    @classmethod
    def from_project_file(
        cls, attrib_name: str, point_radius: int = 5, line_thickness: int = 2
    ):
        return cls(
            lambda x: list(x["_via_img_metadata"].values()),
            attrib_name,
            point_radius,
            line_thickness,
        )


AnnotationReader.register("via2", "from_annotation_file")(VIA2Reader)
AnnotationReader.register("via2_project", "from_project_file")(VIA2Reader)
