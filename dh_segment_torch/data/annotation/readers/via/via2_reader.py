import json
from collections import OrderedDict
from collections.abc import Mapping
from typing import List, Optional, Callable, Any, Tuple, Union

import pandas as pd

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.readers.annotation_reader import AnnotationReader
from dh_segment_torch.data.annotation.readers.via.utils import (
    annotation_data_to_data,
    data_row_to_annotation,
)
from dh_segment_torch.data.annotation.readers.via.via_shapes_parser import (
    parse_via2_shape,
)
from dh_segment_torch.data.annotation.utils import append_image_dir


class VIA2Reader(AnnotationReader):
    def __init__(
        self,
        data_getter: Callable[[str], List[Any]],
        attrib_name: str,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):
        self.data_getter = data_getter
        self.attrib_name = attrib_name
        self.point_radius = point_radius
        self.line_thickness = line_thickness
        super().__init__(file_path, images_dir, image_auth)

    def _read_data(self, path: str, image_dir: Optional[str] = None) -> pd.DataFrame:
        with open(path, "r") as infile:
            data_raw = self.data_getter(
                json.load(infile, object_pairs_hook=OrderedDict)
            )
        annotations_data = []
        all_paths = []
        for item in data_raw:
            if "filename" not in item:
                raise ValueError(
                    "The data was not loaded correctly, "
                    "certainly due to the fact that is is project file "
                    "and not an annotation file."
                )
            path = item["filename"]
            path = append_image_dir(path, image_dir)
            all_paths.append(path)
            for region in item["regions"]:
                shape = parse_via2_shape(region["shape_attributes"])
                labels = []
                if self.attrib_name in region["region_attributes"]:
                    label = region["region_attributes"][self.attrib_name]
                    if isinstance(label, str):
                        if len(label) > 0:
                            labels.append(label)
                    elif isinstance(label, Mapping):
                        for label_name, status in label.items():
                            if status:
                                labels.append(label_name)
                annotations_data.append((path, shape, labels))
        annotations_data = pd.DataFrame(
            annotations_data, columns=["path", "shape", "label"]
        )
        return annotation_data_to_data(annotations_data, all_paths)

    def _transform_data_row_to_annot(self, row) -> Annotation:
        return data_row_to_annotation(row, self.image_auth)

    @classmethod
    def from_annotation_file(
        cls,
        attrib_name: str,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):
        return cls(
            lambda x: list(x.values()),
            attrib_name,
            file_path,
            images_dir,
            image_auth,
            point_radius,
            line_thickness,
        )

    @classmethod
    def from_project_file(
        cls,
        attrib_name: str,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):
        return cls(
            lambda x: list(x["_via_img_metadata"].values()),
            attrib_name,
            file_path,
            images_dir,
            image_auth,
            point_radius,
            line_thickness,
        )


AnnotationReader.register("via2", "from_annotation_file")(VIA2Reader)
AnnotationReader.register("via2_project", "from_project_file")(VIA2Reader)
