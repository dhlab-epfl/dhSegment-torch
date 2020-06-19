import json
import logging
import os
from io import StringIO
from typing import List, Union, Optional, Dict, Any, Callable, Set, Tuple

import pandas as pd

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_reader import AnnotationReader
from dh_segment_torch.data.annotation.readers.via.via_shapes_parser import parse_via3_shape

from dh_segment_torch.data.annotation.readers.via.utils import data_to_annotations, annotation_data_to_data, \
    data_row_to_annotation
from dh_segment_torch.data.annotation.utils import reverse_dict, append_image_dir

logger = logging.getLogger("__name__")

ID_TO_SHAPE = {
    1: "POINT",
    2: "RECTANGLE",
    3: "CIRCLE",
    4: "ELLIPSE",
    5: "LINE",
    6: "POLYLINE",
    7: "POLYGON",
    8: "EXTREME_RECTANGLE",
    9: "EXTREME_CIRCLE",
}


class VIA3Reader(AnnotationReader):
    def __init__(
        self,
        file_path: Union[str, List[str]],
        images_dir: Optional[str] = None,
        image_auth: Optional[Tuple[str, str]] = None,
        attrib_name: Optional[str] = None,
        attrib_id: Union[str, int] = None,
        point_radius: int = 5,
        line_thickness: int = 2,
    ):
        if attrib_id:
            if attrib_name:
                logger.warning(
                    "Both an attribute name and id were given, considering only the id."
                )
            self.attrib_id = attrib_id
            self.attrib_name = None
        elif attrib_name:
            self.attrib_name = attrib_name
            self.attrib_id = None
        else:
            raise ValueError("Should define at least one attribute, name or id.")

        self.point_radius = point_radius
        self.line_thickness = line_thickness
        super().__init__(file_path, images_dir, image_auth)

    def _transform_annotations_data(
        self,
        annotations_data: pd.DataFrame,
        attributes_info,
        id_to_shape: Dict[int, str],
    ) -> pd.DataFrame:
        attribute_id = get_attribute_id(
            attributes_info, self.attrib_id, self.attrib_name
        )
        labels_parser = get_labels_parser(attribute_id, attributes_info)

        annotations_data["label"] = annotations_data["label"].apply(labels_parser)
        annotations_data["shape"] = annotations_data["shape_info"].apply(
            lambda shape_info: parse_via3_shape(
                shape_info, id_to_shape, self.point_radius, self.line_thickness
            )
        )

        return annotations_data


@AnnotationReader.register("via3_project")
class VIA3ProjectReader(VIA3Reader):

    def _read_data(self, path: str, image_dir: Optional[str] = None) -> pd.DataFrame:
        with open(path, "r") as infile:
            via_data = json.load(infile)

        annotations_data = pd.DataFrame.from_dict(via_data["metadata"], orient="index")

        files_info = pd.DataFrame.from_dict(via_data["file"], orient="index")
        if image_dir:
            files_info["fname"] = files_info["fname"].apply(
                lambda f: append_image_dir(f, image_dir)
            )
        file_id_to_path = files_info.set_index("fid")["fname"].to_dict()

        annotations_data["path"] = annotations_data["vid"].apply(file_id_to_path.get)

        annotations_data = annotations_data.rename(columns={"xy": "shape_info", "av": "label"})[
            ["path", "shape_info", "label"]
        ]

        all_paths = set(list(file_id_to_path.values()))

        attributes_info = via_data["attribute"]

        annotations_data = self._transform_annotations_data(annotations_data, attributes_info, ID_TO_SHAPE)

        return annotation_data_to_data(annotations_data, all_paths)

    def _transform_data_row_to_annot(self, row) -> Annotation:
        return data_row_to_annotation(row, self.image_auth)


@AnnotationReader.register("via3")
class VIA3CSVReader(VIA3Reader):

    def _read_data(self, path: str, image_dir: Optional[str] = None) -> pd.DataFrame:
        with open(path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()

        csv_header = get_header_string("CSV_HEADER", lines).split(",")
        shape_to_id = json.loads(get_header_string("SHAPE_ID", lines))
        id_to_shape = reverse_dict(shape_to_id)
        attributes_info = json.loads(get_header_string("ATTRIBUTE", lines))

        annotations_data = read_csv_lines(lines, csv_header)
        annotations_data = (
            annotations_data[["file_list", "spatial_coordinates", "metadata"]]
            .applymap(json.loads)
            .explode("file_list")
            .reset_index(drop=True)
            .set_axis(["path", "shape_info", "label"], axis=1)
        )

        if image_dir:
            annotations_data["path"] = annotations_data["path"].apply(lambda f: os.path.join(image_dir, f))

        all_paths = set(annotations_data["path"].unique().tolist())

        annotations_data = self._transform_annotations_data(annotations_data, attributes_info, id_to_shape)

        return annotation_data_to_data(annotations_data, all_paths)

    def _transform_data_row_to_annot(self, row) -> Annotation:
        return data_row_to_annotation(row, self.image_auth)


def read_csv_lines(lines: List[str], header=List[str]) -> pd.DataFrame:
    csv_lines = [line for line in lines if not line.startswith("#")]
    csv_buffer = StringIO("\n".join(csv_lines))
    data = pd.read_csv(csv_buffer, header=None, names=header)
    return data


def get_header_string(header_name: str, lines: List[str]) -> str:
    header_line = [line for line in lines if line.lstrip("# ").startswith(header_name)]

    if len(header_line) != 1:
        raise ValueError(f"Could not find {header_name} line")

    header = header_line[0].split("=")[1].strip()
    return header


def get_attribute_id(
    attributes_info: Dict[str, Any],
    attrib_id: Optional[str] = None,
    attrib_name: Optional[str] = None,
) -> str:
    if attrib_id:
        if attrib_id not in attributes_info:
            raise ValueError(
                f"The attribute id {attrib_id} was not found in the attributes info of the csv."
            )
        attribute_id = attrib_id
    else:
        attribute_id = None
        for attr_id, attr in attributes_info.items():
            if attrib_name == attr["aname"]:
                if attribute_id:
                    raise ValueError(
                        f"The attribute name {attrib_name} was defined more than once."
                    )
                attribute_id = attr_id
    return attribute_id


def get_labels_parser(
    attribute_id: str, attributes: Dict[str, Any]
) -> Callable[[str, Dict[str, str]], List[str]]:
    attribute = attributes[attribute_id]
    attribute_type = attribute["type"]
    default_id = None

    if attribute_type == 1 or attribute_type == 5:
        return lambda attributes: ["foreground"]
    elif attribute_type == 3 or attribute_type == 4:
        if "default_option_id" in attribute and len(attribute["default_option_id"]) > 0:
            default_id = attribute["default_option_id"]
        options = attribute["options"]
        return lambda attributes: get_attribute(
            attribute_id, attributes, default_id, options
        )
    elif attribute_type == 2:
        if "default_option_id" in attribute and len(attribute["default_option_id"]) > 0:
            default_id = attribute["default_option_id"]
        options = attribute["options"]
        return lambda attributes: get_attribute(
            attribute_id, attributes, default_id, options, lambda id: id.split(",")
        )
    else:
        raise ValueError(f"Attribute type {attribute_type} not supported.")


def get_attribute(
    attribute_id, attributes, default_id, options, id_processor=lambda id: [id]
):
    if attribute_id in attributes:
        ids = id_processor(attributes[attribute_id])
    elif default_id:
        ids = id_processor(default_id)
    else:
        ids = []
    return [options[id] for id in ids]
