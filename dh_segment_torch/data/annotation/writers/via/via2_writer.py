import json
import logging
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dh_segment_torch.data.annotation import shape as sh
from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator
from dh_segment_torch.data.annotation.writers.annotation_writer import AnnotationWriter
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from dh_segment_torch.data.annotation.utils import is_url, extract_image_name_with_ext
from dh_segment_torch.data.annotation.writers.via.utils import (
    DEFAULT_VIA2_DICT,
)
from dh_segment_torch.data.annotation.writers.utils import AnnotationProcessorDataset, _collate_fn

logger = logging.getLogger(__name__)


class VIA2Writer(AnnotationWriter):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        attrib_name: str,
        json_path: str,
        base_json: Optional[Dict[str, Any]] = None,
        multilabel: bool = None,
        overwrite: bool = True,
        progress: bool = False,
    ):
        super().__init__(annotation_iterator, overwrite, progress)
        self.via2_annot_dataset = AnnotationProcessorDataset(
            self.annotation_iterator, process_annotation
        )
        self.attrib_name = attrib_name
        self.json_path = json_path
        if base_json is None:
            base_json = DEFAULT_VIA2_DICT
        self.base_json = base_json
        self.multilabel = multilabel

    def write(self, num_workers: int = 0):
        via2_annotations = {}

        via2_annotation_dataloder = DataLoader(
            dataset=self.via2_annot_dataset,
            batch_size=1,
            num_workers=min(num_workers, len(self.via2_annot_dataset)),
            collate_fn=_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        for via2_annotation in tqdm(
            via2_annotation_dataloder,
            total=len(via2_annotation_dataloder),
            disable=not self.progress,
        ):
            via2_annotations.update(via2_annotation)

        via2_annotations, all_labels, multilabel = self.compile_shape_attributes(
            via2_annotations
        )
        via2_project = deepcopy(self.base_json)
        via2_project.update(
            {
                "_via_img_metadata": via2_annotations,
                "_via_image_id_list": list(via2_annotations.keys()),
            }
        )

        if "_via_attributes" not in via2_project:
            via2_project.update(
                {
                    "_via_attributes": {
                        "region": {
                            self.attrib_name: {
                                "type": "checkbox" if multilabel else "radio",
                                "description": "",
                                "options": {label: "" for label in all_labels},
                            }
                        },
                        "file": {},
                    },
                }
            )
        now = datetime.now()
        via2_project["_via_settings"]["project"]["name"] = now.strftime(
            "via_project_%d%b%Y_%Hh%M"
        )

        with open(self.json_path, "w", encoding="utf-8") as outfile:
            json.dump(via2_project, outfile)

    def compile_shape_attributes(
        self, via_annotation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[str], bool]:
        region_dicts = []
        multilabel = self.multilabel
        all_labels = set()

        for annot in via_annotation.values():
            for region in annot["regions"]:
                region_dicts.append(region)
                label = region["region_attributes"]
                if isinstance(label, tuple):
                    all_labels.update(label)
                    if multilabel is None:
                        multilabel = True
                    if not multilabel:
                        logger.warning(
                            "Mix between multiclass and multilabel annotations"
                            ", making everything multilabel."
                        )
                        multilabel = True
                else:
                    all_labels.add(label)
                    if multilabel is None:
                        multilabel = False
        if multilabel:
            for region_dict in region_dicts:
                labels = region_dict["region_attributes"]
                if isinstance(labels, str):
                    labels = [labels]
                region_dict["region_attributes"] = {
                    self.attrib_name: {label: True for label in labels}
                }
        else:
            for region_dict in region_dicts:
                label = region_dict["region_attributes"]
                region_dict["region_attributes"] = {self.attrib_name: label}
        return via_annotation, sorted(all_labels), multilabel

    @classmethod
    def from_via_project(
        cls,
        via_project: str,
        annotation_iterator: AnnotationIterator,
        attrib_name: str,
        json_path: str,
        overwrite: bool = True,
        progress: bool = False,
    ):
        with open(via_project, "r", encoding="utf-8") as infile:
            via_project = json.load(infile)
        if attrib_name not in via_project["region_attributes"]:
            raise ValueError(
                f"Expected via project to have an attribute named {attrib_name}"
                f", only got {list(via_project['region_attributes'].keys())}"
            )
        region_type = via_project["region_atttributes"][attrib_name]["type"]
        if region_type in {"radio", "dropdown"}:
            multilabel = False
        elif region_type == "checkbox":
            multilabel = True
        else:
            raise ValueError(
                f"The type, {region_type}, "
                f"of the attribute {attrib_name} is not supported."
            )
        return cls(
            annotation_iterator,
            attrib_name,
            json_path,
            base_json=via_project,
            multilabel=multilabel,
            overwrite=overwrite,
            progress=progress,
        )


AnnotationWriter.register("via2")(VIA2Writer)
AnnotationWriter.register("via2_from_project", "from_via_project")(VIA2Writer)


def process_annotation(annotation: Annotation):
    basename = (
        annotation.uri
        if is_url(annotation.uri)
        else extract_image_name_with_ext(annotation.uri)
    )
    return {
        f"{basename}-1": {
            "filename": basename,
            "size": -1,
            "regions": labels_annotation_to_via_shapes(
                annotation.labels_annotations.groupby_shape(), annotation.image_size
            ),
            "file_attributes": {},
        }
    }


def labels_annotation_to_via_shapes(
    labels_annotation: LabelsAnnotations, image_size: ImageSize
):
    via_shapes = []
    for label, shapes in labels_annotation.items():
        for shape in shapes:
            for s in shape_to_via_shapes(shape, image_size):
                via_shapes.append({"region_attributes": label, "shape_attributes": s})
    return via_shapes


def shape_to_via_shapes(shape: sh.Shape, image_size: ImageSize) -> List[Dict[str, Any]]:
    expanded_coords = shape.expanded_coords(image_size)
    if isinstance(shape, sh.Point):
        (cx, cy), _ = expanded_coords
        return [{"name": "point", "cx": cx, "cy": cy}]
    elif isinstance(shape, sh.Circle):
        (cx, cy), radius = expanded_coords
        return [{"name": "circle", "cx": cx, "cy": cy, "r": radius}]
    elif isinstance(shape, sh.LineString) or isinstance(shape, sh.Line):
        all_points_x = [coord[0] for coord in expanded_coords]
        all_points_y = [coord[1] for coord in expanded_coords]
        return [
            {
                "name": "polyline",
                "all_points_x": all_points_x,
                "all_points_y": all_points_y,
            }
        ]
    elif isinstance(shape, sh.Ellipse):
        (cx, cy), (rx, ry), angle = expanded_coords
        return [
            {"name": "ellipse", "cx": cx, "cy": cy, "rx": rx, "ry": ry, "theta": angle}
        ]
    elif isinstance(shape, sh.Rectangle):
        (c1x, c1y), (c2x, c2y) = expanded_coords
        x = min(c1x, c2x)
        width = max(c1x, c2x) - x
        y = min(c1y, c2y)
        height = max(c1y, c2y) - y

        return [{"name": "rect", "x": x, "y": y, "width": width, "height": height}]
    elif isinstance(shape, sh.Polygon) or isinstance(shape, sh.MultiPolygon):
        shapes = []
        for poly in expanded_coords:
            exterior, interiors = poly
            if len(interiors) > 0:
                logger.warning(
                    "Found interior of polygons, "
                    "via2 does not support it. Ignoring it."
                )

            all_points_x = [coord[0] for coord in exterior]
            all_points_y = [coord[1] for coord in exterior]
            shapes.append(
                {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y,
                }
            )
        return shapes
    else:
        raise ValueError(f"The shape {shape} is not supported")
