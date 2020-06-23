import json
import logging
import os
import string
import uuid
from copy import deepcopy
from datetime import datetime
import random
from typing import Dict, Any, List, Tuple, Optional

from torch.utils.data import DataLoader
from tqdm import tqdm

from dh_segment_torch.data.annotation import shape as sh
from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator
from dh_segment_torch.data.annotation.annotation_writer import AnnotationWriter
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from dh_segment_torch.data.annotation.utils import (
    reverse_dict,
    extract_image_filename,
    extract_image_name_with_ext,
    is_url,
)
from dh_segment_torch.data.annotation.writers.via.utils import (
    AnnotationProcessorDataset,
    DEFAULT_VIA3_DICT,
)

logger = logging.getLogger(__name__)


class VIA3Writer(AnnotationWriter):
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
        self.via3_annot_dataset = AnnotationProcessorDataset(
            self.annotation_iterator, process_annotation
        )
        self.attrib_name = attrib_name
        self.json_path = json_path
        if base_json is None:
            base_json = DEFAULT_VIA3_DICT
        self.base_json = base_json
        self.multilabel = multilabel

    def write(self, num_workers: int = 0):
        via3_annotations = {}

        via3_annotation_dataloder = DataLoader(
            dataset=self.via3_annot_dataset,
            batch_size=1,
            num_workers=min(num_workers, len(self.via3_annot_dataset)),
            collate_fn=_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        for via3_annotation in tqdm(
            via3_annotation_dataloder,
            total=len(via3_annotation_dataloder),
            disable=not self.progress,
        ):
            via3_annotations.update(via3_annotation)

        (
            updated_json,
            basename_to_image_id,
            attribute_id,
            label_to_option_id,
            multilabel,
        ) = self.gather_infos(via3_annotations)

        for via3_annotation in via3_annotations.values():
            label = via3_annotation["av"]
            via3_annotation["vid"] = basename_to_image_id[via3_annotation["vid"]]
            via3_annotation["av"] = {
                attribute_id: ",".join([label_to_option_id[lbl] for lbl in label])
                if isinstance(label, tuple)
                else label_to_option_id[label]
            }

        updated_json["metadata"] = via3_annotations

        now = datetime.now()
        updated_json["project"]["created"] = int(round(now.timestamp()))
        updated_json["project"]["vid_list"] = list(updated_json["view"].keys())

        with open(self.json_path, "w", encoding="utf-8") as outfile:
            json.dump(updated_json, outfile)

    def gather_infos(
        self, via_annotation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, str], str, Dict[str, str], bool]:
        all_labels = set()
        all_basenames = set()
        multilabel = self.multilabel

        for annotation in via_annotation.values():
            label = annotation["av"]

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
                    annotation["av"] = annotation["av"]

            all_basenames.add(annotation["vid"])

        updated_json = deepcopy(self.base_json)

        if "attribute" in updated_json:
            attributes = self.base_json["attribute"]
            attribute_id, attribute = find_attrib_by_name(self.attrib_name, attributes)
            option_id_to_label = attribute["options"]
        else:
            attribute_id = "1"
            option_id_to_label = {
                str(idx): label for idx, label in enumerate(sorted(all_labels))
            }
            updated_json["attribute"] = {
                "1": {
                    "aname": self.attrib_name,
                    "anchor_id": "FILE1_Z0_XY1",
                    "type": 2 if multilabel else 3,
                    "desc": "",
                    "options": option_id_to_label,
                    "default_option_id": "",
                }
            }
        label_to_option_id = reverse_dict(option_id_to_label)

        basename_to_image_id = reverse_dict(
            {
                str(idx + 1): basename
                for idx, basename in enumerate(sorted(all_basenames))
            }
        )

        updated_json["file"] = {
            image_id: {
                "fid": image_id,
                "fname": extract_image_name_with_ext(uri),
                "type": 2,
                "loc": 2 if is_url(uri) else 3,
                "src": uri if is_url(uri) else extract_image_name_with_ext(uri),
            }
            for uri, image_id in basename_to_image_id.items()
        }

        updated_json["view"] = {
            image_id: {"fid_list": [image_id]}
            for image_id in basename_to_image_id.values()
        }

        return (
            updated_json,
            basename_to_image_id,
            attribute_id,
            label_to_option_id,
            multilabel,
        )

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
        attributes = via_project["attribute"]
        attribute_id, attribute = find_attrib_by_name(attrib_name, attributes)
        attribute_type = attribute["type"]
        if attribute_type in {3, 4}:
            multilabel = False
        elif attribute_type == 2:
            multilabel = True
        else:
            raise ValueError(
                f"The type, {attribute_type}, "
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


AnnotationWriter.register("via3")(VIA3Writer)
AnnotationWriter.register("via3_from_project", "from_via_project")(VIA3Writer)


def process_annotation(annotation: Annotation):
    return {
        f"1_{get_id()}": {
            "vid": annotation.uri,
            "flg": 0,
            "z": [],
            "xy": shape["xy"],
            "av": shape["av"],
        }
        for shape in labels_annotation_to_via_shapes(
            annotation.labels_annotations.groupby_shape(), annotation.image_size
        )
    }


def labels_annotation_to_via_shapes(
    labels_annotation: LabelsAnnotations, image_size: ImageSize
):
    via_shapes = []
    for label, shapes in labels_annotation.items():
        for shape in shapes:
            for s in shape_to_via_shapes(shape, image_size):
                via_shapes.append({"av": label, "xy": s})
    return via_shapes


def shape_to_via_shapes(shape: sh.Shape, image_size: ImageSize) -> List[List[int]]:
    expanded_coords = shape.expanded_coords(image_size)
    if isinstance(shape, sh.Point):
        (cx, cy), _ = expanded_coords
        return [[1, cx, cy]]
    elif isinstance(shape, sh.Rectangle):
        (c1x, c1y), (c2x, c2y) = expanded_coords
        xmin = min(c1x, c2x)
        xmax = max(c1x, c2x)
        ymin = min(c1y, c2y)
        ymax = max(c1y, c2y)
        return [[2, xmin, ymin, xmax - xmin, ymax - ymin]]
    elif isinstance(shape, sh.Circle):
        (cx, cy), radius = expanded_coords
        return [[3, cx, cy, radius]]
    elif isinstance(shape, sh.Ellipse):
        (cx, cy), (rx, ry), angle = expanded_coords
        return [[4, cx, cy, rx, ry]]
    elif isinstance(shape, sh.Line):
        flattened_coords = [coord for coords in expanded_coords for coord in coords]
        return [[5] + flattened_coords]
    elif isinstance(shape, sh.LineString):
        flattened_coords = [coord for coords in expanded_coords for coord in coords]
        return [[6] + flattened_coords]
    elif isinstance(shape, sh.Polygon) or isinstance(shape, sh.MultiPolygon):
        shapes = []
        for poly in expanded_coords:
            exterior, interiors = poly
            if len(interiors) > 0:
                logger.warning(
                    "Found interior of polygons, "
                    "via3 does not support it. Ignoring it."
                )
            flattened_coords = [coord for coords in exterior for coord in coords]
            shapes.append([7] + flattened_coords)
        return shapes
    else:
        raise ValueError(f"The shape {shape} is not supported")


def _collate_fn(examples):
    res = {}
    for example in examples:
        res.update(example)
    return res


def get_id(size=8, chars=string.ascii_letters + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def find_attrib_by_name(attrib_name, attributes):
    attribute = None
    attribute_id = None
    for attrib_id, attrib in attributes.items():
        if attrib["aname"] == attrib_name:
            if attribute is None:
                attribute = attrib
                attribute_id = attrib_id
            else:
                raise ValueError(f"Several attributes with the name {attrib_name}.")
    if attribute is None or attribute_id is None:
        raise ValueError(f"No attribute matching {attrib_name}")
    return attribute_id, attribute
