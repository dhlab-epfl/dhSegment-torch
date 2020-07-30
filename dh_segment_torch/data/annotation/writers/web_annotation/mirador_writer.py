import json
import uuid
from copy import deepcopy
from typing import Dict, Any, List

from torch.utils.data import DataLoader

from dh_segment_torch.data.annotation.writers.utils import AnnotationProcessorDataset, _collate_fn
from lxml import etree
from shapely import geometry
from tqdm import tqdm

from dh_segment_torch.data.annotation import Annotation
from dh_segment_torch.data.annotation.annotation_iterator import AnnotationIterator
from dh_segment_torch.data.annotation.shape import (
    Shape,
    Rectangle,
    Circle,
    Ellipse,
)
from dh_segment_torch.data.annotation.utils import iiif_url_to_manifest
from dh_segment_torch.data.annotation.writers.annotation_writer import AnnotationWriter


@AnnotationWriter.register("mirador")
class MiradorWebAnnotationWriter(AnnotationWriter):
    def __init__(
        self,
        annotation_iterator: AnnotationIterator,
        json_path: str,
        overwrite: bool = True,
        progress: bool = False,
    ):
        super().__init__(annotation_iterator, overwrite, progress)
        self.mirador_annot_dataset = AnnotationProcessorDataset(annotation_iterator, parse_annotation)
        self.json_path = json_path
        self.overwrite = overwrite
        self.progress = progress

    def write(self, num_workers: int = 0):
        annotations = {}

        mirador_annotation_dataloder = DataLoader(
            dataset=self.mirador_annot_dataset,
            batch_size=1,
            num_workers=min(num_workers, len(self.mirador_annot_dataset)),
            collate_fn=_collate_fn,
            shuffle=False,
            drop_last=False,
            pin_memory=False,
        )

        for annotation in tqdm(
            mirador_annotation_dataloder,
            total=len(mirador_annotation_dataloder),
            disable=not self.progress,
        ):
            annotations.update(annotation)
        with open(self.json_path, "w", encoding="utf-8") as outfile:
            json.dump(annotations, outfile)


def parse_annotation(annotation: Annotation) -> Dict[str, Any]:
    if not annotation.is_iiif:
        raise ValueError("Currently only iiif annotations are supported.")
    all_annots = []

    for labels, shapes in annotation.labels_annotations.items():
        if isinstance(labels, str):
            labels = [labels]
        label_web_annotation = deepcopy(web_annotation_base)
        label_web_annotation["motivation"] = ["oa:tagging"] * len(labels)
        for label in labels:
            label_web_annotation["resource"].append({"@type": "oa:Tag", "chars": label})
        label_web_annotation["on"][0]["full"] = annotation.uri
        label_web_annotation["on"][0]['selector']["within"]["@id"] = iiif_url_to_manifest(
            annotation.uri
        )

        for shape in shapes:
            geoms = shape.geometry(annotation.image_size)
            if not isinstance(geoms, geometry.MultiPolygon) and not isinstance(
                geoms, geometry.MultiLineString
            ):
                geoms = [geoms]
            for geom in geoms:
                web_annotation = deepcopy(label_web_annotation)
                web_annotation["@id"] = str(uuid.uuid4())
                svg = geometry_to_svg(geom, shape)
                x, y, xmax, ymax = geom.bounds
                w = xmax - x
                h = ymax - y
                web_annotation["on"][0]["selector"]["default"][
                    "value"
                ] = f"xywh={x},{y},{w},{h}"
                web_annotation["on"][0]["selector"]["item"]["value"] = svg
                all_annots.append(web_annotation)
    return {annotation.uri : all_annots}


def geometry_to_svg(geom: geometry.base.BaseGeometry, base_shape: Shape):
    if isinstance(geom, geometry.Point):
        d = f"M{geom.x},{geom.y}"
    elif isinstance(geom, geometry.LineString):
        points = (
            etree.fromstring(geom._repr_svg_())
            .find(".//{http://www.w3.org/2000/svg}polyline")
            .attrib["points"]
        ).split()
        d = ["M" + points[0]]
        for point in points[1:]:
            d.append("L" + point)
        d = " ".join(d)
    else:
        d = (
            etree.fromstring(geom._repr_svg_())
            .find(".//{http://www.w3.org/2000/svg}path")
            .attrib["d"]
        )

    id_prefix = "rough_path"
    if isinstance(base_shape, Circle) or isinstance(base_shape, Ellipse):
        id_prefix = "ellipse"
    elif isinstance(base_shape, Rectangle):
        id_prefix = "rectangle"
    return (
        "<svg xmlns='http://www.w3.org/2000/svg'>"
        '<path xmlns="http://www.w3.org/2000/svg"'
        f'd="{d}"'
        'data-paper-data="{&quot;defaultStrokeValue&quot;:1,'
        "&quot;editStrokeValue&quot;:5,"
        "&quot;currentStrokeValue&quot;:1,"
        "&quot;editable&quot;:true,"
        "&quot;deleteIcon&quot;:null,&quot;annotation"
        '&quot;:null}"'
        f'id="{id_prefix}_{str(uuid.uuid4())}"'
        'fill-opacity="0"'
        'fill="#00bfff"'
        'fill-rule="nonzero"'
        'stroke="#00bfff" '
        'stroke-width="0.90909"'
        'stroke-linecap="butt"'
        'stroke-linejoin="miter"'
        'stroke-miterlimit="10"'
        'stroke-dasharray=""'
        'stroke-dashoffset="0"'
        'font-family="none"'
        'font-weight="none"'
        'font-size="none"'
        'text-anchor="none"'
        'style="mix-blend-mode: normal"/>'
        "</svg>"
    )


web_annotation_base = {
    "@context": "http://iiif.io/api/presentation/2/context.json",
    "@type": "oa:Annotation",
    "motivation": [],
    "resource": [],
    "on": [
        {
            "@type": "oa:SpecificResource",
            "full": "",
            "selector": {
                "@type": "oa:Choice",
                "default": {"@type": "oa:FragmentSelector", "value": "xywh="},
                "item": {"@type": "oa:SvgSelector", "value": ""},
                "within": {"@id": "", "@type": "sc:Manifest"},
            },
        }
    ],
    "@id": "",
}
