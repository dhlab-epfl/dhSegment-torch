from typing import Optional, Tuple, Dict, Any

import pandas as pd

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations


def data_to_annotations(data, all_paths, image_auth: Optional[Tuple[str, str]] = None):
    annotated_files = data_to_annotated_files(data)
    annotations = []
    for path in all_paths:
        label_annots = annotated_files.get(path, None)
        annotations.append(
            Annotation(path, labels_annotations=label_annots, auth=image_auth)
        )
    return annotations


def data_row_to_annotation(
    row, image_auth: Optional[Tuple[str, str]] = None
) -> Annotation:
    return Annotation(
        row["path"], labels_annotations=row["labels_annotations"], auth=image_auth
    )


def annotation_data_to_data(annotation_data, all_paths) -> pd.DataFrame:
    annotated_files = data_to_annotated_files(annotation_data)

    data = pd.DataFrame(all_paths, columns=["path"])
    data["labels_annotations"] = data["path"].apply(
        lambda p: annotated_files.get(p, None)
    )
    return data


def data_to_annotated_files(data) -> Dict[str, Any]:
    return (
        data.explode("label")
        .dropna()
        .groupby(["path"])
        .apply(aggregate_label_annotation)
        .to_dict()
    )


def aggregate_label_annotation(group: pd.DataFrame) -> LabelsAnnotations:
    labels_annotations = LabelsAnnotations()
    for label, shape in group[["label", "shape"]].values:
        if label not in labels_annotations:
            labels_annotations[label] = []
        labels_annotations[label].append(shape)
    return labels_annotations
