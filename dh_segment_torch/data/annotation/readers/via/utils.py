import pandas as pd

from dh_segment_torch.data.annotation.annotation import Annotation
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations


def data_to_annotations(data, all_paths):
    annotated_files = data_to_annotated_files(data)
    annotations = []
    for path in all_paths:
        label_annots = annotated_files.get(path, None)
        annotations.append(Annotation(path, labels_annotations=label_annots))
    return annotations


def data_to_annotated_files(data):
    return (
        data.explode("label")
            .dropna()
            .groupby(["path"])
            .apply(aggregate_label_annotation)
            .to_dict()
    )


def aggregate_label_annotation(group: pd.DataFrame):
    labels_annotations = LabelsAnnotations()
    for label, shape in group[['label', 'shape']].values:
        if label not in labels_annotations:
            labels_annotations[label] = []
        labels_annotations[label].append(shape)
    return labels_annotations