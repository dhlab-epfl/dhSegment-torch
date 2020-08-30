from typing import List, Tuple

from dh_segment_torch.data.annotation import Shape, LabelsAnnotations, ImageSize
from dh_segment_torch.post_processing.operation import Operation


@Operation.register("to_labels_annotations")
class ToLabelsAnnotations(Operation):
    def __init__(self):
        pass

    def apply(
        self, labelled_shapes: List[Tuple[str, Shape]], *args, **kwargs
    ) -> LabelsAnnotations:
        labels_annotations = LabelsAnnotations()
        for label, shape in labelled_shapes:
            if label not in labels_annotations:
                labels_annotations[label] = []
            labels_annotations[label].append(shape)
        return labels_annotations


@Operation.register("normalize_labels_annotations")
class NormalizeLabelsAnnotations(Operation):
    def __init__(self):
        pass

    def apply(
        self,
        image_size: ImageSize,
        labels_annotations: LabelsAnnotations,
        *args,
        **kwargs
    ) -> LabelsAnnotations:
        labels_annotations.normalize_shapes(image_size)
        return labels_annotations
