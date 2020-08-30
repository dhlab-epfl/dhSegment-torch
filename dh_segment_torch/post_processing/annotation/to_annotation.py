from dh_segment_torch.data.annotation.annotation import Annotation

from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations

from dh_segment_torch.post_processing.operation import Operation


@Operation.register("to_annotation")
class ToAnnotation(Operation):
    def __init__(self):
        pass

    def apply(
        self, uri: str, labels_annotations: LabelsAnnotations, *args, **kwargs
    ) -> Annotation:
        return Annotation(uri, labels_annotations=labels_annotations)
