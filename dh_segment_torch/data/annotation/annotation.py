from typing import Optional

import cv2

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from dh_segment_torch.data.annotation.utils import ImageSize


class Annotation(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        uri: str,
        image_id: Optional[str] = None,
        image_size: Optional[ImageSize] = None,
        labels_annotations: LabelsAnnotations = None,
        normalize_shapes: bool = True
    ):
        self.uri = uri
        if image_id:
            self.image_id = image_id
        else:
            self.image_id = self.get_image_id()

        if image_size:
            self.image_size = image_size
        else:
            self.image_size = self.get_image_size()

        if labels_annotations:
            self.labels_annotations = labels_annotations
        else:
            self.labels_annotations = LabelsAnnotations()

        if normalize_shapes:
            self.labels_annotations.normalize_shapes(self.image_size)

    def get_image_id(self) -> str:
        return self.uri.split('/\\')[-1].split('.')[-2]

    def get_image_size(self) -> ImageSize:
        image = load_image(self.uri)
        return ImageSize(height=image.shape[0], width=image.shape[1])


def load_image(uri: str):
    image = cv2.imread(uri)
    return image

Annotation.register("default")(Annotation)
