from typing import List

import numpy as np

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.shape import Shape
from dh_segment_torch.data.annotation.image_size import ImageSize


class LabelAnnotation(Registrable):
    default_implementation = "default"

    def __init__(self, label: str, shapes: List[Shape] = None):
        self.label = label
        self.shapes = shapes if shapes else []

    def __str__(self):
        return f"LabelAnnotation({self.label})"

    def __repr__(self):
        return f"LabelAnnotation({self.label}, {self.shapes})"

    def mask(self, mask_size: ImageSize) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width), dtype=bool)
        for shape in self.shapes:
            mask |= shape.mask(mask_size)
        return mask

    def normalize_shapes(self, image_size: ImageSize):
        for shape in self.shapes:
            shape.normalize_coords(image_size)


LabelAnnotation.register("default")(LabelAnnotation)
