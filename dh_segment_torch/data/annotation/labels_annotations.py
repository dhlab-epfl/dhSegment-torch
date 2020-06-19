from collections.abc import MutableMapping
from typing import Dict, List

import numpy as np

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.shape import Shape
from dh_segment_torch.data.annotation.image_size import ImageSize


class LabelsAnnotations(Registrable, MutableMapping):
    default_implementation = "default"

    def __init__(self, *args, **kwargs):
        self.annots: Dict[str, List[Shape]] = {}
        self.update(dict(*args, **kwargs))

    def __getitem__(self, label: str):
        return self.annots[label]

    def __setitem__(self, label: str, shapes: List[Shape]):
        self.annots[label] = shapes

    def __delitem__(self, label: str):
        del self.annots[label]

    def __iter__(self):
        return iter(self.annots)

    def __len__(self):
        return len(self.annots)

    def __repr__(self):
        return self.annots.__repr__()

    def __str__(self):
        return self.annots.__str__()

    def label_mask(self, label: str, mask_size: ImageSize):
        mask = np.zeros((mask_size.height, mask_size.width), dtype=bool)
        for shape in self[label]:
            mask |= shape.mask(mask_size)
        return mask

    def normalize_shapes(self, image_size: ImageSize):
        for shapes in self.values():
            for shape in shapes:
                shape.normalize_coords(image_size)


LabelsAnnotations.register("default")(LabelsAnnotations)
