from collections.abc import MutableMapping
from itertools import groupby
from typing import Dict, List, Union, Tuple

import numpy as np
from shapely import geometry

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.shape import Shape


class LabelsAnnotations(Registrable, MutableMapping):
    default_implementation = "default"

    def __init__(self, *args, **kwargs):
        self.annots: Dict[Union[Tuple[str, ...], str], List[Shape]] = {}
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

    def label_mask(
        self, label: Union[Tuple[str, ...], str], mask_size: ImageSize
    ) -> np.array:
        mask = np.zeros((mask_size.height, mask_size.width), dtype=bool)
        for shape in self[label]:
            mask |= shape.mask(mask_size)
        return mask

    def label_geometries(
        self, label: Union[Tuple[str, ...], str], mask_size: ImageSize
    ) -> List[geometry.base.BaseGeometry]:
        geometries = []
        for shape in self[label]:
            geometries.append(shape.geometry(mask_size))
        return geometries

    def normalize_shapes(self, image_size: ImageSize):
        for shapes in self.values():
            for shape in shapes:
                shape.normalize_coords(image_size)

    def groupby_shape(self):
        label_shape = [
            (shape, label) for label, shapes in self.items() for shape in shapes
        ]
        new_labels_annotations = {}
        for shape, group in groupby(
            sorted(label_shape, key=lambda x: id(x[0])), lambda x: x[0]
        ):
            labels = []
            for _, label in group:
                labels.append(label)
            if len(labels) == 1:
                labels = labels[0]
            else:
                labels = tuple(labels)
            if labels not in new_labels_annotations:
                new_labels_annotations[labels] = []
            new_labels_annotations[labels].append(shape)
        labels_annotations = LabelsAnnotations()
        labels_annotations.update(new_labels_annotations)
        return labels_annotations


LabelsAnnotations.register("default")(LabelsAnnotations)
