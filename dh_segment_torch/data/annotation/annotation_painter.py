import logging
from typing import List

import numpy as np

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.image_size import ImageSize
from dh_segment_torch.data.annotation.labels_annotations import LabelsAnnotations
from dh_segment_torch.data.color_labels import ColorLabels

logger = logging.getLogger(__name__)


class AnnotationPainter(Registrable):
    default_implementation = "default"

    def __init__(
        self, color_labels: ColorLabels, disallowed_overlaps: List[List[str]] = None,
    ):
        if not color_labels.labels:
            raise ValueError("Expected to have label names")
        self.color_labels = color_labels
        if disallowed_overlaps:
            self.disallowed_overlaps = self._get_disallowed_indices(disallowed_overlaps)
        else:
            self.disallowed_overlaps = []
        self.used_colors = {(0, 0, 0)}

    def paint(
        self,
        image_size: ImageSize,
        labels_annotations: LabelsAnnotations,
        track_colors: bool = True,
    ) -> np.array:
        valid_labels = []
        for label in labels_annotations.keys():
            if label in self.color_labels.labels:
                valid_labels.append(label)
            else:
                logger.warning(
                    f"Found label {label} in annotations but it is not defined in the colors."
                )
        valid_labels = sorted(
            valid_labels, key=lambda x: self.color_labels.labels.index(x)
        )

        canvas = np.zeros((image_size.height, image_size.width, 3), dtype=np.uint8)

        if self.color_labels.multilabel:
            return self._paint_multilabel(
                canvas, valid_labels, labels_annotations, track_colors
            )
        else:
            return self._paint_multiclass(
                canvas, valid_labels, labels_annotations, track_colors
            )

    def _paint_multiclass(
        self,
        canvas: np.array,
        valid_labels: List[str],
        labels_annotations: LabelsAnnotations,
        track_colors: bool = True,
    ):
        label_to_color = dict(zip(self.color_labels.labels, self.color_labels.colors))
        for label in valid_labels:
            mask = labels_annotations.label_mask(
                label, ImageSize(canvas.shape[0], canvas.shape[1])
            )
            color = label_to_color[label]
            canvas[mask] = color
            if track_colors:
                self.used_colors.add(color)
        return canvas

    def _paint_multilabel(
        self,
        canvas: np.array,
        valid_labels: List[str],
        labels_annotations: LabelsAnnotations,
        track_colors: bool = True,
    ):
        num_clases = self.color_labels.num_classes
        one_hot = np.zeros((canvas.shape[0], canvas.shape[1], num_clases))
        label_to_index = {
            label: index for index, label in enumerate(self.color_labels.labels)
        }
        for label in valid_labels:
            mask = labels_annotations.label_mask(
                label, ImageSize(canvas.shape[0], canvas.shape[1])
            )
            label_index = label_to_index[label]
            one_hot[mask, label_index] = 1
        one_hot = self._remove_disallowed(one_hot)
        indices = self._one_hot_to_indices(one_hot)
        canvas[:, :, :] = np.array(self.color_labels.colors)[indices]

        if track_colors:
            unique_indices = np.unique(indices)
            used_colors = np.array(self.color_labels.colors)[unique_indices]
            for color in used_colors:
                self.used_colors.add(tuple(color))

        return canvas

    def _remove_disallowed(self, one_hot: np.array):

        for indices in self.disallowed_overlaps:
            one_hot_indices = np.tile(
                np.arange(1, len(indices) + 1), [one_hot.shape[0], one_hot.shape[1], 1]
            ).astype(np.int32)

            first_index = one_hot[:, :, indices].argmax(axis=-1) + 1
            first_index[one_hot[:, :, indices].sum(axis=-1) == 0] = 0

            one_hot[:, :, indices] = (
                one_hot_indices == first_index[:, :, None]
            ).astype(one_hot.dtype)
            del one_hot_indices
        return one_hot

    def _get_disallowed_indices(self, disallowed_overlaps_labels: List[List[str]]):
        disallowed_overlaps = []
        for overlap in disallowed_overlaps_labels:
            indices = []
            for label in overlap:
                try:
                    indices.append(self.color_labels.labels.index(label))
                except ValueError:
                    logger.warning(
                        f"Found label {label} in overlaps but it is not defined in the labels."
                    )
            disallowed_overlaps.append(
                indices
            )  # We do not sort the order of the indices, meaning that the order is important (the first item will have priorities over the rest)
        return disallowed_overlaps

    def _one_hot_to_indices(self, one_hot: np.array):
        one_hot_encoding = np.array(self.color_labels.one_hot_encoding)[None, None]
        return np.abs(one_hot[:, :, None] - one_hot_encoding).sum(axis=3).argmin(axis=2)


AnnotationPainter.register("default")(AnnotationPainter)
