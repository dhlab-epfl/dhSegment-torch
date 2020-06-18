import os
from collections.abc import Sized
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union

import numpy as np
import logging

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.utils import parse_and_validate_color, n_colors

logger = logging.getLogger(__name__)


class ColorLabels(Registrable):
    default_implementation = "labels_list"

    def __init__(
        self,
        colors: List[Tuple[int, int, int]],
        one_hot_encoding: Optional[List[List[int]]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.colors = colors
        self.one_hot_encoding = one_hot_encoding
        self.labels = labels
        self.log_labels = None

        if labels:
            if one_hot_encoding:
                new_names = []
                for line in np.array(one_hot_encoding).astype(bool):
                    new_names.append("+".join(np.array(labels)[line]))
                new_names[0] = "background"
                self.log_labels = new_names
                assert len(self.log_labels) == len(colors)
            else:
                self.log_labels = labels
                assert len(self.log_labels) == len(colors)
            if self.num_classes != len(self.labels):
                raise ValueError(
                    f"Cannot have a different number of classes, {self.num_classes}, and labels, {len(self.labels)}"
                )

    @property
    def multilabel(self):
        return self.one_hot_encoding is not None

    @property
    def num_classes(self):
        if self.one_hot_encoding:
            return len(self.one_hot_encoding[0])
        else:
            return len(self.colors)

    @classmethod
    def from_labels_text_file(
        cls, label_text_file: Union[str, Path], labels: Optional[List[str]] = None
    ):
        label_text_file = str(label_text_file)
        if not os.path.exists(label_text_file):
            raise FileNotFoundError(label_text_file)
        labels_classes = np.loadtxt(label_text_file).astype(np.float32)

        if labels_classes.shape[1] < 3:
            raise ValueError(
                "Text label file did not contain enough information to be colors."
            )

        colors = [parse_and_validate_color(color) for color in labels_classes[:, :3]]

        if labels_classes.shape[1] == 3:
            return cls(colors)
        else:
            one_hot_encoding = [
                parse_validate_one_hot(one_hot) for one_hot in labels_classes[:, 3:]
            ]
            return cls(colors, one_hot_encoding, labels)

    @classmethod
    def from_list_of_color_labels(
        cls, color_labels: List[Dict[str, Any]], labels: Optional[List[str]] = None
    ):
        colors: List[Tuple[int, int, int]] = []
        one_hot_encoding: Optional[List[List[int]]] = None
        labels: Optional[List[str]] = labels

        has_one_hot = None
        has_labels = None
        one_hot_size = None

        for label in color_labels:
            if "color" not in label:
                raise ValueError("Need at least a color to define a label.")
            color = parse_and_validate_color(label["color"])
            colors.append(color)

            if "one_hot" in label:
                if has_one_hot is None:
                    has_one_hot = True
                    one_hot_encoding = []
                one_hot = parse_validate_one_hot(label["one_hot"])

                if one_hot_size is None:
                    one_hot_size = len(one_hot)
                if not has_one_hot:
                    raise ValueError("Some labels have one hot defined, others not.")
                if one_hot_size != len(one_hot):
                    raise ValueError("Some labels have different one hot sizes.")

                one_hot_encoding.append(one_hot)
            else:
                has_one_hot = False

            if "label" in label:
                if has_labels is None:
                    has_labels = True
                    labels = []
                if not has_labels:
                    raise ValueError("Some labels have a name, others not.")
                labels.append(label["label"])
            else:
                has_labels = False

        return cls(colors, one_hot_encoding, labels)

    @classmethod
    def from_colors(
        cls,
        colors: List[Union[str, Tuple[int, int, int]]],
        labels: Optional[List[str]] = None,
    ):
        colors = [parse_and_validate_color(color) for color in colors]
        return cls(colors, labels=labels)

    @classmethod
    def from_colors_multilabel(
        cls,
        colors: List[Union[str, Tuple[int, int, int]]],
        labels: Optional[List[str]] = None,
    ):
        colors = [parse_and_validate_color(color) for color in colors]
        one_hot_encoding, colors = all_one_hot_and_colors(colors)
        return cls(colors, one_hot_encoding, labels)

    @classmethod
    def from_labels(cls, labels: List[str]):
        num_classes = len(labels)
        colors = n_colors(num_classes)
        return cls(colors, labels=labels)

    @classmethod
    def from_labels_multilabel(cls, labels: List[str]):
        num_classes = len(labels)

        num_tries_left = 10
        while num_tries_left:
            num_tries_left -= 1
            base_colors = n_colors(num_classes)
            one_hot_encoding, colors = all_one_hot_and_colors(base_colors)
            if len(colors) == len(set(colors)):
                break
        else:
            logger.warning(f"Could not find a color combinatation for {num_classes}."
                           "Falling back on one color per one hot encoding.")
            one_hot_encoding = get_all_one_hots(num_classes).tolist()
            colors = n_colors(len(one_hot_encoding))
        return cls(colors, one_hot_encoding, labels)


ColorLabels.register("labels_list", "from_list_of_color_labels")(ColorLabels)
ColorLabels.register("txt", "from_labels_text_file")(ColorLabels)
ColorLabels.register("colors", "from_colors")(ColorLabels)
ColorLabels.register("colors_multilabel", "from_colors_multilabel")(ColorLabels)
ColorLabels.register("labels", "from_labels")(ColorLabels)
ColorLabels.register("labels_multilabel", "from_labels_multilabel")(ColorLabels)


def parse_validate_one_hot(one_hot) -> List[int]:
    if not isinstance(one_hot, str) and not isinstance(one_hot, Sized):
        raise ValueError("One hot needs to be defined either by a sequence or a string")
    if isinstance(one_hot, str):
        one_hot = [x for x in one_hot]
    one_hot = np.array(one_hot).astype(np.int32)
    if len(set(np.unique(one_hot).tolist()).difference({0, 1})) > 0:
        raise ValueError("Found not 0 and 1 ")
    return [x for x in one_hot]


def all_one_hot_and_colors(
    colors: List[Tuple[int, int, int]]
) -> Tuple[List[List[int]], List[Tuple[int, int, int]]]:
    num_classes = len(colors)
    colors = np.array(colors)
    one_hots = get_all_one_hots(num_classes)

    final_colors = []
    for one_hot in one_hots:
        if one_hot.sum() == 0:
            color = (0, 0, 0)
        else:
            color = tuple(
                np.round(np.mean(colors[np.where(one_hot)[0]], axis=0))
                .astype(int)
                .tolist()
            )
        final_colors.append(color)
    return one_hots.tolist(), final_colors


def get_all_one_hots(num_classes: int) -> np.array:
    return (
        ((np.arange(0, 2 ** num_classes)[:, None] & (1 << np.arange(num_classes))) > 0)
        .astype(int)
    )
