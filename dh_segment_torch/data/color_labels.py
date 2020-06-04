import os
from collections.abc import Sized
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any, Union

import numpy as np

from dh_segment_torch.config.registrable import Registrable


class ColorLabels(Registrable):
    default_implementation = "labels_list"
    def __init__(
        self,
        colors_labels: List[Tuple[int, int, int]],
        one_hot_labels: Optional[List[List[int]]] = None,
        label_names: Optional[List[str]] = None,
    ):
        self.color_labels = colors_labels
        self.one_hot_labels = one_hot_labels
        self.label_names = label_names

        if one_hot_labels:
            assert len(one_hot_labels) == len(colors_labels)
            if label_names:
                assert len(label_names) == len(one_hot_labels[0])
        elif label_names:
            assert len(label_names) == colors_labels

    @property
    def multilabel(self):
        return self.one_hot_labels is not None

    @property
    def num_classes(self):
        if self.one_hot_labels:
            return len(self.one_hot_labels[0])
        else:
            return len(self.color_labels)

    @classmethod
    def from_labels_text_file(cls, label_text_file: Union[str, Path]):
        label_text_file = str(label_text_file)
        if not os.path.exists(label_text_file):
            raise FileNotFoundError(label_text_file)
        labels_classes = np.loadtxt(label_text_file).astype(np.float32)

        if labels_classes.shape[1] < 3:
            raise ValueError(
                "Text label file did not contain enough information to be colors."
            )

        colors_labels = [parse_and_validate_color(color) for color in labels_classes[:, :3]]

        if labels_classes.shape[1] == 3:
            return cls(colors_labels)
        else:
            one_hot_labels = [parse_validate_one_hot(one_hot) for one_hot in labels_classes[:, 3:]]
            return cls(colors_labels, one_hot_labels)

    @classmethod
    def from_list_of_labels(
        cls, labels: List[Dict[str, Any]], label_names: Optional[List[str]] = None
    ):
        color_labels: List[Tuple[int, int, int]] = []
        one_hot_labels: Optional[List[List[int]]] = None
        label_names: Optional[List[str]] = label_names

        has_one_hot = None
        has_labels = None
        one_hot_size = None

        for label in labels:
            if "color" not in label:
                raise ValueError("Need at least a color to define a label.")
            color = parse_and_validate_color(label["color"])
            color_labels.append(color)

            if "one_hot" in label:
                if has_one_hot is None:
                    has_one_hot = True
                    one_hot_labels = []
                one_hot = parse_validate_one_hot(label["one_hot"])

                if one_hot_size is None:
                    one_hot_size = len(one_hot)
                if not has_one_hot:
                    raise ValueError("Some labels have one hot defined, others not.")
                if one_hot_size != len(one_hot):
                    raise ValueError("Some labels have different one hot sizes.")

                one_hot_labels.append(one_hot)
            else:
                has_one_hot = False

            if "name" in label:
                if has_labels is None:
                    has_labels = True
                    label_names = []
                if not has_labels:
                    raise ValueError("Some labels have a name, others not.")
                label_names.append(label["name"])
            else:
                has_labels = False

        return cls(color_labels, one_hot_labels, label_names)


ColorLabels.register("labels_list", "from_list_of_labels")(ColorLabels)
ColorLabels.register("txt", "from_labels_text_file")(ColorLabels)


def hex_to_rgb(hex: str) -> Tuple[int, ...]:
    hex = hex.lstrip("#")
    return tuple(int(hex[i: i + 2], 16) for i in (0, 2, 4))


def parse_and_validate_color(color) -> Tuple[int, int, int]:
    if not isinstance(color, str) and not (
        isinstance(color, Sized) and len(color) == 3
    ):
        raise ValueError("Colors needs to be defined either by 3 ints or a hex string")
    if isinstance(color, str):
        color = hex_to_rgb(color)
    color = np.array(color).astype(np.float32)
    if (color <= 1.0).all():
        color = np.round(color * 255)
    color = color.astype(np.int32)
    if np.max(color) > 255 or np.min(color) < 0:
        raise ValueError("A color should have values between 0 and 255")
    return color[0], color[1], color[2]


def parse_validate_one_hot(one_hot) -> List[int]:
    if not isinstance(one_hot, str) and not isinstance(one_hot, Sized):
        raise ValueError("One hot needs to be defined either by a sequence or a string")
    if isinstance(one_hot, str):
        one_hot = [x for x in one_hot]
    one_hot = np.array(one_hot).astype(np.int32)
    if len(set(np.unique(one_hot).tolist()).difference({0, 1})) > 0:
        raise ValueError("Found not 0 and 1 ")
    return [x for x in one_hot]
