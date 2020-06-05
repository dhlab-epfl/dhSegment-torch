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
        colors: List[Tuple[int, int, int]],
        one_hot_encoding: Optional[List[List[int]]] = None,
        labels: Optional[List[str]] = None,
    ):
        self.colors = colors
        self.one_hot_encoding = one_hot_encoding
        self.labels = labels

        if labels:
            if one_hot_encoding:
                new_names = []
                for line in np.array(one_hot_encoding).astype(bool):
                    new_names.append("+".join(np.array(labels)[line]))
                new_names[0] = 'background'
                self.labels = new_names
            assert len(self.labels) == len(colors)

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
    def from_labels_text_file(cls, label_text_file: Union[str, Path]):
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
            one_hot_encoding = [parse_validate_one_hot(one_hot) for one_hot in labels_classes[:, 3:]]
            return cls(colors, one_hot_encoding)

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


ColorLabels.register("labels_list", "from_list_of_color_labels")(ColorLabels)
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
