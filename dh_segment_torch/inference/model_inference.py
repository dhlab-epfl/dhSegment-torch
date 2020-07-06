from itertools import product
from typing import Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.data.datasets.dataset import Dataset
from dh_segment_torch.models import Model
from dh_segment_torch.utils.ops import batch_items


class ModelInference(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        model: Model,
        num_classes: int,
        margin: int = 0,
        padding_mode: str = "reflect",
        padding_value: int = 0,
        multilabel: bool = False,
        patch_size: Tuple[int, int] = None,
        patches_overlap: Union[int, float] = 0,
        patches_batch_size: int = 4,
        model_state_dict: Dict[str, Any] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        self.model = model
        self.device = device

        self.num_classes = num_classes

        self.margin = margin
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"Padding mode {padding_mode} not supported.")
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.multilabel = multilabel
        self.patch_size = patch_size
        self.patches_overlap = patches_overlap
        self.patches_batch_size = patches_batch_size

        if patch_size:
            self.predict = self.predict_patches
        else:
            self.predict = self.predict_batch

    def predict_batch(self, images_batch: torch.tensor) -> torch.tensor:
        images_batch = images_batch.to(self.device)
        if self.margin > 0:
            images_batch = F.pad(
                images_batch, [self.margin] * 4, self.padding_mode, self.padding_value
            )

        result = self.model.to(self.device)(images_batch)
        logits = result["logits"]
        if self.margin > 0:
            logits = logits[..., self.margin : -self.margin, self.margin : -self.margin]

        if self.multilabel:
            probas = torch.sigmoid(logits)
        else:
            probas = torch.softmax(logits, dim=1)

        return probas

    def predict_patches(
        self, images_batch: torch.tensor, shapes: torch.tensor = None
    ) -> torch.tensor:
        if self.patch_size is None:
            raise ValueError("In order to predict ")
        if shapes:
            if len(images_batch) != len(shapes):
                raise ValueError(
                    "Images shapes and images batch should have the same number of samples."
                )
        else:
            if len(images_batch) != 1:
                raise ValueError("If no shapes given, should be a batch of 1.")
            shapes = torch.tensor([images_batch.size()[-2:]])
        results = []
        for image, shape in zip(images_batch, shapes):
            h, w = shape
            image = image[..., :h, :w]

            x_step = compute_step(
                h, self.patch_size[0], self.margin, self.patches_overlap
            )
            y_step = compute_step(
                w, self.patch_size[1], self.margin, self.patches_overlap
            )

            x_pos = np.round(
                np.arange(x_step + 1) / x_step * (h - self.patch_size[0])
            ).astype(np.int32)
            y_pos = np.round(
                np.arange(y_step + 1) / y_step * (w - self.patch_size[1])
            ).astype(np.int32)

            counts = torch.zeros((h, w), dtype=torch.long).to(self.device)
            probas_sum = torch.zeros([self.num_classes, h, w]).to(self.device)

            for positions in batch_items(
                list(product(x_pos, y_pos)), self.patches_batch_size
            ):
                crops = torch.stack(
                    [
                        pos2crop(image, pos, self.patch_size, self.margin)
                        for pos in positions
                    ]
                )

                probas = self.predict_batch(crops)
                for idx, (x, y) in positions:
                    counts[x : x + self.patch_size[0], y : y + self.patch_size[1]] += 1
                    probas_sum[
                        :, x : x + self.patch_size[0], y : y + self.patch_size[1]
                    ] += probas[idx]
            image_probas = probas_sum / counts
            results.append(image_probas)
        return torch.stack(results)

    @classmethod
    def from_partial(
        cls,
        model: Lazy[Model],
        num_classes: int,
        margin: int = 0,
        padding_mode: str = "reflect",
        padding_value: int = 0,
        multilabel: bool = False,
        patch_size: Tuple[int, int] = None,
        patches_overlap: Union[int, float] = 0,
        patches_batch_size: int = 4,
        model_state_dict: Dict[str, Any] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        model = model.construct(
            num_classes=num_classes, multilabel=multilabel, margin=margin
        )
        return cls(
            model,
            num_classes,
            margin,
            padding_mode,
            padding_value,
            multilabel,
            patch_size,
            patches_overlap,
            patches_batch_size,
            model_state_dict,
            device,
        )

    @classmethod
    def from_color_labels_and_dataset(
        cls,
        model: Lazy[Model],
        color_labels: ColorLabels,
        dataset: Dataset,
        margin: int = 0,
        padding_mode: str = "reflect",
        padding_value: int = 0,
        patches_overlap: Union[int, float] = 0,
        patches_batch_size: int = 4,
        model_state_dict: Dict[str, Any] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        if hasattr(dataset, "patches_size"):
            patches_size = dataset.patches_size
        else:
            patches_size = None

        return cls.from_partial(
            model,
            color_labels.num_classes,
            margin,
            padding_mode,
            padding_value,
            color_labels.multilabel,
            patches_size,
            patches_overlap,
            patches_batch_size,
            model_state_dict,
            device,
        )


ModelInference.register("default", "from_partial")(ModelInference)
ModelInference.register("training_config", "from_color_labels_and_dataset")(
    ModelInference
)


def compute_step(size: int, patch_shape: int, margin: int, min_overlap: float) -> int:
    return np.ceil(
        (size - patch_shape - margin) / ((patch_shape - margin) * (1 - min_overlap))
    )


def pos2crop(
    image: torch.tensor,
    pos: Tuple[int, int],
    patch_shape: Tuple[int, int] = (500, 500),
    margin: int = 0,
) -> np.array:
    x, y = pos
    return image[
        ..., x : x + patch_shape[0] + margin * 2, y : y + patch_shape[1] + margin * 2
    ]