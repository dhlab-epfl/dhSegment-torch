from itertools import product
from typing import Dict, Any, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.data.datasets.dataset import Dataset
from dh_segment_torch.models.model import Model
from dh_segment_torch.utils.ops import batch_items


class InferenceModel(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        model: Model,
        num_classes: int,
        margin: int = 0,
        padding_mode: str = "reflect",
        padding_value: int = 0,
        multilabel: bool = False,
        patch_size: Union[int, Tuple[int, int]] = None,
        patches_overlap: Union[int, float] = 0,
        patches_batch_size: int = 4,
        model_state_dict: Dict[str, Any] = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict, strict=False)
        self.model = model
        self.device = device

        self.num_classes = num_classes

        self.margin = margin
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"Padding mode {padding_mode} not supported.")
        self.padding_mode = padding_mode
        self.padding_value = padding_value

        self.multilabel = multilabel

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.patches_overlap = patches_overlap
        self.patches_batch_size = patches_batch_size

        if patch_size:
            self.predict = self.predict_patches
            self.patches_margin = self.margin
            self.margin = 0
        else:
            self.predict = self.predict_batch

    def predict_batch(
        self, images_batch: torch.tensor, shapes: torch.tensor = None
    ) -> torch.tensor:
        images_batch = images_batch.to(self.device)
        if self.margin > 0:
            images_batch = F.pad(
                images_batch, [self.margin] * 4, self.padding_mode, self.padding_value
            )

        with torch.no_grad():
            self.model.eval()
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
        if shapes is not None:
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
            h, w = shape.numpy()
            image = image[..., :h, :w]
            image = F.pad(
                image.unsqueeze(0),
                [self.patches_margin] * 4,
                self.padding_mode,
                self.padding_value,
            ).squeeze(0)

            x_step = compute_step(
                h, self.patch_size[0], self.margin, self.patches_overlap
            )
            y_step = compute_step(
                w, self.patch_size[1], self.margin, self.patches_overlap
            )

            x_pos = (
                np.round(np.arange(x_step + 1) / x_step * (h - self.patch_size[0]))
                .astype(np.int32)
                .tolist()
            )
            y_pos = (
                np.round(np.arange(y_step + 1) / y_step * (w - self.patch_size[1]))
                .astype(np.int32)
                .tolist()
            )

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
                for idx, (x, y) in enumerate(positions):
                    counts[
                        x
                        + self.patches_margin: x
                        + self.patch_size[0]
                        - self.patches_margin,
                        y
                        + self.patches_margin: y
                        + self.patch_size[1]
                        - self.patches_margin,
                    ] += 1
                    probas_sum[
                        :,
                        x
                        + self.patches_margin: x
                        + self.patch_size[0]
                        - self.patches_margin,
                        y
                        + self.patches_margin: y
                        + self.patch_size[1]
                        - self.patches_margin,
                    ] += probas[idx][
                        ...,
                        self.patches_margin: -self.patches_margin,
                        self.patches_margin: -self.patches_margin,
                    ]
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
        model_state_dict: str = None,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        model = model.construct(
            num_classes=num_classes, multilabel=multilabel, margin=margin
        )

        model_state_dict = torch.load(model_state_dict, map_location=device)
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
        model_state_dict: str = None,
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


InferenceModel.register("default", "from_partial")(InferenceModel)
InferenceModel.register("training_config", "from_color_labels_and_dataset")(
    InferenceModel
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
