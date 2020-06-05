from collections.abc import Sequence
from typing import Dict, Optional, Union, List

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.training.metrics.metric import MetricType
from dh_segment_torch.training.schedulers import Scheduler
from dh_segment_torch.utils.ops import move_and_detach_batch, join_not_none, should_run, cut_with_padding


class Logger(Registrable):
    def __init__(
        self,
        color_labels: ColorLabels,
        log_every: int = 200,
        log_images_every: int = 500,
        probas_threshold: float = 0.5,
        max_images_to_log: int = 4,
        ignore_padding: bool = False,
        margin: int = 0,
        names_separator: str = "/"
    ):
        self.color_labels = color_labels
        self.log_every = log_every
        self.log_images_every = log_images_every
        self.probas_threshold = probas_threshold
        self.max_images_to_log = max_images_to_log
        self.ignore_padding = ignore_padding
        self.margin = margin
        self.names_separator = names_separator

        self.multilabel = color_labels.multilabel

        self.colors = torch.tensor(color_labels.colors)

    def log(
        self,
        iteration: int,
        metrics: Optional[Dict[str, MetricType]] = None,
        losses: Optional[Dict[str, float]] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None,
        logits: Optional[torch.Tensor] = None,
        scheduler: Optional[Scheduler] = None,
        prefix: str = "",
        ignore_iters: bool = False
    ):
        if ignore_iters or should_run(iteration, self.log_every):
            if metrics:
                self.log_scalars(metrics, iteration, self._join(prefix, "metrics"))
            if scheduler:
                lrs = scheduler.get_last_lr()
                self.log_scalars({"learning_rate": lrs}, iteration, prefix)
            if losses:
                self.log_scalars(losses, iteration, prefix)

        if (ignore_iters or should_run(iteration, self.log_images_every)) and batch:
            if logits is not None:
                batch["logits"] = logits
            batch = move_and_detach_batch(batch, "cpu", non_blocking=True)
            self.log_batch(batch, iteration, prefix)


    def log_scalars(
        self, scalars: Dict[str, MetricType], iteration: int, prefix: str = "",
    ):
        for name, scalar in scalars.items():

            if isinstance(scalar, Sequence):
                if len(scalar) == 1:
                    self.log_scalar(scalar[0], iteration, self._join(prefix, name))
                    continue
                for idx, item in enumerate(scalar):
                    self.log_scalar(item, iteration, self._join(prefix, name, str(idx)))
            else:
                self.log_scalar(scalar, iteration, self._join(prefix, name))

    def log_scalar(self, scalar: float, iteration: int, name: str):
        raise NotImplementedError()

    def log_batch(
        self, batch: Dict[str, torch.Tensor], iteration: int, prefix: str = ""
    ):
        batch_size = len(batch["input"])
        indices_sel, _ = torch.sort(
            torch.randperm(batch_size)[: self.max_images_to_log]
        )

        shapes = batch['shapes'][indices_sel] if 'shapes' in batch else None

        self.log_images(batch['input'][indices_sel], iteration, self._join(prefix, "image"), shapes)

        if "logits" in batch:
            logits = batch["logits"][indices_sel]

            if self.color_labels.multilabel:
                one_hot = (torch.sigmoid(logits) > self.probas_threshold).to(torch.long)
                indices = self._one_hot_to_indices(one_hot)
            else:
                indices = logits.argmax(dim=1)
            self.log_masks(indices, iteration, self._join(prefix, "mask_pred"), shapes)

        if "target" in batch:
            indices = batch['target'][indices_sel]

            if self.color_labels.multilabel:
                indices = self._one_hot_to_indices(indices)
            self.log_masks(indices, iteration, self._join(prefix, "mask_gt"), shapes)

    def log_images(self, images: torch.Tensor, iteration: int, prefix: str, shapes: Optional[torch.Tensor] = None):
        if len(images) == 1:
            self.log_image(images[0], iteration, prefix)
            return
        for idx in range(images.shape[0]):
            image = images[idx]
            if self.ignore_padding:
                image = cut_with_padding(image, shapes[idx], margin=self.margin)
            self.log_image(image, iteration, self._join(prefix, str(idx)))

    def log_masks(self, masks: torch.Tensor, iteration: int, prefix: str, shapes: Optional[torch.Tensor] = None):
        if len(masks) == 1:
            self.log_mask(masks[0], iteration, prefix)
            return
        for idx in range(masks.shape[0]):
            mask = masks[idx]
            if self.ignore_padding:
                mask = cut_with_padding(mask, shapes[idx], margin=self.margin)
            self.log_mask(mask, iteration, self._join(prefix, str(idx)))

    def log_image(self, image: torch.Tensor, iteration: int, name: str):
        raise NotImplementedError

    def log_mask(self, mask: torch.Tensor, iteration: int, name: str):
        raise NotImplementedError

    def _one_hot_to_indices(self, one_hot: torch.Tensor):
        one_hot_encoding = torch.tensor(
            self.color_labels.one_hot_encoding, dtype=torch.long
        ).permute(1, 0)[None, :, :, None, None]
        return torch.abs(one_hot[:, :, None] - one_hot_encoding).sum(dim=1).argmin(dim=1)
    
    def _join(self, *items):
        return join_not_none(*items, join_str=self.names_separator)
