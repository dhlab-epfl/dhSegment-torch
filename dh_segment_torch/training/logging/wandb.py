from typing import Dict, Any, Optional

import numpy as np
import torch

from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.training.logging.logger import Logger
from dh_segment_torch.utils.ops import cut_with_padding


@Logger.register("wandb")
class WandbLogger(Logger):
    def __init__(
        self,
        project_name: str,
        color_labels: ColorLabels,
        config: Optional[Dict[str, Any]] = None,
        log_every: int = 200,
        log_images_every: int = 500,
        probas_threshold: float = 0.5,
        max_images_to_log: int = 4,
        ignore_padding: bool = False,
        margin: int = 0,
        names_separator: str = "/",
    ):
        super().__init__(
            color_labels,
            log_every,
            log_images_every,
            probas_threshold,
            max_images_to_log,
            ignore_padding,
            margin,
            names_separator,
        )
        import wandb

        self.wandb = wandb

        self.wandb.init(project=project_name, config=config)

    def log_scalar(self, scalar: float, iteration: int, name: str):
        self.wandb.log({name: scalar}, commit=False, step=iteration)

    def log_batch(
        self, batch: Dict[str, torch.Tensor], iteration: int, prefix: str = ""
    ):
        batch_size = len(batch["input"])
        indices_sel, _ = torch.sort(
            torch.randperm(batch_size)[: self.max_images_to_log]
        )

        shapes = batch["shapes"][indices_sel] if "shapes" in batch else None
        images = batch["input"][indices_sel]
        preds = None
        gts = None

        if "logits" in batch:
            logits = batch["logits"][indices_sel]

            if self.color_labels.multilabel:
                one_hot = (torch.sigmoid(logits) > self.probas_threshold).to(torch.long)
                preds = self._one_hot_to_indices(one_hot)
            else:
                preds = logits.argmax(dim=1)

        if "target" in batch:
            gts = batch["target"][indices_sel]

            if self.color_labels.multilabel:
                gts = self._one_hot_to_indices(gts)

        all_images = []
        for idx in range(len(indices_sel)):
            shape = shapes[idx]
            image = images[idx]
            if self.ignore_padding:
                image = cut_with_padding(image, shape, self.margin)
            image = (image.permute(1, 2, 0)*255).numpy().astype(np.uint8)

            mask_dict = {}
            if preds is not None:
                pred = preds[idx]
                if self.ignore_padding:
                    pred = cut_with_padding(pred, shape, self.margin)
                pred = pred.numpy().astype(np.uint8)
                mask_dict["prediction"] = {
                    "mask_data": pred,
                    "class_labels": dict(enumerate(self.color_labels.labels))
                    if self.color_labels.labels
                    else dict(enumerate([str(x) for x in range(self.color_labels.num_classes)])),
                }


            if gts is not None:
                gt = gts[idx]
                if self.ignore_padding:
                    gt = cut_with_padding(gt, shape, self.margin)
                gt = gt.numpy().astype(np.uint8)
                mask_dict["ground_truth"] = {
                    "mask_data": gt,
                    "class_labels": dict(enumerate(self.color_labels.labels))
                    if self.color_labels.labels
                    else dict(enumerate([str(x) for x in range(self.color_labels.num_classes)])),
                }
            if len(mask_dict) == 0:
                mask_dict = None
            all_images.append(self.wandb.Image(image, masks=mask_dict))
        self.wandb.log({"images": all_images}, commit=False, step=iteration)
