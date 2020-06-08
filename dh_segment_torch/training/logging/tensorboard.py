from pathlib import Path
from typing import Union, Optional, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.training.logging.logger import Logger


@Logger.register("tensorboard")
class TensorboardLogger(Logger):
    def __init__(
        self,
        log_dir: Union[str, Path],
        color_labels: ColorLabels,
        log_every: int = 200,
        log_images_every: int = 500,
        probas_threshold: float = 0.5,
        max_images_to_log: int = 4,
        ignore_padding: bool = False,
        margin: int = 0,
        names_separator: str = "/",
        exp_name: str = "",
        config: Optional[Dict[str, Any]] = None
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
        self.log_dir = log_dir

        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, scalar: float, iteration: int, name: str):
        self.writer.add_scalar(name, scalar, global_step=iteration)

    def log_image(self, image: torch.Tensor, iteration: int, name: str):
        self.writer.add_image(name, image, global_step=iteration, dataformats="CHW")

    def log_mask(self, mask: torch.Tensor, iteration: int, name: str):
        image = torch.index_select(
            self.colors.to(torch.float32) / 255.0, dim=0, index=mask.flatten()
        ).reshape(*mask.shape, -1)
        self.writer.add_image(name, image, global_step=iteration, dataformats="HWC")
