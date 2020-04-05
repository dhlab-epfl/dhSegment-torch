import numbers
from typing import List, Optional, Tuple

import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .metrics import Metric
from .utils import should_run, cut_with_padding


class TensorboardLogMetrics:
    def __init__(
        self,
        writer: SummaryWriter,
        metrics: List[Metric],
        metrics_names: Optional[List[str]] = None,
        prefix: Optional[str] = None,
        log_every: int = 100,
    ):
        self.writer = writer
        self.metrics = {}
        if metrics_names is not None:
            if len(metrics_names) != len(metrics):
                raise ValueError("Metrics names and metrics should match in size")
        else:
            metrics_names = [metric.__class__.__name__ for metric in metrics]
        for name, metric in zip(metrics_names, metrics):
            self.metrics[name] = metric

        if prefix is not None:
            self.prefix = prefix + "/"
        else:
            self.prefix = ""

        self.log_every = log_every

    def log(self, iteration: int, reset: bool = False):

        if not should_run(iteration, self.log_every):
            return

        for name, metric in self.metrics.items():
            value = metric.value
            if (
                isinstance(value, numbers.Number)
                or isinstance(value, torch.Tensor)
                and value.ndimension() == 0
            ):
                self.writer.add_scalar(self.prefix + name, value, iteration)
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for idx, val in enumerate(value):
                    self.writer.add_scalar(
                        f"{self.prefix}{name}2/{idx}", val.item(), iteration
                    )
            if reset:
                metric.reset()


class TensorboardLogImages:
    def __init__(
        self,
        writer: SummaryWriter,
        colors: np.array,
        one_hot: Optional[np.array] = None,
        margin: int = 0,
        multilabel: bool = False,
        random_sample_image: bool = True,
        prefix: Optional[str] = None,
        log_every: int = 200,
    ):
        if multilabel and one_hot is None:
            raise ValueError("Cannot be multilabel without one hot encoding")

        self.writer = writer

        if not torch.is_tensor(colors):
            colors = np.array(colors)
            if np.max(colors) == 255:
                colors /= 255
            self.colors = torch.from_numpy(colors.astype(np.float32))
        if one_hot is not None and not torch.is_tensor(one_hot):
            one_hot = np.array(one_hot)
            self.one_hot = torch.from_numpy(one_hot).unsqueeze(0)

        self.margin = margin
        self.multilabel = multilabel
        self.random_sample_image = random_sample_image
        if prefix is not None:
            self.prefix = prefix + "/"
        else:
            self.prefix = ""

        self.log_every = log_every

    def log(self, iteration: int, x, y, y_pred):
        if not should_run(iteration, self.log_every):
            return

        xs = x.detach().cpu()
        ys, shapes = y
        ys = ys.detach().cpu()
        shapes = shapes.detach().cpu()
        y_preds = y_pred.detach().cpu()

        if self.random_sample_image:
            idx = torch.randint(low=0, high=xs.shape[0], size=(1,)).item()
        else:
            idx = -1

        x, y, shape, y_pred = xs[idx], ys[idx], shapes[idx], y_preds[idx]

        if self.multilabel:
            y = one_hot_indices(y, self.one_hot)
            y_pred = torch.sigmoid(y_pred)
            y_pred = one_hot_indices(y_pred, self.one_hot)
        else:
            y_pred = y_pred.argmax(dim=0)

        y = indices_to_image(y, self.colors)
        y_pred = indices_to_image(y_pred, self.colors)

        x = cut_with_padding(x, shape, self.margin)
        y = cut_with_padding(y, shape, self.margin)
        y_pred = cut_with_padding(y_pred, shape, self.margin)

        images = torch.cat([image.unsqueeze(0) for image in [x, y, y_pred]])
        image = make_grid(images, padding=5, pad_value=0.5, normalize=True, nrow=3)
        self.writer.add_image(
            f"{self.prefix}Image", image, global_step=iteration, dataformats="CHW"
        )


def one_hot_indices(probas, one_hot):
    probas = (probas > 0.5).float()
    probas = probas.permute((1, 2, 0))
    indices = torch.cdist(probas, one_hot).argmin(dim=2).long()

    return indices


def indices_to_image(indices, colors, batch=False):
    colors_transform = colors.T.unsqueeze(1).expand(-1, indices.shape[0], -1)
    indices_transform = indices.unsqueeze(0).expand(3, -1, -1)
    return torch.gather(colors_transform, -1, indices_transform)
