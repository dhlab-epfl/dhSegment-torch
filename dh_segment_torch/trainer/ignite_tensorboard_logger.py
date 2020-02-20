from functools import partial

import numpy as np

import torch

import torch.nn.functional as F

from ignite.contrib.handlers import TensorboardLogger
from ignite.contrib.handlers.base_logger import BaseHandler
from ignite.engine import Events

from torchvision.utils import make_grid

from .utils import cut_with_padding


class LogImagesHandler(BaseHandler):
    def __init__(
        self,
        colors,
        one_hot=None,
        multilabel=False,
        one_large_image=False,
        max_images=1,
        global_step_engine=None,
        global_step_event=Events.ITERATION_COMPLETED,
    ):
        if not torch.is_tensor(colors):
            colors = np.array(colors)
            if np.max(colors) == 255:
                colors /= 255
            colors = torch.from_numpy(colors.astype(np.float32))
        if one_hot is not None and not torch.is_tensor(one_hot):
            one_hot = np.array(one_hot)
            self.one_hot = torch.from_numpy(one_hot).unsqueeze(0)
        self.colors = colors
        self.multilabel = multilabel
        self.one_large_image = one_large_image
        self.max_images = max_images
        self.global_step_engine = global_step_engine
        self.get_global_step = partial(get_global_step, event_name=global_step_event)

    def __call__(self, engine, logger, event_name):
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'LogImagesHandler' works only with TensorboardLogger"
            )
        output = engine.state.output
        if self.global_step_engine:
            global_step = self.get_global_step(self.global_step_engine)
        else:
            global_step = self.get_global_step(engine)

        xs, (ys, shapes), y_preds = output

        xs = xs.cpu()
        ys = ys.cpu()
        shapes = shapes.cpu()
        y_preds = y_preds.cpu()

        out_images = []

        for idx in range(len(xs)):
            if idx == self.max_images:
                break
            x, y, shape, y_pred = xs[idx], ys[idx], shapes[idx], y_preds[idx]
            if self.multilabel:
                y = one_hot_indices(y, self.one_hot)
                y_pred = torch.sigmoid(y_pred)
                y_pred = one_hot_indices(y_pred, self.one_hot)
            else:
                y_pred = y_pred.argmax(dim=0)

            y = indices_to_image(y, self.colors)
            y_pred = indices_to_image(y_pred, self.colors)

            if not self.one_large_image:
                x = cut_with_padding(x, shape)
                y = cut_with_padding(y, shape)
                y_pred = cut_with_padding(y_pred, shape)
            images = concat_images(x, y, y_pred)
            out_image = make_grid(
                images, padding=5, pad_value=0.5, normalize=True, nrow=3
            )
            out_images.append(out_image)
        if not self.one_large_image:
            for out_image in out_images:
                logger.writer.add_image(
                    tag=f"Image outputs (image, annotation, prediction)",
                    img_tensor=out_image,
                    global_step=global_step,
                    dataformats="CHW",
                )
        else:
            images = concat_images(*out_images)
            out_image = make_grid(images, padding=0, nrow=1)
            logger.writer.add_image(
                tag=f"Image outputs (image, annotation, prediction)",
                img_tensor=out_image,
                global_step=global_step,
                dataformats="CHW",
            )


def indices_to_image(indices, colors, batch=False):
    if batch:
        colors_transform = (
            colors.T.unsqueeze(1)
            .unsqueeze(0)
            .expand(indices.shape[0], -1, indices.shape[1], -1)
        )
        indices_transform = indices.unsqueeze(1).expand(-1, 3, -1, -1)
    else:
        colors_transform = colors.T.unsqueeze(1).expand(-1, indices.shape[0], -1)
        indices_transform = indices.unsqueeze(0).expand(3, -1, -1)
    return torch.gather(colors_transform, -1, indices_transform)


def get_global_step(engine, event_name):
    return engine.state.get_event_attrib_value(event_name)


def concat_images(*images):
    return torch.cat([image.unsqueeze(0) for image in images])


def one_hot_indices(probas, one_hot):
    probas = (probas > 0.5).float()
    probas = probas.permute((1, 2, 0))
    indices = torch.cdist(probas, one_hot).argmin(dim=2).long()

    return indices
