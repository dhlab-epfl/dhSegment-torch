from logging import Logger
from typing import Optional, Union, Dict, List, NewType

import torch

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.training.metrics.utils import (
    batch_multilabel_confusion_matrix,
    nanaverage,
)
from dh_segment_torch.utils.ops import cut_with_padding
from dh_segment_torch.utils.ops import detach_and_move_tensors

logger = Logger(__name__)

MetricType = NewType(
    "Metric", Union[float, List[float], Dict[str, float], Dict[str, List[float]]]
)


class Metric(Registrable):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        multilabel: bool = False,
        ignore_padding: bool = False,
        margin: int = 0,
        device: Optional[str] = "cpu",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.ignore_padding = ignore_padding
        self.margin = margin
        self.device = device

    def __call__(
        self,
        labels: torch.Tensor,
        logits: torch.Tensor,
        shapes: Optional[torch.Tensor] = None,
    ):
        labels, logits, shapes = detach_and_move_tensors(
            labels, logits, shapes, device=self.device, non_blocking=True,
        )
        if self.multilabel:
            probas = torch.sigmoid(logits)
        else:
            probas = torch.softmax(logits, dim=1)
        if self.ignore_padding and logits.shape[0] > 1:
            if shapes is None or shapes.shape[0] != logits.shape[0]:
                raise ValueError(
                    "Ignoring padding, thus shapes should be set"
                    "and have the same size as the input"
                )
            for idx in range(logits.shape[0]):
                shape = shapes[idx]
                sub_labels = (
                    cut_with_padding(labels[idx], shape, self.margin)
                    .unsqueeze(0)
                    .contiguous()
                )
                sub_probas = (
                    cut_with_padding(probas[idx], shape, self.margin)
                    .unsqeeze(0)
                    .contiguous()
                )

                self._update(sub_labels, sub_probas)
        else:
            self._update(labels, probas)

    def _update(
        self, labels: torch.Tensor, probas: torch.Tensor,
    ):
        raise NotImplementedError

    def get_metric_value(self, reset: bool = False) -> MetricType:
        raise NotImplementedError

    def _reset_if_needed(self, reset: bool = False):
        if reset:
            self.reset()

    def reset(self):
        raise NotImplementedError

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class MultilabelConfusionMetric(Metric):
    def __init__(
        self,
        num_classes: int,
        classes_labels: Optional[List[str]] = None,
        probas_threshold: float = 0.5,
        average: Optional[str] = "micro",
        batch_average: bool = False,
        multilabel: bool = False,
        ignore_padding: bool = False,
        margin: int = 0,
        device: Optional[str] = "cpu",
    ):
        super().__init__(num_classes, multilabel, ignore_padding, margin, device)
        self.classes_labels = (
            classes_labels
            if classes_labels
            else [str(idx) for idx in range(num_classes)]
        )
        self.probas_threshold = probas_threshold
        self.average = average
        self.batch_average = batch_average
        self._batch_multilabel_confusion_matrix = torch.LongTensor()

    def _update(self, labels: torch.Tensor, probas: torch.Tensor):
        if self.multilabel:
            predictions = (probas > self.probas_threshold).to(torch.long)
        else:
            predictions = probas.argmax(dim=1).to(torch.long)

        matrix = detach_and_move_tensors(batch_multilabel_confusion_matrix(
            labels, predictions, self.num_classes, self.multilabel
        ), device='cpu', non_blocking=True)
        self._batch_multilabel_confusion_matrix = torch.cat(
            [self._batch_multilabel_confusion_matrix, matrix], 0
        )

    @property
    def multilabel_confusion_matrix(self):
        mcm = self._batch_multilabel_confusion_matrix.numpy()
        labels_axis = 1
        if not self.batch_average:
            mcm = mcm.sum(axis=0)
            labels_axis = 0
        if self.average == "micro":
            mcm = mcm.sum(axis=labels_axis)
        return mcm

    @property
    def support(self):
        return (
            self.multilabel_confusion_matrix[..., 1, 0]
            + self.multilabel_confusion_matrix[..., 1, 1]
        )

    def _reduce_metric(self, metric) -> MetricType:
        weights = None
        if self.average == "weighted":
            weights = self.support
        if self.batch_average:
            metric = nanaverage(metric, axis=0, weights=weights)
            weights = None
        if self.average is None:
            return dict(zip(self.classes_labels, metric))

        return nanaverage(metric, weights=weights)

    @property
    def is_initialized(self):
        return self._batch_multilabel_confusion_matrix.ndim > 1

    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:

        raise NotImplementedError

    def reset(self):
        self._batch_multilabel_confusion_matrix = torch.LongTensor()
