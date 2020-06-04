from typing import List, Optional, Union, Dict

from dh_segment_torch.training.metrics.metric import Metric, MultilabelConfusionMetric, logger


@Metric.register("iou")
class IoU(MultilabelConfusionMetric):
    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:
        if not self.is_initialized:
            logger.warning("Trying to get not initialized metric, returning 0.")
            return 0.0
        mcm = self.multilabel_confusion_matrix

        numerator = mcm[..., 1, 1]
        denominator = mcm[..., 1, 1] + mcm[..., 0, 1] + mcm[..., 1, 0]

        iou = self._reduce_metric(numerator / denominator)

        self._reset_if_needed(reset)

        return iou


@Metric.register("accuracy")
class Accuracy(MultilabelConfusionMetric):
    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:
        if not self.is_initialized:
            logger.warning("Trying to get not initialized metric, returning 0.")
            return 0.0
        mcm = self.multilabel_confusion_matrix

        numerator = mcm[..., 1, 1] + mcm[..., 0, 0]
        denominator = mcm[..., 1, 1] + mcm[..., 0, 0] + mcm[..., 0, 1] + mcm[..., 1, 0]

        accuracy = self._reduce_metric(numerator / denominator)

        self._reset_if_needed(reset)

        return accuracy


@Metric.register("precision")
class Precision(MultilabelConfusionMetric):
    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:
        if not self.is_initialized:
            logger.warning("Trying to get not initialized metric, returning 0.")
            return 0.0
        mcm = self.multilabel_confusion_matrix

        numerator = mcm[..., 1, 1]
        denominator = mcm[..., 1, 1] + mcm[..., 0, 1]

        precision = self._reduce_metric(numerator / denominator)

        self._reset_if_needed(reset)

        return precision


@Metric.register("recall")
class Recall(MultilabelConfusionMetric):
    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:
        if not self.is_initialized:
            logger.warning("Trying to get not initialized metric, returning 0.")
            return 0.0
        mcm = self.multilabel_confusion_matrix

        numerator = mcm[..., 1, 1]
        denominator = mcm[..., 1, 1] + mcm[..., 1, 0]

        precision = self._reduce_metric(numerator / denominator)

        self._reset_if_needed(reset)

        return precision


@Metric.register("f_score")
class FScore(MultilabelConfusionMetric):
    def __init__(
        self,
        beta: float,
        num_classes: int,
        classes_labels: Optional[List[str]] = None,
        probas_threshold: float = 0.5,
        average: Optional[str] = "weighted",
        batch_average: bool = False,
        multilabel: bool = False,
        ignore_padding: bool = False,
        margin: int = 0,
        device: Optional[str] = "cpu",
    ):
        super().__init__(
            num_classes,
            classes_labels,
            probas_threshold,
            average,
            batch_average,
            multilabel,
            ignore_padding,
            margin,
            device,
        )
        self.beta2 = beta ** 2

    def get_metric_value(self, reset: bool = False) -> Union[float, Dict[str, float]]:
        if not self.is_initialized:
            logger.warning("Trying to get not initialized metric, returning 0.")
            return 0.0
        mcm = self.multilabel_confusion_matrix

        tp_sum = mcm[:, 1, 1]
        pred_sum = tp_sum + mcm[:, 0, 1]
        true_sum = tp_sum + mcm[:, 1, 0]

        precision = tp_sum / pred_sum
        recall = tp_sum / true_sum

        f_score = (
            (1 + self.beta2) * precision * recall / (self.beta2 * precision + recall)
        )

        f_score = self._reduce_metric(f_score)

        self._reset_if_needed(reset)

        return f_score


@Metric.register("f1_score")
class F1Score(FScore):
    def __init__(
        self,
        num_classes: int,
        classes_labels: Optional[List[str]] = None,
        probas_threshold: float = 0.5,
        average: Optional[str] = "weighted",
        batch_average: bool = False,
        multilabel: bool = False,
        ignore_padding: bool = False,
        margin: int = 0,
        device: Optional[str] = "cpu",
    ):
        super().__init__(
            1,
            num_classes,
            classes_labels,
            probas_threshold,
            average,
            batch_average,
            multilabel,
            ignore_padding,
            margin,
            device,
        )
