import torch

from dh_segment_torch.config.params import Params
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase
from dh_segment_torch.training.metrics.metric import Metric


class OptimizerTest(DhSegmentTestCase):
    def test_metrics_multiclass(self):
        n_classes = 3
        classes_labels = ["b", "t1", "t2"]

        logits = (
            torch.tensor(
                [
                    [[[10, 1, 1], [10, 1, 1], [1, 1, 10], [1, 1, 10]]],
                    [[[1, 10, 1], [1, 1, 10], [1, 1, 10], [1, 10, 1]]],
                ]
            )
            .to(torch.float)
            .permute(0, 3, 1, 2)
        )

        y_true = torch.tensor([[[0, 1, 2, 2]], [[1, 2, 2, 0]]])

        params = {"type": "precision", "average": "micro"}

        precision = Metric.from_params(
            Params(params), num_classes=n_classes, classes_labels=classes_labels
        )
        precision.get_metric_value()
        precision(y_true, logits)
        val = precision.get_metric_value()
        assert val == 5 / 6

        params = {
            "type": "precision",
            "average": "micro",
            "batch_average": True,
        }

        precision = Metric.from_params(
            Params(params), num_classes=n_classes, classes_labels=classes_labels
        )

        precision(y_true, logits)
        val = precision.get_metric_value()

        assert val == (2 / 2 + 3 / 4) / 2

    def test_metrics_multilabel(self):
        n_classes = 2
        classes_labels = ["t1", "t2"]
        y_pred = torch.tensor(
            [[[1, 0], [1, 1]], [[0, 1], [0, 0]]], dtype=torch.float
        ).permute(0, 2, 1)

        y_true = torch.tensor(
            [[[1, 1], [1, 0]], [[1, 1], [0, 0]]], dtype=torch.long
        ).permute(0, 2, 1)

        params = {"type": "precision", "average": "micro", "multilabel": True}

        precision = Metric.from_params(
            Params(params), num_classes=n_classes, classes_labels=classes_labels
        )
        precision(y_true, y_pred)
        val = precision.get_metric_value()
        assert val == 3 / 4

        params = {
            "type": "precision",
            "average": "micro",
            "batch_average": True,
            "multilabel": True,
        }

        precision = Metric.from_params(
            Params(params), num_classes=n_classes, classes_labels=classes_labels
        )

        precision(y_true, y_pred)
        val = precision.get_metric_value()

        assert val == (2 / 3 + 1 / 1) / 2
