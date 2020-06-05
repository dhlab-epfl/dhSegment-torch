from math import inf
from typing import Any, Dict

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.training.metrics.metric import MetricType


class MetricTracker(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        metric_name: str,
        threshold: float = 1e-5,
        threshold_mode: str = "abs",
    ):
        if metric_name[0] not in {'-', '+'}:
            raise ValueError("Expected metric name to start by - or +"
                             "to indicate if it should be maximized or minimized")
        self.mode = 'min' if metric_name[0] == '-' else 'max'
        self.metric_name = metric_name[1:]

        self.threshold = threshold
        self.threshold_mode = threshold_mode

        self.best = None
        self.is_last_best = False
        self.last_value = None

        if self.mode not in {"min", "max"}:
            raise ValueError("mode " + self.mode + " is unknown!")
        if self.threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + self.mode + " is unknown!")
        if self.mode == "min":
            self.best = inf
            self.last_value = inf
        else:  # mode == 'max':
            self.best = -inf
            self.last_value = -inf

    def update(self, metrics: Dict[str, MetricType], losses: Dict[str, float]):
        if self.metric_name in metrics:
            curr_value = metrics[self.metric_name]
        elif self.metric_name in losses:
            curr_value = losses[self.metric_name]
        else:
            raise ValueError(f"Metric {self.metric_name} is not in metrics dict.")
        if self._is_best(curr_value):
            self.best = curr_value
            self.is_last_best = True
        else:
            self.is_last_best = False
        self.last_value = curr_value

    def _is_best(self, metric_value) -> bool:
        best_value = self.best
        if self.threshold == "rel":
            if self.mode == "min":
                rel_epsilon = 1.0 - self.threshold
                return metric_value < best_value * rel_epsilon
            else:  # mode = 'max'
                rel_epsilon = 1.0 + self.threshold
                return metric_value > best_value * rel_epsilon
        else:  # threshold = 'abs'
            if self.mode == "min":
                return metric_value < best_value - self.threshold
            else:  # mode = 'max'
                return metric_value >= best_value + self.threshold

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "model"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


MetricTracker.register("default")(MetricTracker)
