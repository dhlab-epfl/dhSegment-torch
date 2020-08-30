from typing import Optional

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.metrics import MetricTracker


class EarlyStopping(Registrable):
    default_implementation = "default"

    def __init__(
        self, tracker: MetricTracker, patience: Optional[int] = None,
    ):
        self.tracker = tracker

        self.patience = patience

        self.num_bad_epochs = 0

    def should_terminate(self):
        if self.patience is None:
            return False

        if self.tracker.is_last_best:
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        return self.num_bad_epochs > self.patience

    def reset(self):
        self.num_bad_epochs = 0

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "tracker"}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


EarlyStopping.register("default")(EarlyStopping)
