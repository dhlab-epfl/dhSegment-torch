import os
import time
from typing import Optional, List, Tuple, Dict, Any

import torch
import numpy as np
import logging

from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.training.metrics.metric_tracker import MetricTracker
from dh_segment_torch.utils.ops import join_not_none, format_time

logger = logging.getLogger(__name__)


class Checkpoint(Registrable):
    """
    Abstract checkpoint class that takes save checkpoints to a specific dir and keep only a certain number.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        prefix: str = "",
        checkpoints_to_keep: int = 5,
    ):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.prefix = prefix
        self.checkpoint_to_keep = checkpoints_to_keep
        self.saved_checkpoints: List[Tuple[float, str]] = []
        self._last_save = 0.0
        self._last_permanent_save = 0.0

    def maybe_save(self, save_dict: Dict[str, Any]):
        self._update_value()
        if self.should_save():
            self.save(save_dict)
        if self.should_save_permanent():
            self.save(save_dict, True)

    def save(self, save_dict: Dict[str, Any], permanent: bool = False):
        current_value, suffix = self.get_save_infos()
        self._save_and_delete_if_needed(save_dict, current_value, suffix, permanent)
        if permanent:
            self._last_permanent_save = current_value
        else:
            self._last_save = current_value

    def should_save(self) -> bool:
        raise NotImplementedError

    def should_save_permanent(self) -> bool:
        raise NotImplementedError

    def get_save_infos(self) -> Tuple[float, str]:
        raise NotImplementedError

    def _update_value(self):
        raise NotImplementedError

    def _save_and_delete_if_needed(
        self,
        save_dict: Dict[str, Any],
        value: float,
        suffix: str = "",
        permanent: bool = False,
    ):
        if self.checkpoint_dir is None:
            return
        # Do not save twice the same checkpoint

        if len(self.saved_checkpoints) > 0 and (np.array([v for v, _ in self.saved_checkpoints]) - value).min() < 1e-6:
            return
        save_path = os.path.join(
            self.checkpoint_dir,
            join_not_none(self.prefix, "checkpoint", suffix, ".pth"),
        )
        torch.save(save_dict, save_path)
        if permanent:
            return
        self.saved_checkpoints.append((value, save_path))
        if len(self.saved_checkpoints) > self.checkpoint_to_keep:
            self.saved_checkpoints = sorted(self.saved_checkpoints)
            _, delete_path = self.saved_checkpoints.pop(0)
            try:
                os.remove(delete_path)
            except FileNotFoundError:
                logger.warning(f"Tried to delete {delete_path} but did not exist")

    def state_dict(self):
        return self.__dict__

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


@Checkpoint.register("time")
class TimeCheckpoint(Checkpoint):
    """
    Checkpoint that saves every n seconds.
    The maybe_save method should be called at the end of each iteration.
    """

    def __init__(
        self,
        every_n_seconds: int = 300,
        permanent_every_n_seconds: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        prefix: str = "",
        checkpoints_to_keep: int = 5,
    ):
        super().__init__(checkpoint_dir, prefix, checkpoints_to_keep)
        self.every_n_seconds = every_n_seconds
        self.permanent_every_n_seconds = permanent_every_n_seconds

        self._start_time = time.time()
        self._last_save = time.time()
        self._last_permanent_save = time.time()
        self._current_time = time.time()

    def should_save(self) -> bool:
        return self._current_time - self._last_save >= self.every_n_seconds

    def should_save_permanent(self) -> bool:
        return (
            self.permanent_every_n_seconds
            and self._current_time - self._last_permanent_save
            >= self.permanent_every_n_seconds
        )

    def _update_value(self):
        self._current_time = time.time()

    def get_save_infos(self) -> Tuple[float, str]:
        current_time = time.time()
        return current_time, f"t={format_time(current_time)}"


@Checkpoint.register("iteration")
class IterationCheckpoint(Checkpoint):
    """
    Checkpoint that saves every n iterations.
    The maybe_save method should be called at the end of each iteration.
    """

    def __init__(
        self,
        every_n_iterations: int = 500,
        permanent_every_n_iterations: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        prefix: str = "",
        checkpoints_to_keep: int = 5,
    ):
        super().__init__(checkpoint_dir, prefix, checkpoints_to_keep)
        self.every_n_iterations = every_n_iterations
        self.permanent_every_n_iterations = permanent_every_n_iterations

        self._num_iterations = 0
        self._last_save = 0
        self._last_permanent_save = 0

    def should_save(self) -> bool:
        return self._num_iterations - self._last_save >= self.every_n_iterations

    def should_save_permanent(self) -> bool:
        return (
            self.permanent_every_n_iterations
            and self._num_iterations - self._last_permanent_save
            >= self.permanent_every_n_iterations
        )

    def _update_value(self):
        self._num_iterations += 1

    def get_save_infos(self) -> Tuple[float, str]:
        iterations = self._num_iterations
        return iterations, f"iter={iterations}"


@Checkpoint.register("best")
class BestCheckpoint(Checkpoint):
    """
    Checkpoint that saves the best results for a metric.
    The maybe_save method should be called at the end of each complete evaluation.
    """

    def __init__(
        self,
        tracker: MetricTracker,
        checkpoint_dir: Optional[str] = None,
        prefix: str = "best",
        checkpoints_to_keep: int = 5,
    ):
        super().__init__(checkpoint_dir, prefix, checkpoints_to_keep)
        self.tracker = tracker

    def should_save(self) -> bool:
        return self.tracker.is_last_best

    def should_save_permanent(self) -> bool:
        return False

    def _update_value(self):
        pass

    def get_save_infos(self) -> Tuple[float, str]:
        best = self.tracker.best
        metric_name = self.tracker.metric_name
        return best, f"{metric_name}={best:.3f}"

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "tracker"}
