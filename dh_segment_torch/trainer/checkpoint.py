import os
import tempfile
from typing import Any, Dict, Optional

import numpy as np
import torch

from torch import nn

from .metrics import Metric
from .utils import should_run


class Checkpoint:
    def __init__(
        self,
        checkpoint_dir: str,
        prefix: str,
        save_dict: Dict[str, nn.Module],
        metric: Optional[Metric] = None,
        metric_name: Optional[str] = None,
        max_checkpoints: int = 1,
        save_every: int = 1,
    ):
        self.checkpoint_dir = checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.prefix = prefix
        self.save_dict = save_dict
        self.metric = metric
        if metric is not None and metric_name is None:
            metric_name = metric.__class__.__name__
        self.metric_name = metric_name
        self.max_checkpoints = max_checkpoints
        self.save_every = save_every

        self.best_score = 0
        self.save_names = []

    def save(self, iteration):
        if not should_run(iteration, self.save_every):
            return

        if len(self.save_names) <= self.max_checkpoints:
            self._save_and_remove(iteration)
        elif self.metric is None:
            self._save_and_remove(iteration, self._find_earliest())
        else:
            lowest_score_name = self._find_lowest_score()
            lowest_score = save_name_to_score(lowest_score_name, self.metric_name)
            if self.metric.value > lowest_score:
                self._save_and_remove(iteration, lowest_score_name)

    def _save_and_remove(self, iteration, to_remove: Optional[str] = None):
        save_name = self._get_save_name(iteration)
        save_path = os.path.join(self.checkpoint_dir, save_name)
        if to_remove is not None:
            remove_path = os.path.join(self.checkpoint_dir, to_remove)

        tmp = tempfile.NamedTemporaryFile(delete=False, dir=self.checkpoint_dir)
        try:
            torch.save(self._get_save_data(), tmp.name)
        except BaseException:
            tmp.close()
            os.remove(tmp.name)
        else:
            tmp.close()
            os.rename(tmp.name, save_path)
            if to_remove is not None:
                os.remove(remove_path)
                self.save_names.remove(to_remove)
            self.save_names.append(save_name)

    def _find_earliest(self):
        min_idx = np.argmin(list(map(save_name_to_iter, self.save_names)))
        return self.save_names[min_idx]

    def _find_lowest_score(self):
        if self.metric is None:
            return self._find_earliest()
        else:
            save_names = sorted(self.save_names, key=save_name_to_iter)
            min_idx = np.argmin(
                [
                    save_name_to_score(save_name, self.metric_name)
                    for save_name in save_names
                ]
            )
            return save_names[min_idx]

    def _get_save_data(self):
        save_data = {}
        for name, item in self.save_dict.items():
            if hasattr(item, "state_dict"):
                save_data[name] = item.state_dict()
            elif callable(item):
                save_data[name] = item()
            else:
                save_data[name] = item
        return save_data

    def _get_save_name(self, iteration):
        name_parts = [self.prefix]

        if self.metric is not None:
            name_parts.append(f"{self.metric_name}={self.metric.value:.4f}")

        name_parts.append(f"iter={iteration}")

        return "_".join(name_parts) + ".pth"


def save_name_to_iter(save_name):
    return int(save_name.split("iter=")[-1].replace(".pth", ""))


def save_name_to_score(save_name, metric_name):
    return float(save_name.split(f"{metric_name}=")[1].split("_")[0])
