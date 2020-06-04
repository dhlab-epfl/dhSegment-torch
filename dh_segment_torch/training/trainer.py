from typing import Optional, List, Union, Dict

import torch
from torch.utils import data
from tqdm import tqdm
import logging

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.data.data_loader import DataLoader
from dh_segment_torch.data.dataset import PatchesDataset
from dh_segment_torch.data.dataset.dataset import Dataset
from dh_segment_torch.data.transform import AssignMultilabel, AssignLabel
from dh_segment_torch.models.model import Model
from dh_segment_torch.training.checkpoint import Checkpoint
from dh_segment_torch.training.early_stopping import EarlyStopping
from dh_segment_torch.training.metrics.metric import Metric
from dh_segment_torch.training.checkpoint import BestCheckpoint, IterationCheckpoint
from dh_segment_torch.training.metrics.metric_tracker import MetricTracker
from dh_segment_torch.training.optimizers import Optimizer, AdamOptimizer
from dh_segment_torch.training.regularizers import Regularizer
from dh_segment_torch.training.schedulers import Scheduler, ReduceOnPlateauScheduler
from dh_segment_torch.utils.ops import batch_items, move_batch

logger = logging.getLogger(__name__)


class Trainer(Registrable):
    default_implementation = "default"
    def __init__(
        self,
        train_loader: data.DataLoader,
        model: Model,
        optimizer: torch.optim.Optimizer,
        val_loader: Optional[data.DataLoader] = None,
        val_metric_tracker: Optional[MetricTracker] = None,
        lr_scheduler: Optional[Scheduler] = None,
        regularizer: Optional[Regularizer] = None,
        early_stopping: Optional[EarlyStopping] = None,
        train_checkpoint: Optional[Checkpoint] = None,
        val_checkpoint: Optional[BestCheckpoint] = None,
        num_epochs: int = 20,
        evaluate_every_epoch: int = 10,
        num_accumulation_steps: int = 1,
        track_train_metrics: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):
        self.train_loader = train_loader
        self.model = model
        self.optimizer = optimizer
        self.val_loader = val_loader
        self.val_metric_tracker = val_metric_tracker
        self.lr_scheduler = lr_scheduler
        self.regularizer = regularizer
        self.early_stopping = early_stopping
        self.train_checkpoint = train_checkpoint
        self.val_checkpoint = val_checkpoint
        self.num_epochs = num_epochs
        self.evaluate_every_epoch = evaluate_every_epoch
        self.num_accumulation_steps = num_accumulation_steps
        self.track_train_metrics = track_train_metrics
        self.device = device

        self.iteration = 0
        self.epoch = 0

    def train(self):
        pbar = tqdm(range(self.num_epochs), desc=f"")
        for epochs in batch_items(
            range(1, self.num_epochs + 1), self.evaluate_every_epoch
        ):
            for epoch in epochs:
                self.epoch = epoch
                metrics = self.train_epoch()
                pbar.set_description(f"epoch {self.epoch}: loss={metrics['loss']:.5f}")
                pbar.refresh()
                pbar.update(1)
            self.validate()
            if self.should_terminate:
                logger.info("Reached an early stop threshold, stopping early.")
                break

    def train_epoch(self):
        pbar = tqdm(desc=f"", leave=True)
        train_loss = 0.0
        train_reg_loss = 0.0
        iterations_this_epoch = 0

        self.optimizer.zero_grad()

        for batches in batch_items(self.train_loader, self.num_accumulation_steps):
            self.iteration += 1
            iterations_this_epoch += 1
            for batch in batches:
                result = self.train_step(batch)
                train_loss += result["loss"].item()
                train_reg_loss += result["reg_loss"].item()

            # Optimizer + scheduler
            self.optimizer.step()
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, ReduceOnPlateauScheduler):
                    self.lr_scheduler.step(self.val_metric_tracker.last_value)
                else:
                    self.lr_scheduler.step()

            # Logging
            metrics = self.get_metrics(
                train_loss, train_reg_loss, iterations_this_epoch, is_train=True
            )
            pbar.set_description(f"iter {self.iteration}: loss={metrics['loss']:.5f}")
            pbar.refresh()
            pbar.update(1)
            # TODO add logger

            # Checkpoint
            if self.train_checkpoint:
                self.train_checkpoint.maybe_save({})

        pbar.close()

        metrics = self.get_metrics(
            train_loss, train_reg_loss, iterations_this_epoch, is_train=True, reset=True
        )

        return metrics

    def train_step(self, batch: Dict[str, torch.Tensor]):
        self.model.train()
        result = self.apply_model_to_batch(batch, track_metrics=self.track_train_metrics)
        result['loss'].backward()
        return result

    def validate(self):
        if self.val_loader is not None:
            val_loss = 0.0
            val_reg_loss = 0.0
            num_iterations = 0

            for batch in self.val_loader:
                num_iterations += 1
                result = self.val_step(batch)
                val_loss += result["loss"].item()
                val_reg_loss += result["reg_loss"].item()

            metrics = self.get_metrics(
                val_loss, val_reg_loss, num_iterations, reset=True
            )
            self.val_metric_tracker.update(metrics)
            if self.val_checkpoint:
                self.val_checkpoint.maybe_save({})
            # TODO log metrics here
            print("val", metrics)

    def val_step(self, batch: Dict[str, torch.Tensor]):
        with torch.no_grad():
            self.model.eval()
            return self.apply_model_to_batch(batch, track_metrics=True)

    def apply_model_to_batch(
        self, batch: Dict[str, torch.Tensor], track_metrics: bool = False
    ):
        batch = move_batch(batch, self.device, non_blocking=True)
        result = self.model(**batch, track_metrics=track_metrics)
        self.apply_regularization(result)
        return result

    def apply_regularization(self, result: Dict[str, torch.Tensor]):
        if self.regularizer:
            penalty = self.regularizer.get_penalty()
            result["loss"] += penalty
            result["reg_loss"] = penalty
        else:
            result["reg_loss"] = torch.zeros_like(result["loss"])

    def get_metrics(
        self,
        loss: float,
        reg_loss: float,
        num_iterations: int,
        is_train: bool = False,
        reset: bool = False,
    ):
        if is_train and not self.track_train_metrics:
            metrics = {}
        else:
            metrics = self.model.get_metrics(reset)
        metrics["loss"] = float(loss / num_iterations) if num_iterations > 0 else 0.0
        metrics["reg_loss"] = (
            float(reg_loss / num_iterations) if num_iterations > 0 else 0.0
        )
        return metrics

    @property
    def should_terminate(self):
        return self.early_stopping and self.early_stopping.should_terminate()

    @classmethod
    def from_partial(
        cls,
        color_labels: ColorLabels,
        train_dataset: Lazy[Dataset],
        model: Lazy[Model],
        optimizer: Optional[Lazy[Optimizer]] = None,
        metrics: Optional[
            Union[Dict[str, Lazy[Metric]], List[Lazy[Metric]], Lazy[Metric]]
        ] = None,
        train_loader: Optional[Lazy[DataLoader]] = None,
        val_dataset: Optional[Lazy[Dataset]] = None,
        val_loader: Optional[Lazy[DataLoader]] = None,
        lr_scheduler: Optional[Lazy[Scheduler]] = None,
        regularizer: Optional[Lazy[Regularizer]] = None,
        early_stopping: Optional[Lazy[EarlyStopping]] = None,
        train_checkpoint: Optional[Lazy[Checkpoint]] = None,
        val_checkpoint: Optional[Lazy[BestCheckpoint]] = None,  # TODO check if can do
        val_metric_tracker: Optional[Lazy[MetricTracker]] = None,
        val_metric: str = "-loss",
        batch_size: int = 8,
        shuffle_train: bool = True,
        num_data_workers: int = 4,
        ignore_padding: bool = False,
        training_margin: int = 16,
        num_epochs: int = 20,
        evaluate_every_epoch: int = 10,
        num_accumulation_steps: int = 1,
        model_out_dir: str = "./model",
        track_train_metrics: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ):

        if color_labels.multilabel:
            assign_transform = AssignMultilabel(color_labels.color_labels, color_labels.one_hot_labels)
        else:
            assign_transform = AssignLabel(color_labels.color_labels)

        train_dataset = train_dataset.construct(assign_transform=assign_transform)

        # Data
        if isinstance(train_dataset, PatchesDataset):
            train_dataset.set_shuffle(shuffle_train)

        if train_loader:
            train_loader = train_loader.construct(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=num_data_workers,
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=num_data_workers,
                shuffle=True,
            )

        if val_dataset:
            val_dataset = val_dataset.construct(assign_transform=assign_transform)
            if val_loader:
                val_loader = val_loader.construct(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_data_workers,
                )
            else:
                val_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    num_workers=num_data_workers,
                    shuffle=False,
                )

        model = model.construct(
            num_classes=color_labels.num_classes,
            metrics=metrics,
            multilabel=color_labels.multilabel,
            ignore_padding=ignore_padding,
            margin=training_margin,
        )

        parameters = [tuple([n, p])for n, p in model.named_parameters() if p.requires_grad]

        if optimizer:
            optimizer = optimizer.construct(model_params=parameters)
        else:
            optimizer = AdamOptimizer(model_params=parameters)

        if regularizer:
            regularizer = regularizer.construct(model_params=parameters)

        if lr_scheduler:
            lr_scheduler = lr_scheduler.construct(optimizer=optimizer)

        if train_checkpoint:
            train_checkpoint = train_checkpoint.construct(
                checkpoint_dir=model_out_dir, prefix="train"
            )
        else:
            train_checkpoint = IterationCheckpoint(
                checkpoint_dir=model_out_dir, prefix="train"
            )

        if val_dataset is None:
            if val_metric_tracker:
                raise ValueError("Cannot track a metric without a validation set.")
            if early_stopping:
                raise ValueError("Cannot early stop without a validation set.")
            if val_checkpoint:
                raise ValueError(
                    "Cannot checkpoint on best score without a validation set."
                )
            if isinstance(lr_scheduler, ReduceOnPlateauScheduler):
                raise ValueError(
                    "Cannot use reduce on plateau without a validation set."
                )
        else:
            if val_metric_tracker:
                val_metric_tracker = val_metric_tracker.construct(
                    metric_name=val_metric
                )
            else:
                val_metric_tracker = MetricTracker(val_metric)
            if val_metric_tracker.metric_name not in model.get_available_metrics().union(
                {"loss"}
            ):
                raise ValueError(
                    "Validation metric is not in the available metrics (or loss)."
                )
            if early_stopping:
                early_stopping = early_stopping.construct(tracker=val_metric_tracker)
            if val_checkpoint:
                val_checkpoint = val_checkpoint.construct(
                    tracker=val_metric_tracker, checkpoint_dir=model_out_dir
                )
            else:
                val_checkpoint = BestCheckpoint(
                    tracker=val_metric_tracker, checkpoint_dir=model_out_dir
                )
        return cls(
            train_loader,
            model,
            optimizer,
            val_loader,
            val_metric_tracker,
            lr_scheduler,
            regularizer,
            early_stopping,
            train_checkpoint,
            val_checkpoint,
            num_epochs,
            evaluate_every_epoch,
            num_accumulation_steps,
            track_train_metrics,
            device,
        )


Trainer.register("default", "from_partial")(Trainer)


def should_run(iteration: int, every: int):
    return iteration >= every and iteration % every == 0
