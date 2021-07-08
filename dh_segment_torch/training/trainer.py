import logging
from typing import Optional, List, Union, Dict, Tuple, Any

import torch
from torch.utils import data
from tqdm.auto import tqdm
import numpy as np

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.color_labels import ColorLabels
from dh_segment_torch.data.data_loader import DataLoader
from dh_segment_torch.data.datasets import PatchesDataset
from dh_segment_torch.data.datasets.dataset import Dataset
from dh_segment_torch.data.transforms import AssignMultilabel, AssignLabel
from dh_segment_torch.models.model import Model
from dh_segment_torch.nn.initializers import InitializerApplier
from dh_segment_torch.training.checkpoint import BestCheckpoint, IterationCheckpoint
from dh_segment_torch.training.checkpoint import Checkpoint
from dh_segment_torch.training.early_stopping import EarlyStopping
from dh_segment_torch.training.logging.logger import Logger
from dh_segment_torch.metrics.metric import Metric, MetricType
from dh_segment_torch.metrics import MetricTracker
from dh_segment_torch.training.optimizers import Optimizer, AdamOptimizer
from dh_segment_torch.training.regularizers import Regularizer
from dh_segment_torch.training.schedulers import (
    Scheduler,
    ReduceOnPlateauScheduler,
    ConstantScheduler,
)
from dh_segment_torch.training.utils import worker_init_fn
from dh_segment_torch.utils.ops import batch_items, move_batch

logger_console = logging.getLogger(__name__)


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
        loggers: Optional[List[Logger]] = None,
        num_epochs: int = 20,
        evaluate_every_epoch: int = 10,
        num_accumulation_steps: int = 1,
        track_train_metrics: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        reset_early_stopping: bool = True,
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
        if loggers is None:
            loggers = []
        self.loggers = loggers
        self.num_epochs = num_epochs
        self.evaluate_every_epoch = evaluate_every_epoch
        self.num_accumulation_steps = num_accumulation_steps
        self.track_train_metrics = track_train_metrics
        self.device = device
        self.reset_early_stopping = reset_early_stopping

        self.iteration = 0
        self.epoch = 0

    def train(self):
        if self.reset_early_stopping and self.early_stopping:
            self.early_stopping.reset()
            
        self.model = self.model.to(self.device)

        pbar = tqdm(range(self.num_epochs), desc=f"epoch {self.epoch}: loss=???")
        for epochs in batch_items(
            range(self.epoch+1, self.epoch+self.num_epochs + 1), self.evaluate_every_epoch
        ):
            self.model.train()
            for epoch in epochs:
                self.epoch = epoch
                metrics, losses = self.train_epoch()
                pbar.set_description(f"epoch {self.epoch}: loss={losses['loss']:.5f}")
                pbar.refresh()
                pbar.update(1)
            self.validate()
            if isinstance(self.lr_scheduler, ReduceOnPlateauScheduler):
                self.lr_scheduler.step(self.val_metric_tracker.last_value)
            if self.should_terminate:
                logger_console.info("Reached an early stop threshold, stopping early.")
                break
        pbar.close()
        self.final_save()

    def train_epoch(self):
        pbar = tqdm(desc=f"iter={self.iteration}: loss=???", leave=False)
        train_loss = 0.0
        train_reg_loss = 0.0
        iterations_this_epoch = 0

        self.optimizer.zero_grad()

        for batches in batch_items(self.train_loader, self.num_accumulation_steps):
            self.iteration += 1
            iterations_this_epoch += 1
            batch = None
            result = None
            for batch in batches:
                result = self.train_step(batch)
                train_loss += result["loss"].item()
                train_reg_loss += result["reg_loss"].item()

            # Optimizer + scheduler
            self.optimizer.step()
            if not isinstance(self.lr_scheduler, ReduceOnPlateauScheduler):
                self.lr_scheduler.step()

            # Logging
            metrics, losses = self.get_metrics_and_losses(
                train_loss, train_reg_loss, iterations_this_epoch, is_train=True
            )
            pbar.set_description(f"iter {self.iteration}: loss={losses['loss']:.5f}")
            pbar.refresh()
            pbar.update(1)

            for logger in self.loggers:
                logger.log(
                    self.iteration,
                    metrics,
                    losses,
                    batch,
                    result["logits"],
                    self.lr_scheduler,
                    self.optimizer,
                    prefix="train",
                )

            # Checkpoint
            if self.train_checkpoint:
                self.train_checkpoint.maybe_save(self.state_dict())

        pbar.close()

        metrics, losses = self.get_metrics_and_losses(
            train_loss, train_reg_loss, iterations_this_epoch, is_train=True, reset=True
        )

        return metrics, losses

    def train_step(self, batch: Dict[str, torch.Tensor]):
        result = self.apply_model_to_batch(
            batch, track_metrics=self.track_train_metrics
        )
        result["loss"].backward()
        return result

    def validate(self):
        if self.val_loader is not None:
            val_loss = 0.0
            val_reg_loss = 0.0
            num_iterations = 0

            batch = None
            result = None

            pbar = tqdm(desc=f"Evaluating", leave=False)
            with torch.no_grad():
                self.model.eval()
                for batch in self.val_loader:
                    num_iterations += 1
                    result = self.val_step(batch)
                    val_loss += result["loss"].item()
                    val_reg_loss += result["reg_loss"].item()
                    pbar.update()
            pbar.close()

            metrics, losses = self.get_metrics_and_losses(
                val_loss, val_reg_loss, num_iterations, reset=True
            )
            self.val_metric_tracker.update(metrics, losses)
            if self.val_checkpoint:
                self.val_checkpoint.maybe_save(self.state_dict())
            for logger in self.loggers:
                logger.log(
                    self.iteration,
                    metrics,
                    losses,
                    batch,
                    result["logits"],
                    prefix="val",
                    ignore_iters=True,
                )

    def val_step(self, batch: Dict[str, torch.Tensor]):
        return self.apply_model_to_batch(batch, track_metrics=True)

    def apply_model_to_batch(
        self, batch: Dict[str, torch.Tensor], track_metrics: bool = False
    ):
        batch = move_batch(batch, self.device, non_blocking=True)
        result = self.model(**batch, track_metrics=track_metrics)
        if 'loss' in result:
            self.apply_regularization(result)
        return result

    def apply_regularization(self, result: Dict[str, torch.Tensor]):
        if self.regularizer:
            penalty = self.regularizer.get_penalty()
            result["loss"] += penalty
            result["reg_loss"] = penalty
        else:
            result["reg_loss"] = torch.zeros_like(result["loss"])

    def get_metrics_and_losses(
        self,
        loss: float,
        reg_loss: float,
        num_iterations: int,
        is_train: bool = False,
        reset: bool = False,
    ) -> Tuple[Dict[str, MetricType], Dict[str, float]]:
        if is_train and not self.track_train_metrics:
            metrics: Dict[str, MetricType] = {}
        else:
            metrics = self.model.get_metrics(reset)
        losses = {}
        losses["loss"] = float(loss / num_iterations) if num_iterations > 0 else 0.0
        if self.regularizer:
            losses["reg_loss"] = (
                float(reg_loss / num_iterations) if num_iterations > 0 else 0.0
            )
        return metrics, losses

    def final_save(self):
        self.train_checkpoint.save(self.state_dict(), permanent=True)

    def state_dict(self):
        state_dict = {}
        for key, value in self.__dict__.items():
            if key in {
                "model",
                "optimizer",
                "lr_scheduler",
                "regularizer",
                "val_metric_tracker",
                "early_stopping",
                "train_checkpoint",
                "val_checkpoint",
            }:
                state_dict[key] = state_dict_or_none(value)
            elif key not in {"train_loader", "val_loader", "loggers"}:
                state_dict[key] = value
        return state_dict

    def load_state_dict(self, state_dict):
        for key, value in state_dict.items():
            if key in {
                "model",
                "optimizer",
                "lr_scheduler",
                "regularizer",
                "val_metric_tracker",
                "early_stopping",
                "train_checkpoint",
                "val_checkpoint",
            }:
                load_state_dict_not_none(self.__dict__.get(key, None), value)
            elif key not in {"train_loader", "val_loader", "loggers"}:
                self.__dict__[key] = value
        
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
            Union[
                Dict[str, Lazy[Metric]],
                List[Union[Tuple[str, Lazy[Metric]], Lazy[Metric]]],
                Lazy[Metric],
            ]
        ] = None,
        train_loader: Optional[Lazy[DataLoader]] = None,
        val_dataset: Optional[Lazy[Dataset]] = None,
        val_loader: Optional[Lazy[DataLoader]] = None,
        lr_scheduler: Optional[Lazy[Scheduler]] = None,
        regularizer: Optional[Lazy[Regularizer]] = None,
        initializer: Optional[InitializerApplier] = None,
        early_stopping: Optional[Lazy[EarlyStopping]] = None,
        train_checkpoint: Optional[Lazy[Checkpoint]] = None,
        val_checkpoint: Optional[Lazy[BestCheckpoint]] = None,  # TODO check if can do
        val_metric_tracker: Optional[Lazy[MetricTracker]] = None,
        loggers: Optional[Union[List[Lazy[Logger]], Lazy[Logger]]] = None,
        val_metric: str = "-loss",
        batch_size: int = 8,
        shuffle_train: bool = True,
        num_data_workers: int = 4,
        ignore_padding: bool = False,
        training_margin: int = 0,
        num_epochs: int = 20,
        evaluate_every_epoch: int = 10,
        num_accumulation_steps: int = 1,
        model_out_dir: str = "./model",
        track_train_metrics: bool = False,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        reset_early_stopping: bool = True,
        exp_name: str = "dhSegment_experiment",
        config: Optional[Dict[str, Any]] = None,
    ):

        if color_labels.multilabel:
            assign_transform = AssignMultilabel(
                color_labels.colors, color_labels.one_hot_encoding
            )
        else:
            assign_transform = AssignLabel(color_labels.colors)

        train_dataset = train_dataset.construct(assign_transform=assign_transform)

        is_patches = isinstance(train_dataset, PatchesDataset)
        # Data
        if is_patches:
            train_dataset.set_shuffle(shuffle_train)

        if train_loader:
            train_loader = train_loader.construct(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=min(num_data_workers, train_dataset.num_images),
                shuffle=True and not is_patches,
                worker_init_fn=worker_init_fn,
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                num_workers=min(num_data_workers, train_dataset.num_images),
                shuffle=True and not is_patches,
                worker_init_fn=worker_init_fn,
            )

        if val_dataset:
            val_dataset = val_dataset.construct(assign_transform=assign_transform)
            if val_loader:
                val_loader = val_loader.construct(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    num_workers=min(num_data_workers, val_dataset.num_images),
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                )
            else:
                val_loader = DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    num_workers=min(num_data_workers, val_dataset.num_images),
                    shuffle=False,
                    worker_init_fn=worker_init_fn,
                )

        model = model.construct(
            num_classes=color_labels.num_classes,
            metrics=metrics,
            multilabel=color_labels.multilabel,
            classes_labels=color_labels.labels,
            ignore_padding=ignore_padding,
            margin=training_margin,
        )

        parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        if initializer:
            initializer.apply(parameters)

        if optimizer:
            optimizer = optimizer.construct(model_params=parameters)
        else:
            optimizer = AdamOptimizer(model_params=parameters)

        if regularizer:
            regularizer = regularizer.construct(model_params=parameters)

        if lr_scheduler:
            lr_scheduler = lr_scheduler.construct(optimizer=optimizer)
        else:
            lr_scheduler = ConstantScheduler(optimizer)

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

        if loggers is not None:
            built_loggers = []
            if not isinstance(loggers, List):
                loggers = [loggers]
            for logger in loggers:
                built_loggers.append(
                    logger.construct(
                        color_labels=color_labels,
                        ignore_padding=ignore_padding,
                        margin=training_margin,
                        exp_name=exp_name,
                        config=config,
                    )
                )
        else:
            built_loggers = None

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
            built_loggers,
            num_epochs,
            evaluate_every_epoch,
            num_accumulation_steps,
            track_train_metrics,
            device,
            reset_early_stopping,
        )


Trainer.register("default", "from_partial")(Trainer)


def state_dict_or_none(obj):
    if obj is not None:
        return obj.state_dict()
    return None


def load_state_dict_not_none(obj, state_dict):
    if obj is not None:
        obj.load_state_dict(state_dict)

