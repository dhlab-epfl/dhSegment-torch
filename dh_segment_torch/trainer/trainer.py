import importlib
import os
from functools import partial
from glob import glob
from typing import Tuple, Any

import torch
import torch.nn as nn
from ignite.utils import convert_tensor
from torch.optim import lr_scheduler as lrs, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm

from .checkpoint import Checkpoint, save_name_to_iter
from .metrics import (
    ConfusionMatrix,
    mIoU,
    AverageAccuracy,
    AverageLoss,
    IoU,
    LearningRate,
)
from .tensorboard import TensorboardLogMetrics, TensorboardLogImages
from .utils import (
    patch_loss_with_padding,
    patch_metric_with_padding,
    WeightedBCEWithLogitsLoss,
    should_run,
)
from ..data.input_dataset import get_dataset, collate_fn, patches_worker_init_fn
from ..data.input_patches import get_patches_dataset
from ..data.transforms import (
    make_transforms,
    make_eval_transforms,
    make_global_transforms,
    make_local_transforms,
)
from ..network import SegmentationModel
from ..params import TrainingParams, DataParams, ModelParams, PredictionType


class Trainer:
    def __init__(
        self,
        model_params: ModelParams,
        data_params: DataParams,
        training_params: TrainingParams,
    ):
        self.model_params = model_params
        self.data_params = data_params
        self.training_params = training_params

        self.model_dir = training_params.model_out_dir

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.training_params.tensorboard_log_dir, exist_ok=True)

        self.model = self.get_model()
        self.model.to(training_params.device)

        train_loader, val_loader = self.get_train_val_loaders()
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = self.get_optimizer()
        self.lr_scheduler = self.get_lr_scheduler()

        self.criterion = self.get_criterion()
        self.criterion.to(training_params.device)

        self.preparch_batch_fn = self.get_prepare_batch_fn()

        self.iteration = 0

        self.accumulation_steps = self.training_params.accumulation_steps

        self.epoch = 0
        self.max_epochs = self.training_params.n_epochs

        is_multilabel = data_params.prediction_type == PredictionType.MULTILABEL

        self.train_loss = AverageLoss()

        lr_metric = LearningRate(self.lr_scheduler)

        self.train_metrics = [self.train_loss, lr_metric]

        confusion_matrix = patch_metric_with_padding(
            ConfusionMatrix, self.training_params.training_margin
        )(data_params.n_classes, is_multilabel)
        iou = IoU(confusion_matrix)
        miou = mIoU(confusion_matrix)
        acc = AverageAccuracy(confusion_matrix)

        metrics_names = ["Loss", "iou", "mIoU", "Accuracy"]
        val_loss = AverageLoss()
        self.val_metrics = [val_loss, confusion_matrix]

        self.tensorboard_writer = SummaryWriter(training_params.tensorboard_log_dir)

        self.tensorboard_train_metrics = TensorboardLogMetrics(
            self.tensorboard_writer,
            self.train_metrics,
            prefix="Train",
            metrics_names=["Loss", "LR"],
            log_every=50,
        )

        self.tensorboard_train_images = TensorboardLogImages(
            self.tensorboard_writer,
            data_params.color_codes,
            data_params.onehot_labels,
            training_params.training_margin,
            is_multilabel,
            prefix="Train",
            log_every=200,
        )

        self.tensorboard_val_metrics = TensorboardLogMetrics(
            self.tensorboard_writer,
            [val_loss, iou, miou, acc],
            metrics_names=metrics_names,
            prefix="Val",
            log_every=1,
        )

        self.tensorboard_val_images = TensorboardLogImages(
            self.tensorboard_writer,
            data_params.color_codes,
            data_params.onehot_labels,
            training_params.training_margin,
            is_multilabel,
            prefix="Val",
            log_every=1,
        )

        self.trainer_checkpoint = Checkpoint(
            self.model_dir,
            prefix="model",
            save_dict={
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": self.lr_scheduler,
                "iteration": self.get_iteration,
                "epoch": self.get_epoch,
            },
            max_checkpoints=5,
            save_every=500,
        )

        self.best_checkpoint = Checkpoint(
            self.model_dir,
            prefix="best",
            save_dict={"model": self.model},
            metric=miou,
            max_checkpoints=2,
        )

        if training_params.resume_training:
            self.restore_training()
        else:
            if len(glob(os.path.join(self.model_dir, "model*.pth"))) > 0:
                raise ValueError(
                    "Model dir contained saved models but did not want to restore"
                )

    def train(self):
        for epoch in tqdm(range(self.max_epochs)):
            self.train_epoch()
            if should_run(epoch, self.training_params.evaluate_every_epoch):
                self.validate()
        self.validate()

    def train_epoch(self):
        self.epoch += 1
        pbar = tqdm(desc=f"", leave=False)
        for batch in self.train_loader:
            x, y, y_pred, loss = self.train_step(batch)
            self.lr_scheduler.step()
            self.update_metrics(self.train_metrics, y_pred, y, loss)
            self.trainer_checkpoint.save(self.iteration)
            pbar.set_description(
                f"iter {self.iteration}: loss={self.train_loss.value:.5f}"
            )
            pbar.refresh()
            pbar.update()
            self.tensorboard_train_metrics.log(self.iteration, reset=True)
            self.tensorboard_train_images.log(self.iteration, x, y, y_pred)
        self.reset_metrics(self.train_metrics)
        pbar.close()

    def train_step(self, batch):
        self.iteration += 1
        self.model.train()
        x, y = self.preparch_batch_fn(batch)
        y_pred = self.model(x)
        loss = self.criterion(y_pred, y)
        loss.backward()

        if self.iteration % self.accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return x, y, y_pred, loss.item()

    def validate(self):
        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            x, y, y_pred, loss = self.val_step(batch)
            self.update_metrics(self.val_metrics, y_pred, y, loss)
        self.best_checkpoint.save(self.iteration)
        self.tensorboard_val_metrics.log(self.iteration)
        self.tensorboard_val_images.log(self.iteration, x, y, y_pred)
        self.reset_metrics(self.val_metrics)

    def val_step(self, batch):
        with torch.no_grad():
            self.model.eval()
            x, y = self.preparch_batch_fn(batch)
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)

            return x, y, y_pred, loss.item()

    def restore_training(self):
        all_savepoints = glob(os.path.join(self.model_dir, "model*.pth"))
        if len(all_savepoints) > 0:
            latest_checkpoint = sorted(all_savepoints, key=save_name_to_iter)[-1]
            checkpoint_data = torch.load(latest_checkpoint)
            self.model.load_state_dict(checkpoint_data["model"])
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint_data["scheduler"])
            self.iteration = checkpoint_data["iteration"]
            self.epoch = checkpoint_data["epoch"]

    def get_model(self) -> nn.Module:
        encoder = get_class_from_name(self.model_params.encoder_network)(
            pretrained=self.model_params.pretraining, **self.model_params.encoder_params
        )

        decoder = get_class_from_name(self.model_params.decoder_network)(
            encoder_channels=encoder.output_dims,
            n_classes=self.data_params.n_classes,
            **self.model_params.decoder_params,
        )

        model = SegmentationModel(encoder, decoder)

        return model

    def get_train_val_loaders(self) -> Tuple[DataLoader, DataLoader]:
        if self.data_params.make_patches:
            return get_patches_train_val_loaders(self.data_params, self.training_params)
        else:
            return get_dataset_train_val_loaders(self.data_params, self.training_params)

    def get_optimizer(self):
        return Adam(
            self.model.parameters(),
            lr=self.training_params.learning_rate,
            weight_decay=self.training_params.weight_decay,
        )

    def get_lr_scheduler(self):
        if self.training_params.exponential_learning:
            return lrs.LambdaLR(
                self.optimizer, lambda epoch: 0.95 ** (epoch / 200)
            )  # TODO hardcoded exponential decay
        else:
            return lrs.LambdaLR(self.optimizer, lambda epoch: 1.0)

    def get_criterion(self):
        if self.training_params.weights_labels is not None:
            weight_labels = torch.tensor(
                self.training_params.weights_labels, dtype=torch.float32
            )
        else:
            weight_labels = torch.tensor(
                [1.0] * self.data_params.n_classes, dtype=torch.float32
            )
        if self.data_params.prediction_type == PredictionType.CLASSIFICATION:
            criterion_class = nn.CrossEntropyLoss
        elif self.data_params.prediction_type == PredictionType.MULTILABEL:
            criterion_class = WeightedBCEWithLogitsLoss
        else:
            raise ValueError("Prediction type does not have a defined loss")

        return patch_loss_with_padding(
            criterion_class, margin=self.training_params.training_margin
        )(weight=weight_labels)

    def get_prepare_batch_fn(self):
        def prepare_batch(batch):
            convert_tensor_fn = partial(
                convert_tensor,
                device=self.training_params.device,
                non_blocking=self.training_params.non_blocking,
            )
            return (
                convert_tensor_fn(batch["images"]),
                (
                    convert_tensor_fn(batch["labels"]),
                    convert_tensor_fn(batch["shapes"]),
                ),
            )

        return prepare_batch

    def get_iteration(self):
        return self.iteration

    def get_epoch(self):
        return self.epoch

    @staticmethod
    def update_metrics(metrics, y_pred, y, loss):
        for metric in metrics:
            metric.update((y_pred, y, loss))

    @staticmethod
    def reset_metrics(metrics):
        for metric in metrics:
            metric.reset()


def set_seed(seed: int):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_class_from_name(full_class_name: str) -> Any:
    """
    Tries to load the class from its naming, will import the corresponding module.
    Raises an Error if it does not work.
    :param full_class_name: full name of the class, for instance `foo.bar.Baz`
    :return: the loaded class
    """
    module_name, class_name = full_class_name.rsplit(".", maxsplit=1)
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def get_dataset_train_val_loaders(
    data_params: DataParams, training_params: TrainingParams
) -> Tuple[DataLoader, DataLoader]:
    train_transforms = make_transforms(data_params)
    train_dataset = get_dataset(data_params.train_data, train_transforms)
    train_num_images = len(train_dataset.dataframe)
    train_loader = DataLoader(
        train_dataset,
        training_params.batch_size,
        shuffle=True,
        drop_last=training_params.drop_last_batch,
        collate_fn=collate_fn,
        num_workers=min(train_num_images, training_params.num_data_workers),
        pin_memory=training_params.pin_memory,
    )

    val_transforms = make_eval_transforms(data_params)
    val_dataset = get_dataset(data_params.validation_data, val_transforms)
    val_num_images = len(val_dataset.dataframe)
    val_loader = DataLoader(
        val_dataset,
        training_params.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=min(val_num_images, training_params.num_data_workers),
        pin_memory=training_params.pin_memory,
    )

    return train_loader, val_loader


def get_patches_train_val_loaders(
    data_params: DataParams, training_params: TrainingParams
) -> Tuple[DataLoader, DataLoader]:
    train_pre_transforms = make_global_transforms(data_params)
    train_post_transforms = make_local_transforms(data_params)
    train_dataset = get_patches_dataset(
        data_params.train_data,
        train_pre_transforms,
        train_post_transforms,
        patch_size=data_params.patch_shape,
        batch_size=training_params.batch_size,
        shuffle=True,
        prefetch_shuffle=training_params.patches_images_buffer_size,
        drop_last=training_params.drop_last_batch,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=training_params.num_data_workers,
        pin_memory=training_params.pin_memory,
        worker_init_fn=patches_worker_init_fn,
    )

    val_pre_transforms = make_global_transforms(data_params, eval=True)
    val_post_transforms = make_local_transforms(data_params, eval=True)

    val_dataset = get_patches_dataset(
        data_params.validation_data,
        val_pre_transforms,
        val_post_transforms,
        patch_size=data_params.patch_shape,
        batch_size=training_params.batch_size,
        shuffle=False,
        prefetch_shuffle=1,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=training_params.num_data_workers,
        pin_memory=training_params.pin_memory,
        worker_init_fn=patches_worker_init_fn,
    )
    return train_loader, val_loader
