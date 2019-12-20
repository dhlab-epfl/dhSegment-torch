import importlib
import os
from typing import Tuple, Any
from glob import glob

import torch
import torch.nn as nn
from ignite.contrib.engines import common
from ignite.contrib.engines.common import get_default_score_fn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import mIoU, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.optim import Adam, lr_scheduler as lrs
from torch.utils.data import DataLoader

from .tensorboard_logger import LogImagesHandler
from .utils import patch_loss_with_padding, patch_metric_with_padding
from ..data.input_dataset import get_dataset, collate_fn
from ..data.transforms import make_transforms, make_eval_transforms
from ..network import SegmentationModel
from ..params import TrainingParams, DataParams, ModelParams, PredictionType


def get_train_val_loaders(data_params: DataParams, training_params: TrainingParams) -> Tuple[DataLoader, DataLoader]:
    train_transforms = make_transforms(data_params)
    train_dataset = get_dataset(data_params.train_data, train_transforms)
    train_loader = DataLoader(train_dataset, training_params.batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=training_params.num_data_workers, pin_memory=training_params.pin_memory)

    val_transforms = make_eval_transforms(data_params)
    val_dataset = get_dataset(data_params.validation_data, val_transforms)
    val_loader = DataLoader(val_dataset, training_params.batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=training_params.num_data_workers, pin_memory=training_params.pin_memory)

    return train_loader, val_loader


def create_model(model_params: ModelParams, data_params: DataParams) -> nn.Module:
    encoder = get_class_from_name(model_params.encoder_network)(
        pretrained=model_params.pretraining,
        **model_params.encoder_params)

    decoder = get_class_from_name(model_params.decoder_network)(
        encoder_channels=encoder.output_dims,
        n_classes=data_params.n_classes,
        **model_params.decoder_params)

    model = SegmentationModel(encoder, decoder)

    return model


def train(model_params: ModelParams, training_params: TrainingParams, data_params: DataParams) -> Engine:

    model = create_model(model_params, data_params)

    train_loader, val_loader = get_train_val_loaders(data_params, training_params)

    optimizer = Adam(model.parameters(), lr=training_params.learning_rate)

    if training_params.exponential_learning:
        lr_scheduler = lrs.LambdaLR(optimizer, lambda epoch: 0.95**(epoch / 200)) # TODO hardcoded exponential decay
    else:
        lr_scheduler = lrs.LambdaLR(optimizer, lambda epoch: 1.0)


    if data_params.prediction_type == PredictionType.CLASSIFICATION:
        criterion_class = nn.CrossEntropyLoss
    elif data_params.prediction_type == PredictionType.MULTILABEL:
        criterion_class = nn.BCEWithLogitsLoss
    else:
        raise ValueError("Prediction type does not have a defined loss")

    criterion = patch_loss_with_padding(criterion_class, margin=training_params.training_margin)()

    model.to(training_params.device)
    criterion.to(training_params.device)

    def train_update_fn(engine, batch):
        model.train()
        x, y = prepare_batch(batch, device=training_params.device, non_blocking=training_params.non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        output = {'loss': loss.item()}

        if engine.state.iteration % training_params.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return output

    trainer = Engine(train_update_fn)

    if training_params.resume_training:
        latest_checkpoint = find_latest_checkpoint(training_params.model_out_dir)
        if latest_checkpoint is not None:
            checkpoint_data = torch.load(latest_checkpoint)
            model.load_state_dict(checkpoint_data['model'])
            optimizer.load_state_dict(checkpoint_data['optimizer'])
            lr_scheduler.load_state_dict(checkpoint_data['scheduler'])

            resume_iterations = checkpoint_data['scheduler']['last_epoch']

            @trainer.on(Events.STARTED)
            def resume_training(engine):
                resume_epoch = resume_iterations // len(engine.state.dataloader)
                engine.state.iteration = resume_iterations
                engine.state.epoch = resume_epoch


    common.setup_common_training_handlers(trainer,
                                          output_path=training_params.model_out_dir,
                                          lr_scheduler=lr_scheduler,
                                          with_gpu_stats=can_gpu_info(),
                                          output_names=['loss'],
                                          with_pbars=True,
                                          with_pbar_on_iters=True,
                                          log_every_iters=1
                                          )

    checkpoint_handler = ModelCheckpoint(dirname=training_params.model_out_dir, filename_prefix='training',
                                         require_empty=not training_params.resume_training)
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=training_params.train_checkpoint_interval * len(train_loader)),
        checkpoint_handler, {'model': model, 'optimizer': optimizer, 'scheduler': lr_scheduler})

    cm_patched = patch_metric_with_padding(ConfusionMatrix,
                                          margin=training_params.training_margin)

    cm_metric = cm_patched(data_params.n_classes, output_transform=metric_output_transform)

    val_metrics = {
        'IoU': IoU(cm_metric),
        'mIoU': mIoU(cm_metric)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                            device=training_params.device,
                                            non_blocking=training_params.non_blocking,
                                            prepare_batch=prepare_batch,
                                            output_transform=lambda x, y, y_pred: (x, y, y_pred))

    ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=training_params.evaluate_every_epoch),
                              lambda _ : evaluator.run(val_loader))

    score_metric_name = 'mIoU'

    if training_params.early_stopping_patience is not None:
        common.add_early_stopping_by_val_score(training_params.early_stopping_patience,
                                               evaluator, trainer, metric_name=score_metric_name)
    tb_logger = common.setup_tb_logging(training_params.tensorboard_log_dir, trainer, optimizer,
                                        evaluators={'validation': evaluator}, log_every_iters=2)

    tb_logger.attach(evaluator,
                     log_handler=LogImagesHandler(data_params.color_codes,
                                                  one_large_image=False,
                                                  max_images=4,
                                                  global_step_engine=trainer),

                     event_name=Events.EPOCH_COMPLETED
                     )

    save_best_model_by_val_score(training_params.model_out_dir, evaluator, model,
                                 metric_name=score_metric_name, trainer=trainer,
                                 require_empty=not training_params.resume_training)

    trainer.run(train_loader, max_epochs=training_params.n_epochs)


def prepare_batch(batch, device=None, non_blocking=False):
    return (convert_tensor(batch['images'], device=device, non_blocking=non_blocking),
            (convert_tensor(batch['labels'], device=device, non_blocking=non_blocking),
             convert_tensor(batch['shapes'], device=device, non_blocking=non_blocking)))


def metric_output_transform(output):
    x,y,y_pred = output
    return y_pred, y


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
    module_name, class_name = full_class_name.rsplit('.', maxsplit=1)
    # load the module, will raise ImportError if module cannot be loaded
    m = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    c = getattr(m, class_name)
    return c


def can_gpu_info():
    return torch.cuda.is_available() and importlib.util.find_spec('pynvml') is not None

def find_latest_checkpoint(model_dir: str):
    checkpoints = glob(os.path.join(model_dir, 'training_checkpoint*.pth'))
    if len(checkpoints) > 0:
        return max(checkpoints, key=lambda x: int(x.split('_')[-1].replace('.pth', '')))

def save_best_model_by_val_score(output_path, evaluator, model, metric_name, n_saved=3, trainer=None, tag="val", require_empty=True):
    """Method adds a handler to `evaluator` to save best models based on the score (named by `metric_name`)
        provided by `evaluator`.
        Args:
            output_path (str): output path to indicate where to save best models
            evaluator (Engine): evaluation engine used to provide the score
            model (nn.Module): model to store
            metric_name (str): metric name to use for score evaluation. This metric should be present in
                `evaluator.state.metrics`.
            n_saved (int, optional): number of best models to store
            trainer (Engine, optional): trainer engine to fetch the epoch when saving the best model.
            tag (str, optional): score name prefix: `{tag}_{metric_name}`. By default, tag is "val".
        """
    global_step_transform = None
    if trainer is not None:
        global_step_transform = global_step_from_engine(trainer)

    best_model_handler = ModelCheckpoint(dirname=output_path,
                                         filename_prefix="best",
                                         n_saved=n_saved,
                                         global_step_transform=global_step_transform,
                                         score_name="{}_{}".format(tag, metric_name.lower()),
                                         score_function=get_default_score_fn(metric_name),
                                         require_empty=require_empty
                                         )
    evaluator.add_event_handler(Events.COMPLETED, best_model_handler, {'model': model, })