import importlib
from typing import Tuple, Any

import torch.nn as nn
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, create_supervised_evaluator, Events
from ignite.metrics import mIoU, ConfusionMatrix, IoU
from ignite.utils import convert_tensor
from torch.optim import Adam, lr_scheduler as lrs
from torch.utils.data import DataLoader

from .tensorboard_logger import LogImagesHandler
from .utils import patch_loss_with_padding, patch_metric_with_padding
from ..network import SegmentationModel, PredictionType
from ..params import TrainingParams, DataParams, ModelParams


def get_train_val_loaders(data_params: DataParams) -> Tuple[DataLoader, DataLoader]:
    pass


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

    train_loader, val_loader = get_train_val_loaders(data_params)

    optimizer = Adam(model.parameters(), lr=training_params.learning_rate)

    if data_params.prediction_type == PredictionType.CLASSIFICATION:
        criterion_class = nn.CrossEntropyLoss
    elif data_params.prediction_type == PredictionType.MULTILABEL:
        criterion_class = nn.BCEWithLogitsLoss
    else:
        raise ValueError("Prediction type does not have a defined loss")

    criterion = patch_loss_with_padding(criterion_class, margin=training_params.training_margin)()

    model.to(training_params.device)
    criterion.to(training_params.device)

    if training_params.exponential_learning:
        lr_scheduler = lrs.ExponentialLR(optimizer, gamma=0.95)
    else:
        lr_scheduler = lrs.LambdaLR(optimizer, lambda epoch: 1.0)

    # TODO restore from checkpoint, not yet implemeted, should wait on pytorch/ignite/pull/640

    def train_update_fn(engine, batch):
        x, y = prepare_batch(batch, training_params.device, training_params.non_blocking)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()

        output = {'loss': loss.item()}

        if engine.state.iteration % training_params.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        return output

    trainer = Engine(train_update_fn)

    common.setup_common_training_handlers(trainer,
                                          to_save={'model': model, 'optimizer': optimizer},
                                          save_every_iters=1000,  # TODO check tensorflow defaults
                                          output_path=training_params.model_out_dir,
                                          lr_scheduler=lr_scheduler,
                                          with_gpu_stats=True,
                                          output_names=['loss'],
                                          with_pbars=True,
                                          with_pbar_on_iters=True,
                                          log_every_iters=1
                                          )

    cm_metric = patch_metric_with_padding(ConfusionMatrix,
                                          margin=training_params.training_margin)(data_params.n_classes)

    val_metrics = {
        'IoU': IoU(cm_metric),
        'mIoU': mIoU(cm_metric)
    }

    evaluator = create_supervised_evaluator(model, metrics=val_metrics,
                                            device=training_params.device,
                                            non_blocking=training_params.non_blocking,
                                            prepare_batch=prepare_batch,
                                            output_transform=lambda x, y, y_pred: (y_pred, y))

    ProgressBar(persist=False, desc="Val Evaluation").attach(evaluator)

    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=training_params.evaluate_every_epoch),
                              evaluator.run(val_loader))

    score_metric_name = 'mIoU'

    if training_params.early_stopping_patience is not None:
        common.add_early_stopping_by_val_score(training_params.early_stopping_patience,
                                               evaluator, trainer, metric_name=score_metric_name)
    tb_logger = common.setup_tb_logging(training_params.tensorboard_log_dir, trainer, optimizer,
                                        evaluators={'validation': evaluator})

    tb_logger.attach(evaluator,
                     log_handler=LogImagesHandler(data_params.color_codes,
                                                 one_large_image=True,
                                                 max_images=4,
                                                 global_step_engine=trainer),

                     event_name=Events.EPOCH_COMPLETED
                     )

    common.save_best_model_by_val_score(training_params.model_out_dir, evaluator, model,
                                        metric_name=score_metric_name, trainer=trainer)

    trainer.run(train_loader, max_epochs=training_params.n_epochs)


def prepare_batch(batch, device=None, non_blocking=False):
    return (convert_tensor(batch['images'], device=device, non_blocking=non_blocking),
            (convert_tensor(batch['labels'], device=device, non_blocking=non_blocking),
             convert_tensor(batch['shapes'], device=device, non_blocking=non_blocking)))


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