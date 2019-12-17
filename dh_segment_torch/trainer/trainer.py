from ignite.contrib.handlers import CustomPeriodicEvent, TensorboardLogger
from ignite.engine import Engine, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, mIoU, ConfusionMatrix
from ignite.utils import convert_tensor

import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader

from .utils import patch_loss_with_padding, patch_metric_with_padding
from .params import TrainingParams


def get_trainer(model: nn.Module, dataloader: DataLoader, params: TrainingParams) -> Engine:
    tb_logger = TensorboardLogger(log_dir="test_tensorboard/")
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = patch_loss_with_padding(CrossEntropyLoss)()

    trainer = create_supervised_trainer(model,
                                        optimizer,
                                        loss_fn,
                                        prepare_batch=prepare_batch
                                        )

    evaluator = create_supervised_evaluator(model,
                                            prepare_batch=prepare_batch,
                                            output_transform=lambda x, y, y_pred: (x, y, y_pred),
                                            metrics={
                                                'accuracy': patch_metric_with_padding(Accuracy)(
                                                    output_transform=metric_output_transform),
                                                'loss': patch_metric_with_padding(Loss)(
                                                    F.cross_entropy, output_transform=metric_output_transform),
                                                'mIoU': mIoU(
                                                    patch_metric_with_padding(ConfusionMatrix)(
                                                        2, output_transform=metric_output_transform))
                                            }
                                            )

    every_2_it = CustomPeriodicEvent(n_iterations=2)

    evaluate_trigger = CustomPeriodicEvent(n_epochs=params.evaluate_every_epoch)
    evaluate_trigger.attach(trainer)




def prepare_batch(batch, device=None, non_blocking=False):
    return (convert_tensor(batch['images'], device=device, non_blocking=non_blocking),
            #            convert_tensor(batch['labels'], device=device, non_blocking=non_blocking))
            (convert_tensor(batch['labels'], device=device, non_blocking=non_blocking),
             convert_tensor(batch['shapes'], device=device, non_blocking=non_blocking)))


def metric_output_transform(output):
    x, y, y_pred = output
    return y_pred, y