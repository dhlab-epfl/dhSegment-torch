from dh_segment_torch.training.checkpoint import *
from dh_segment_torch.training.early_stopping import *
from dh_segment_torch.training.logging.logger import *
from dh_segment_torch.training.logging.tensorboard import *
from dh_segment_torch.training.logging.wandb import *
from dh_segment_torch.training.optimizers import *
from dh_segment_torch.training.regularizers import *
from dh_segment_torch.training.schedulers import *
from dh_segment_torch.training.trainer import *

_TRAINER = ["Trainer", "EarlyStopping"]

_REGULARIZER = ["Regularizer", "L1Regularizer", "L2Regularizer"]

_OPTIMIZER = [
    "Optimizer",
    "AdamOptimizer",
    "AdadeltaOptimizer",
    "AdagradOptimizer",
    "AdamWOptimizer",
    "SparseAdamOptimizer",
    "AdamaxOptimizer",
    "ASGDOptimizer",
    "LBFGSOptimizer",
    "RMSpropOptimizer",
    "RpropOptimizer",
    "SGDOptimizer",
]

_SCHEDULER = [
    "Scheduler",
    "ConstantScheduler",
    "StepScheduler",
    "MultiStepScheduler",
    "ExponentialScheduler",
    "CosineAnnealingScheduler",
    "ReduceOnPlateauScheduler",
    "CyclicScheduler",
    "OneCycleScheduler",
    "CosineAnnealingWarmRestartsScheduler",
    "ConcatScheduler",
]

_CHECKPOINT = ["Checkpoint", "TimeCheckpoint", "IterationCheckpoint", "BestCheckpoint"]


_LOGGER = ["Logger", "TensorboardLogger", "WandbLogger"]

__all__ = (
    _TRAINER + _REGULARIZER + _OPTIMIZER + _SCHEDULER + _CHECKPOINT + _LOGGER
)
