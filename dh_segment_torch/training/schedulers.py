from bisect import bisect_right
from itertools import accumulate
from typing import List, Union, Optional, Callable, Dict, Any

from torch.optim import lr_scheduler

from dh_segment_torch.config.lazy import Lazy
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.training.optimizers import Optimizer


class Scheduler(Registrable):
    default_implementation = "default"

    def __init__(
        self,
        scheduler: Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau],
        step_duration: int = 1,
    ):
        self.scheduler = scheduler
        self.step_duration = step_duration
        self._step_count = 0

    def step(self, metric: float = None):
        self._step_count += 1
        if self._step_count % self.step_duration == 0:
            self.scheduler.step()

    def state_dict(self):
        return {
            "scheduler": self.scheduler.state_dict(),
            "step_duration": self.step_duration,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.scheduler = self.scheduler.load_state_dict(state_dict["scheduler"])
        self.step_duration = state_dict["step_duration"]
        self._step_count = state_dict["_step_count"]

    def get_last_lr(self):
        return self.scheduler.get_last_lr()

    def get_lr(self):
        return self.scheduler.get_lr()

@Scheduler.register("step")
class StepScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
        super().__init__(scheduler, step_duration)


@Scheduler.register("multi_step")
class MultiStepScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma, last_epoch)
        super().__init__(scheduler, step_duration)


@Scheduler.register("exponential")
class ExponentialScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float = 0.1,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch)
        super().__init__(scheduler, step_duration)

    @classmethod
    def default(cls, optimizer: Optimizer):
        cls(optimizer, gamma=0.95, step_duration=200)


Scheduler.register("default", "default")(ExponentialScheduler)


@Scheduler.register("cosine_annealing")
class CosineAnnealingScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.MultiStepLR(optimizer, T_max, eta_min, last_epoch)
        super().__init__(scheduler, step_duration)


@Scheduler.register("reduce_on_plateau")
class ReduceOnPlateauScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        verbose: bool = False,
        threshold: float = 0.0001,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: Union[float, List[float]] = 0,
        eps: float = 1e-08,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode,
            factor,
            patience,
            verbose,
            threshold,
            threshold_mode,
            cooldown,
            min_lr,
            eps,
        )
        super().__init__(scheduler, step_duration)

        self._last_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, metric: float = None):
        self._step_count += 1
        if self._step_count % self.step_duration == 0:
            self.scheduler.step(metric)
        self._last_lr = [group['lr'] for group in self.scheduler.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr



@Scheduler.register("cyclic")
class CyclicScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: str = "triangular",
        gamma: float = 1.0,
        scale_fn: Optional[Callable[[int], float]] = None,
        scale_mode: str = "cycle",
        cycle_momentum: bool = True,
        base_momentum: float = 0.8,
        max_momentum: float = 0.9,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.CyclicLR(
            optimizer,
            base_lr,
            max_lr,
            step_size_up,
            step_size_down,
            mode,
            gamma,
            scale_fn,
            scale_mode,
            cycle_momentum,
            base_momentum,
            max_momentum,
            last_epoch,
        )
        super().__init__(scheduler, step_duration)


@Scheduler.register("one_cycle")
class OneCycleScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: Optional[int] = None,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        cycle_momentum: bool = True,
        base_momentum: float = 0.85,
        max_momentum: float = 0.95,
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr,
            total_steps,
            epochs,
            steps_per_epoch,
            pct_start,
            anneal_strategy,
            cycle_momentum,
            base_momentum,
            max_momentum,
            div_factor,
            final_div_factor,
            last_epoch,
        )
        super().__init__(scheduler, step_duration)


@Scheduler.register("cosine_annealing_warm_restarts")
class CosineAnnealingWarmRestartsScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: int = 0,
        last_epoch: int = -1,
        step_duration: int = 1,
    ):
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0, T_mult, eta_min, last_epoch
        )
        super().__init__(scheduler, step_duration)


@Scheduler.register("concat")
class ConcatScheduler(Scheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: List[Lazy[Scheduler]],
        durations: List[int],
        step_duration: int = 1,
    ):
        super().__init__(None)
        if len(schedulers) != len(durations) + 1:
            raise ValueError(
                f"There should be {len(schedulers)-1} durations for {len(schedulers)} schedulers."
            )
        self.schedulers: List[Scheduler] = [
            scheduler.construct(optimizer=optimizer) for scheduler in schedulers
        ]

        self.duration_steps = [x * step_duration for x in accumulate(durations)]
        self.step_duration = step_duration
        self._step_count = 0

    def step(self, metric: float = None):
        self._step_count += 1
        if self._step_count % self.step_duration == 0:
            self.current_scheduler.step(metric)

    @property
    def current_scheduler(self) -> Scheduler:

        return self.schedulers[bisect_right(self.duration_steps, self._step_count-1)]

    def state_dict(self):
        return {
            "schedulers": [scheduler.state_dict() for scheduler in self.schedulers],
            "duration_steps": self.duration_steps,
            "step_duration": self.step_duration,
            "_step_count": self._step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.schedulers = [
            scheduler.load_state_dict(sched_dict)
            for scheduler, sched_dict in zip(self.schedulers, state_dict["schedulers"])
        ]
        self.duration_steps = state_dict["duration_steps"]
        self.step_duration = state_dict["step_duration"]
        self._step_count = state_dict["_step_count"]

    def get_last_lr(self):
        return self.current_scheduler.get_last_lr()

    def get_lr(self):
        return self.current_scheduler.get_lr()

    @classmethod
    def warmup_scheduler(
        cls,
        optimizer: Optimizer,
        scheduler: Lazy[Scheduler],
        warmup_start_value: float,
        warmup_end_value: float,
        warmup_duration: int,
        step_duration: int = 1,
    ):
        warmup = Lazy(
            lambda optimizer: CyclicScheduler(
                optimizer,
                warmup_start_value,
                warmup_end_value,
                warmup_duration * step_duration,
                cycle_momentum=True,
            )
        )
        return cls(optimizer, [warmup, scheduler], durations=[warmup_duration,])


Scheduler.register("warmup", "warmup_scheduler")(ConcatScheduler)