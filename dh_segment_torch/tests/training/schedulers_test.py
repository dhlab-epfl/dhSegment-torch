import numpy as np
import torch

from dh_segment_torch.config.params import Params
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase
from dh_segment_torch.training.optimizers import Optimizer
from dh_segment_torch.training.schedulers import Scheduler


class OptimizerTest(DhSegmentTestCase):
    def test_steps_scheduler(self):
        named_parameters = [("x", torch.nn.Parameter())]
        lr = 10
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )

        step_size = 10
        gamma = 0.1
        sched_params = {"type": "step", "step_size": step_size, "gamma": gamma}
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)

        assert sched.get_last_lr()[0] == lr
        for _ in range(step_size - 1):
            opt.step()
            sched.step()
            assert sched.get_last_lr()[0] == lr
        opt.step()
        sched.step()
        assert np.allclose(sched.get_last_lr()[0], lr * gamma)

        lr = 10
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )
        gamma = 0.1
        sched_params = {"type": "multi_step", "milestones": [5, 20], "gamma": gamma}
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)

        assert sched.get_last_lr()[0] == lr
        for _ in range(4):
            opt.step()
            sched.step()
            assert sched.get_last_lr()[0] == lr
        opt.step()
        sched.step()
        assert np.allclose(sched.get_last_lr()[0], lr * gamma)
        for _ in range(14):
            sched.step()
            opt.step()
            assert np.allclose(sched.get_last_lr()[0], lr * gamma)
        opt.step()
        sched.step()
        assert np.allclose(sched.get_last_lr()[0], lr * gamma ** 2)

        lr = 10
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )

        gamma = 0.1

        sched_params = {"type": "exponential", "gamma": gamma}
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)

        assert sched.get_last_lr()[0] == lr
        for i in range(50):
            opt.step()
            sched.step()
            assert np.allclose(sched.get_last_lr()[0], lr * gamma ** (i + 1))

    def test_reduce_lr_on_plateau(self):
        named_parameters = [("x", torch.nn.Parameter())]
        lr = 1
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )

        gamma = 0.01
        patience = 2
        sched_params = {
            "type": "reduce_on_plateau",
            "mode": "max",
            "factor": gamma,
            "patience": patience,
            "min_lr": 1e-3,
        }
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)
        assert np.allclose(sched.get_last_lr()[0], lr)
        sched.step(0.5)
        assert np.allclose(sched.get_last_lr()[0], lr)
        sched.step(0.4)
        assert np.allclose(sched.get_last_lr()[0], lr)
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], lr)
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], lr * gamma)

        # test min_lr
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], lr * gamma)
        sched.step(0.4)
        assert np.allclose(sched.get_last_lr()[0], lr * gamma)
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], max(1e-3, lr * gamma ** 2))

    def test_lr_concat(self):
        named_parameters = [("x", torch.nn.Parameter())]
        lr = 1
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )

        gamma_exp = 0.95
        gamma_plateau = 0.01
        patience = 2
        sched_params = {
            "type": "concat",
            "schedulers": [
                {"type": "exponential", "gamma": gamma_exp},
                {
                    "type": "reduce_on_plateau",
                    "mode": "max",
                    "factor": gamma_plateau,
                    "patience": patience,
                    "min_lr": 1e-3,
                },
            ],
            "durations": [10,],
        }
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)

        assert np.allclose(sched.get_last_lr()[0], lr)
        for i in range(10):
            opt.step()
            sched.step(0.1)
            assert np.allclose(sched.get_last_lr()[0], lr * gamma_exp ** (i + 1))
        start_lr = lr * gamma_exp ** 10
        sched.step(0.5)
        assert np.allclose(sched.get_last_lr()[0], start_lr)
        sched.step(0.4)
        assert np.allclose(sched.get_last_lr()[0], start_lr)
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], start_lr)
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], start_lr * gamma_plateau)

        # test min_lr
        sched.step(0.3)
        assert np.allclose(sched.get_last_lr()[0], start_lr * gamma_plateau)
        sched.step(0.4)
        assert np.allclose(sched.get_last_lr()[0], start_lr * gamma_plateau)
        sched.step(0.3)
        assert np.allclose(
            sched.get_last_lr()[0], max(1e-3, start_lr * gamma_plateau ** 2)
        )

    def test_warmup_lr(self):
        named_parameters = [("x", torch.nn.Parameter())]
        lr = 1
        opt = Optimizer.from_params(
            Params({"type": "sgd", "lr": lr}), model_params=named_parameters
        )

        warmup_start = 0.1
        warmup_end = lr
        warmup_duration = 10
        gamma_exp = 0.95
        sched_params = {
            "type": "warmup",
            "scheduler": {"type": "exponential", "gamma": gamma_exp},
            "warmup_start_value": warmup_start,
            "warmup_end_value": warmup_end,
            "warmup_duration": warmup_duration,
        }
        sched = Scheduler.from_params(Params(sched_params), optimizer=opt)
        assert np.allclose(sched.get_last_lr()[0], warmup_start)
        for _ in range(warmup_duration):
            opt.step()
            sched.step()
        assert np.allclose(sched.get_last_lr()[0], warmup_end)

        for i in range(50):
            opt.step()
            sched.step()
            assert np.allclose(
                sched.get_last_lr()[0], warmup_end * gamma_exp ** (i + 1)
            )
