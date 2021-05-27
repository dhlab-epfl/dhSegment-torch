import argparse
import os
from copy import deepcopy

import torch
from sacred import Experiment

from dh_segment_torch.config.params import Params
from dh_segment_torch.training.trainer import Trainer

parser = argparse.ArgumentParser(description="Train a dhSegment model")
parser.add_argument(
    "config", type=str, help="The configuration file for training a dhSegment model"
)
parser.add_argument(
    "--trainer-checkpoint",
    type=str,
    nargs="?",
    default=None,
    help="trainer checkpoint to resume from",
)
parser.add_argument(
    "--model-checkpoint",
    type=str,
    nargs="?",
    default=None,
    help="model checkpoint to resume from",
)

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)

    model_out_dir = params.get("model_out_dir", "./model")
    os.makedirs(model_out_dir, exist_ok=True)
    params.to_file(os.path.join(model_out_dir, "config.json"))

    exp_name = params.pop("experiment_name", "dhSegment_experiment")
    config = params.as_dict()

    ex = Experiment(exp_name)
    ex.add_config(config)

    trainer = Trainer.from_params(params, exp_name=exp_name, config=deepcopy(config))

    state_dict = {}

    if args.trainer_checkpoint:
        state_dict = torch.load(args.trainer_checkpoint)

    if args.model_checkpoint:
        state_dict["model"] = torch.load(args.model_checkpoint)

    trainer.load_state_dict(state_dict)

    try:
        trainer.train()
    except KeyboardInterrupt as e:
        trainer.final_save()
        raise e
