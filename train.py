from copy import deepcopy

from dh_segment_torch.config.params import Params
from dh_segment_torch.training.trainer import Trainer

from sacred import Experiment
import argparse

parser = argparse.ArgumentParser(description="Train a dhSegment model")
parser.add_argument("config", type=str, help='The configuration file for training a dhSegment model')

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)
    exp_name = params.pop("experiment_name", "dhSegment_experiment")
    config = params.as_dict()

    ex = Experiment(exp_name)
    ex.add_config(config)

    trainer = Trainer.from_params(params, exp_name=exp_name, config=deepcopy(config))
    trainer.train()
