import argparse
import logging
import os

import numpy as np

from dh_segment_torch.config.params import Params
from dh_segment_torch.inference import PredictProcess

parser = argparse.ArgumentParser(description="Predict probabilities.")
parser.add_argument(
    "config", type=str, help="The configuration file for predicting probabilities."
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)

    output_dir = params.pop("output_directory")

    os.makedirs(output_dir, exist_ok=True)

    params['post_process'] = None
    params['add_path'] = True

    predict_annots = PredictProcess.from_params(params)
    results = predict_annots.process_to_probas_files(output_dir)
