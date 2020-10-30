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

parser.add_argument("output_dir", type=str, help="The output directory to store the probabilities.")

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)

    output_dir = args.output_dir

    params['post_process'] = None
    params['add_path'] = True

    predict_annots = PredictProcess.from_params(params)
    results = predict_annots.process()

    for result in results:
        basename = os.path.splitext(os.path.basename(result['path']))[0] + ".npy"
        output_path = os.path.join(output_dir, basename)
        np.save(output_path, result['probas'])
