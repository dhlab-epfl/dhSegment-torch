import argparse
import logging

from dh_segment_torch.config.params import Params
from dh_segment_torch.data.annotation import Annotation, AnnotationWriter
from dh_segment_torch.inference import PredictProcess

parser = argparse.ArgumentParser(description="Predict annotations.")
parser.add_argument(
    "config", type=str, help="The configuration file for predicting annotations."
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)

    writer_params = params.pop("writer")
    writer_num_processes = params.pop("num_processes", 4)

    predict_annots = PredictProcess.from_params(params)
    annots = predict_annots.process()

    if len(annots) > 0 and isinstance(annots[0], Annotation):
        writer_params["annotation_iterator"] = annots
        writer = AnnotationWriter.from_params(writer_params)
        writer.write(writer_num_processes)
    else:
        logger.error("Either no annotations was found or the post processing pipeline does not produce annotations.")
