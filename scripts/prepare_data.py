import argparse
import logging
import os

from dh_segment_torch.config.params import Params
from dh_segment_torch.data import DataSplitter
from dh_segment_torch.data.annotation.writers.annotation_writer import AnnotationWriter

parser = argparse.ArgumentParser(description="Prepare data for dhSegment.")
parser.add_argument(
    "config", type=str, help="The configuration file for training a dhSegment model"
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    args = parser.parse_args()
    params = Params.from_file(args.config)

    num_processes = params.pop("num_processes", 4)

    data_path = params.pop("data_path")
    os.makedirs(data_path, exist_ok=True)

    # Setting default params for image writer
    params.setdefault("labels_dir", os.path.join(data_path, "labels"))
    labels_dir = params.get("labels_dir")

    params.setdefault("images_dir", os.path.join(data_path, "images"))
    images_dir = params.get("images_dir")
    params.setdefault(
        "color_labels_file_path", os.path.join(data_path, "color_labels.json")
    )
    params.setdefault("csv_path", os.path.join(data_path, "data.csv"))

    # Getting data splitter params
    data_splitter_params = params.pop("data_splitter", None)
    train_csv_path = params.pop("train_csv", os.path.join(data_path, "train.csv"))
    val_csv_path = params.pop("val_csv", os.path.join(data_path, "val.csv"))
    test_csv_path = params.pop("test_csv", os.path.join(data_path, "test.csv"))

    params.setdefault("type", "image")
    image_writer = AnnotationWriter.from_params(params)
    data = image_writer.write(num_processes)

    relative_path = params.pop("relative_path", True)

    if relative_path:
        data['image'] = data['image'].apply(lambda path: os.path.join(images_dir, os.path.basename(path)))
        data['label'] = data['label'].apply(lambda path: os.path.join(labels_dir, os.path.basename(path)))

    if data_splitter_params:
        data_splitter = DataSplitter.from_params(data_splitter_params)
        data_splitter.split_data(data, train_csv_path, val_csv_path, test_csv_path)
