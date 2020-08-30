import argparse
import logging
import os
from typing import Optional

import pandas as pd

from dh_segment_torch.config.params import Params
from dh_segment_torch.config.registrable import Registrable
from dh_segment_torch.data.annotation.writers.annotation_writer import AnnotationWriter


class DataSplitter(Registrable):
    def __init__(
        self,
        train_ratio: float = 1.0,
        val_ratio: float = 0,
        test_ratio: float = 0,
    ):
        ratio_sum = train_ratio
        if val_ratio > 0:
            ratio_sum += val_ratio
        if test_ratio > 0:
            ratio_sum += test_ratio

        if ratio_sum != 1.0:

            train_ratio /= ratio_sum
            val_ratio /= ratio_sum
            test_ratio /= ratio_sum

            logger.warning(
                f"Ratio sum is different than 1.0 ({ratio_sum})"
                ", normalizing ratios by it, new ratios are: "
                f"train ratio = {train_ratio*100:.2f}%, "
                f"val ratio = {val_ratio*100:.2f}%, "
                f"test ratio = {test_ratio*100:.2f}%."
            )
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split_data(
        self,
        data: pd.DataFrame,
        train_csv_path: str,
        val_csv_path: Optional[str] = None,
        test_csv_path: Optional[str] = None,
    ):
        data = data.copy(deep=True)
        num_samples = len(data)
        num_train = int(num_samples * self.train_ratio)
        num_val = 0
        num_test = 0
        if self.test_ratio:
            num_test = int(num_samples * self.test_ratio)
        if self.val_ratio:
            num_val = int(num_samples * self.val_ratio)

        num_left = num_samples - num_train - num_val - num_test
        if num_left > 0:
            num_train += num_left

        train = data.sample(n=num_train).copy(deep=True)
        train.to_csv(train_csv_path, index=False, header=False)
        data.drop(train.index, inplace=True)

        if num_test > 0:
            test = data.sample(n=num_test).copy(deep=True)
            test.to_csv(test_csv_path, index=False, header=False)
            data.drop(test.index, inplace=True)

        if num_val > 0:
            val = data.sample(n=num_val).copy(deep=True)
            val.to_csv(val_csv_path, index=False, header=False)
            data.drop(val.index, inplace=True)

        assert len(data) == 0


DataSplitter.register("default")(DataSplitter)


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

    relative_path = params.pop("relative_path", True)

    params.setdefault("labels_dir", os.path.join(data_path, "labels"))
    labels_dir = params.get("labels_dir")

    params.setdefault("images_dir", os.path.join(data_path, "images"))
    images_dir = params.get("images_dir")

    params.setdefault(
        "color_labels_file_path", os.path.join(data_path, "color_labels.json")
    )
    params.setdefault("csv_path", os.path.join(data_path, "data.csv"))

    data_splitter_params = params.pop("data_splitter", None)
    train_csv_path = params.pop("train_csv", os.path.join(data_path, "train.csv"))
    val_csv_path = params.pop("val_csv", os.path.join(data_path, "val.csv"))
    test_csv_path = params.pop("test_csv", os.path.join(data_path, "test.csv"))

    params.setdefault("type", "image")
    image_writer = AnnotationWriter.from_params(params)
    data = image_writer.write(num_processes)

    if relative_path:
        data['image'] = data['image'].apply(lambda path: os.path.join("images", os.path.basename(path)))
        data['label'] = data['label'].apply(lambda path: os.path.join("labels", os.path.basename(path)))

    if data_splitter_params:
        data_splitter = DataSplitter.from_params(data_splitter_params)
        data_splitter.split_data(data, train_csv_path, val_csv_path, test_csv_path)
