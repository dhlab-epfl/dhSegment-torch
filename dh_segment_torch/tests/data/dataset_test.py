import cv2
import pytest
from torchvision.transforms.functional import to_tensor

from dh_segment_torch.config import ConfigurationError
from dh_segment_torch.config.params import Params
from dh_segment_torch.data.dataset.dataset import Dataset
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase


class DatasetTest(DhSegmentTestCase):
    def test_basic_load_dataset(self):
        params = {
            "type": "image_csv",
            "csv_filename": self.FIXTURES_ROOT / "dataset" / "multiclass" / "train.csv",
            "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
        }

        dataset = Dataset.from_params(Params(params))
        assert len(dataset) == 10
        assert dataset.num_images == 10
        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert "shape" in sample

        patches_size = 32
        params = {
            "type": "patches_folder",
            "folder": self.FIXTURES_ROOT / "dataset" / "multilabel",
            "patches_size": patches_size,
        }

        dataset = Dataset.from_params(Params(params))
        assert dataset.num_images == 15
        for sample in dataset:
            break
        assert "image" in sample
        assert "label" in sample
        assert "shape" not in sample
        assert sample["image"].shape[1] == patches_size

    def test_transform_dataset(self):
        first_image = cv2.imread(
            str(
                self.FIXTURES_ROOT
                / "dataset"
                / "multiclass"
                / "images"
                / "image_001.png"
            )
        )
        first_image_tensor = to_tensor(first_image)

        params = {
            "type": "image_csv",
            "csv_filename": self.FIXTURES_ROOT / "dataset" / "multiclass" / "train.csv",
            "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
            "compose_transform": {
                "transforms": [
                    {"type": "fixed_size_resize", "output_size": 1e5},
                    "gaussian_blur",
                    "flip",
                    {
                        "type": "assign_label_classification",
                        "colors_array": [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
                    },
                ]
            },
        }

        dataset = Dataset.from_params(Params(params))
        sample = dataset[0]

        assert "image" in sample
        assert "label" in sample
        assert "shape" in sample

        assert sample["label"].unique().numpy().tolist() == [0, 1, 2]

        with pytest.raises(ConfigurationError):
            params = {
                "type": "image_csv",
                "csv_filename": self.FIXTURES_ROOT / "dataset" / "multiclass" / "train.csv",
                "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
                "compose_transform": {
                    "blur"
                },
            }

            Dataset.from_params(Params(params))

    def test_patches_transform(self):
        patches_size = 32
        params = {
            "type": "patches_csv",
            "csv_filename": self.FIXTURES_ROOT / "dataset" / "multiclass" / "train.csv",
            "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
            "patches_size": patches_size,
            "add_shapes": True,
            "pre_compose_transform": {
                "transforms": [
                    {"type": "fixed_size_resize", "output_size": 1e5},
                ]
            },
            "post_compose_transform": {
                "transforms": [
                    "gaussian_blur",
                    "flip",
                    {
                        "type": "assign_label_classification",
                        "colors_array": [[0, 0, 0], [255, 0, 0], [0, 0, 255]],
                    },
                ]
            },
        }

        dataset = Dataset.from_params(Params(params))
        for sample in dataset:
            break
        assert "image" in sample
        assert "label" in sample
        assert "shape" in sample


    def test_assign_multilabel(self):
        params = {
            "type": "image_csv",
            "csv_filename": self.FIXTURES_ROOT / "dataset" / "multilabel" / "train.csv",
            "base_dir": self.FIXTURES_ROOT / "dataset" / "multilabel",
            "compose_transform": {
                "transforms": [
                    {"type": "fixed_size_resize", "output_size": 1e5},
                    {"type": "gaussian_blur"},
                    {"type": "flip"},
                    {
                        "type": "assign_label_multilabel",
                        "colors_array": [
                            [0, 0, 0],
                            [255, 0, 0],
                            [0, 0, 255],
                            [128, 0, 128],
                        ],
                        "onehot_label_array": [[0, 0], [1, 0], [0, 1], [1, 1]],
                    },
                ]
            },
        }

        dataset = Dataset.from_params(Params(params))
        sample = dataset[0]

        assert "image" in sample
        assert "label" in sample
        assert "shape" in sample
