from dh_segment_torch.config.params import Params
from dh_segment_torch.tests.dhsegment_test_case import DhSegmentTestCase
from dh_segment_torch.training.trainer import Trainer


class TrainerTest(DhSegmentTestCase):
    def test_trainer(self):

        blocks = 2
        num_channels = 32
        params = Params(
            {
                "color_labels": {
                    "type": "txt",
                    "label_text_file": self.FIXTURES_ROOT
                    / "dataset"
                    / "multiclass"
                    / "classes.txt",
                },
                "train_dataset": {
                    "type": "image_csv",
                    "csv_filename": self.FIXTURES_ROOT
                    / "dataset"
                    / "multiclass"
                    / "train.csv",
                    "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
                },
                "model": {
                    "encoder": "resnet50",
                    "decoder": {'decoder_channels': [512, 256, 128, 64, 32]}
                    # "loss": {"type": "dice"},
                },
                'metrics': ['iou', ('iou_class', {'type': 'iou', 'average': None}), 'precision'],
                'val_dataset': {
                    "type": "image_csv",
                    "csv_filename": self.FIXTURES_ROOT
                    / "dataset"
                    / "multiclass"
                    / "test.csv",
                    "base_dir": self.FIXTURES_ROOT / "dataset" / "multiclass",
                },
                'lr_scheduler': {"type": "exponential", "gamma": 0.95},
                'early_stopping': {'patience': 20},
                'model_out_dir': str(self.TEMPORARY_DIR / "model"),
                'num_epochs': 2,
                'evaluate_every_epoch': 1
            }
        )

        trainer = Trainer.from_params(params)
        trainer.train()
        # 1/0

