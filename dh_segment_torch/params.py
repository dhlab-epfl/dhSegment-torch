import logging
import os
from enum import Enum

import numpy as np
import torch


class PredictionType(Enum):
    """

    :cvar CLASSIFICATION:
    :cvar MULTILABEL:
    """
    CLASSIFICATION = 'CLASSIFICATION'
    MULTILABEL = 'MULTILABEL'

    @classmethod
    def parse(cls,
              prediction_type: str):
        if prediction_type == 'CLASSIFICATION':
            return PredictionType.CLASSIFICATION
        elif prediction_type == 'MULTILABEL':
            return PredictionType.MULTILABEL
        else:
            raise NotImplementedError('Unknown prediction type : {}'.format(prediction_type))

class BaseParams:
    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, d):
        result = cls()
        keys = result.to_dict().keys()
        for k, v in d.items():
            assert k in keys, k
            setattr(result, k, v)
        result.check_params()
        return result

    def check_params(self):
        pass


class ModelParams(BaseParams):
    def __init__(self, **kwargs):
        self.encoder_network = kwargs.get('encoder_network', 'dh_segment_torch.network.encoders.Resnet50Encoder')
        self.encoder_params = kwargs.get('encoder_params', dict())
        self.decoder_network = kwargs.get('decoder_network', 'dh_segment_torch.network.decoders.UnetDecoder')
        self.decoder_params = kwargs.get('decoder_params', {'decoder_channels': [512, 256, 128, 64, 32]})

        self.pretraining = kwargs.get('pretraining', True)


    def check_params(self):
        pass


class DataParams(BaseParams):
    """Parameters to configure input pipeline / data preparation

    :ivar train_data: Training data (csv file, folder name or list of csv files)
    :vartype train_data: str, List[str]
    :ivar validation_data: Validation data (csv file, folder name or list of csv files)
    :vartype validation_data: str, List[str]
    :ivar classes_file: Class.txt file containing the info on color code and labels of classes
    :vartype classes_file: str
    :ivar input_resized_size: size (in pixel) of the image after resizing. The original ratio is kept. If no resizing \
    is wanted, set it to -1
    :vartype input_resized_size: int
    :ivar prediction_type: type of prediction, either 'CLASSIFICATION' or 'MULTILABEL'
    :vartype prediction_type: str
    :ivar make_patches: option to crop image into patches. This will cut the entire image in several patches
    :vartype make_patches: bool
    :ivar patch_shape: shape of the patches
    :vartype patch_shape: tuple
    :ivar data_augmentation: option to use data augmentation (by default is set to False)
    :vartype data_augmentation: bool
    :ivar data_augmentation_flip_lr: option to use image flipping in right-left direction
    :vartype data_augmentation_flip_lr: bool
    :ivar data_augmentation_flip_ud: option to use image flipping in up down direction
    :vartype data_augmentation_flip_ud: bool
    :ivar data_augmentation_color: option to use data augmentation with color
    :vartype data_augmentation_color: bool
    :ivar data_augmentation_max_rotation: maximum angle of rotation (in degrees) for data augmentation
    :vartype data_augmentation_max_rotation: float
    :ivar data_augmentation_max_scaling: maximum factor of zooming during data augmentation (range: ]1,inf]).
    It creates a range [size/scale; size*scale].
    :vartype data_augmentation_max_scaling: float
    :ivar maximum_input_size: maximum size of the image (in pixels), also taking into account max scaling. Default: 1e9
    :vartype maximum_input_size: int
    :ivar minimum_input_size: minimum size of the image (in pixels), also taking into account max scaling. Default: 1e4
    :vartype minimum_input_size: int
    """
    def __init__(self, **kwargs):
        self.train_data = kwargs.get('train_data')
        self.validation_data = kwargs.get('validation_data')
        self.classes_file = kwargs.get('classes_file')
        self.input_resized_size = int(kwargs.get('input_resized_size', 72e4))  # (600*1200)
        self.prediction_type = self._assign_prediction_type(kwargs.get('prediction_type', 'CLASSIFICATION'))
        self.make_patches = kwargs.get('make_patches', True)
        self.patch_shape = kwargs.get('patch_shape', (300, 300))

        self.data_augmentation = kwargs.get('data_augmentation', False)
        self.data_augmentation_horizontal_flip = kwargs.get('data_augmentation_flip_lr', False)
        self.data_augmentation_vertical_flip = kwargs.get('data_augmentation_flip_ud', False)
        self.data_augmentation_color = kwargs.get('data_augmentation_color', False)
        self.data_augmentation_max_rotation = kwargs.get('data_augmentation_max_rotation', 10)
        self.data_augmentation_max_scaling = kwargs.get('data_augmentation_max_scaling', 1.5)

        self.maximum_input_size = kwargs.get('maximum_input_size', 1e9)
        self.minimum_input_size = kwargs.get('minimum_input_size', 1e4)

        color_codes, onehot_labels, n_classes = self._get_classes_from_file()

        self.color_codes = color_codes
        self.onehot_labels = onehot_labels
        self.n_classes = n_classes

    @staticmethod
    def _assign_prediction_type(prediction_type_string: str):
        return PredictionType.parse(prediction_type_string)

    def _get_classes_from_file(self):
        if not os.path.exists(self.classes_file):
            raise FileNotFoundError(self.classes_file)

        content = np.loadtxt(self.classes_file).astype(np.float32)

        if self.prediction_type == PredictionType.CLASSIFICATION:
            assert content.shape[1] == 3, "Color file should represent RGB values"
            color_codes = content
            onehot_labels = None
            n_classes = len(color_codes)
        elif self.prediction_type == PredictionType.MULTILABEL:
            assert content.shape[1] > 3, "The number of columns should be greater in multilabel framework"
            color_codes = content[:, :3]
            onehot_labels = content[:, 3:]
            n_classes = onehot_labels.shape[1]
        return color_codes, onehot_labels, n_classes

    def check_params(self):
        # todo: check patch shape >= h x w
        # Check data augmentation params
        if not self.data_augmentation:
            logging.info('Data augmentation is disabled. All augmentation parameters will be disabled.')
            self.data_augmentation_horizontal_flip = False
            self.data_augmentation_vertical_flip = False
            self.data_augmentation_color = False
            self.data_augmentation_max_rotation = 0
            self.data_augmentation_max_scaling = 1.0
        else:
            assert self.data_augmentation_max_scaling >= 1.0, "Scaling factor should be greater or equal to 1.0"
            if self.data_augmentation_max_scaling < 1.0:
                self.data_augmentation_max_scaling = 1.0
                logging.error(f'Scaling factor should be greater or equal to 1.0. '
                              f'Changed its value to {self.data_augmentation_max_scaling}')

            if self.data_augmentation_max_rotation > 180:
                self.data_augmentation_max_rotation = 180
                logging.error(f'Rotation angle should be lower or equal to 180. '
                              f'Changed its value to {self.data_augmentation_max_rotation}')

            if int(self.input_resized_size / self.data_augmentation_max_scaling) < self.minimum_input_size:
                self.data_augmentation_max_scaling = self.input_resized_size / self.minimum_input_size
                logging.error(f"Scaling factor exceeded minimum image size. "
                              f"Changed it value to {self.data_augmentation_max_scaling}")
            if int(self.input_resized_size * self.data_augmentation_max_scaling) > self.maximum_input_size:
                self.data_augmentation_max_scaling = self.maximum_input_size / self.input_resized_size
                logging.error(f"Scaling factor exceeded maximum image size. "
                              f"Changed it value to {self.data_augmentation_max_scaling}")


class TrainingParams(BaseParams):
    """Parameters to configure training process

    :ivar n_epochs: number of epoch for training
    :vartype n_epochs: int
    :ivar evaluate_every_epoch: the model will be evaluated every `n` epochs
    :vartype evaluate_every_epoch: int
    :ivar learning_rate: the starting learning rate value
    :vartype learning_rate: float
    :ivar exponential_learning: option to use exponential learning rate
    :vartype exponential_learning: bool
    :ivar batch_size: size of batch
    :vartype batch_size: int
    :ivar data_augmentation: option to use data augmentation (by default is set to False)
    :vartype data_augmentation: bool
    :ivar data_augmentation_flip_lr: option to use image flipping in right-left direction
    :vartype data_augmentation_flip_lr: bool
    :ivar data_augmentation_flip_ud: option to use image flipping in up down direction
    :vartype data_augmentation_flip_ud: bool
    :ivar data_augmentation_color: option to use data augmentation with color
    :vartype data_augmentation_color: bool
    :ivar data_augmentation_max_rotation: maximum angle of rotation (in radians) for data augmentation
    :vartype data_augmentation_max_rotation: float
    :ivar data_augmentation_max_scaling: maximum scale of zooming during data augmentation (range: [0,1])
    :vartype data_augmentation_max_scaling: float
    :ivar make_patches: option to crop image into patches. This will cut the entire image in several patches
    :vartype make_patches: bool
    :ivar patch_shape: shape of the patches
    :vartype patch_shape: tuple
    :ivar input_resized_size: size (in pixel) of the image after resizing. The original ratio is kept. If no resizing \
    is wanted, set it to -1
    :vartype input_resized_size: int
    :ivar weights_labels: weight given to each label. Should be a list of length = number of classes
    :vartype weights_labels: list_labe
    :ivar training_margin: size of the margin to add to the images. This is particularly useful when training with \
    patches
    :vartype training_margin: int
    :ivar local_entropy_ratio:
    :vartype local_entropy_ratio: float
    :ivar local_entropy_sigma:
    :vartype local_entropy_sigma: float
    :ivar focal_loss_gamma: value of gamma for the focal loss. See paper : https://arxiv.org/abs/1708.02002
    :vartype focal_loss_gamma: float
    """
    def __init__(self, **kwargs):
        self.n_epochs = kwargs.get('n_epochs', 20)
        self.evaluate_every_epoch = kwargs.get('evaluate_every_epoch', 10)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.exponential_learning = kwargs.get('exponential_learning', True)
        self.batch_size = kwargs.get('batch_size', 5)
        self.accumulation_steps = kwargs.get('accumulation_steps', 1)
        self.weight_decay = kwargs.get('weight_decay', 1e-6)
        self.weights_labels = kwargs.get('weights_labels', None)
        self.weights_evaluation_miou = kwargs.get('weights_evaluation_miou', None)
        self.training_margin = kwargs.get('training_margin', 16)
        self.local_entropy_ratio = kwargs.get('local_entropy_ratio', 0.)
        self.local_entropy_sigma = kwargs.get('local_entropy_sigma', 3)
        self.focal_loss_gamma = kwargs.get('focal_loss_gamma', 0.)
        self.device = kwargs.get('device', 'cpu') if torch.cuda.is_available() else 'cpu'
        self.non_blocking = kwargs.get('non_blocking', True)
        self.pin_memory = kwargs.get('pin_memory', True)
        self.model_out_dir = kwargs.get('model_out_dir', './model')
        self.tensorboard_log_dir = kwargs.get('tensorboard_log_dir', os.path.join(self.model_out_dir, 'logs'))
        self.early_stopping_patience = kwargs.get("early_stopping_patience", None)
        self.num_data_workers = kwargs.get("num_data_workers", 16)
        self.resume_training = kwargs.get("restore_training", False)
        self.train_checkpoint_interval = kwargs.get("train_checkpoint_interval", 1000) # TODO check TF default
        self.patches_images_buffer_size = kwargs.get("patches_images_buffer_size", 5)
        self.drop_last_batch = kwargs.get("drop_last_batch", False)

    def check_params(self) -> None:
        """Checks if there is no parameter inconsistency
        """
        assert self.training_margin*2 < min(self.patch_shape)
        assert check_valid_device(self.device)


def check_valid_device(device):
    try:
        device = torch.device(device)
    except RuntimeError:
        return False
    if device.type == 'cpu':
        return True
    elif device.type == 'cuda':
        if device.index is None or device.index < torch.cuda.device_count():
            return True
    return False