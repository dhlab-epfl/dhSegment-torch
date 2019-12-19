#!/usr/bin/env python
import os
import numpy as np
import warnings
import logging


class PredictionType:
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
    def from_dict(cls,
                  d: dict):
        result = cls()
        keys = result.to_dict().keys()
        for k, v in d.items():
            assert k in keys, k
            setattr(result, k, v)
        result.check_params()
        return result

    def check_params(self):
        pass


class VGG16ModelParams:
    PRETRAINED_MODEL_FILE = 'pretrained_models/vgg_16.ckpt'
    INTERMEDIATE_CONV = [
        [(256, 3)]
    ]
    UPSCALE_PARAMS = [
        [(32, 3)],
        [(64, 3)],
        [(128, 3)],
        [(256, 3)],
        [(512, 3)],
        [(512, 3)]
    ]
    SELECTED_LAYERS_UPSCALING = [
        True,
        True,  # Must have same length as vgg_upscale_params
        True,
        True,
        False,
        False
    ]
    CORRECTED_VERSION = None


class ResNetModelParams:
    PRETRAINED_MODEL_FILE = 'pretrained_models/resnet_v1_50.ckpt'
    INTERMEDIATE_CONV = None
    UPSCALE_PARAMS = [
        # (Filter size (depth bottleneck's output), number of bottleneck)
        (32, 0),
        (64, 0),
        (128, 0),
        (256, 0),
        (512, 0)
    ]
    SELECTED_LAYERS_UPSCALING = [
        # Must have the same length as resnet_upscale_params
        True,
        True,
        True,
        True,
        True
    ]
    CORRECT_VERSION = False


class UNetModelParams:
    PRETRAINED_MODEL_FILE = None
    INTERMEDIATE_CONV = None
    UPSCALE_PARAMS = None
    SELECTED_LAYERS_UPSCALING = None
    CORRECT_VERSION = False


class ModelParams(BaseParams):
    """Parameters related to the model

    """
    def __init__(self, **kwargs):
        self.batch_norm = kwargs.get('batch_norm', True)  # type: bool
        self.batch_renorm = kwargs.get('batch_renorm', True)  # type: bool
        self.weight_decay = kwargs.get('weight_decay', 1e-6)  # type: float
        self.n_classes = kwargs.get('n_classes', None)  # type: int
        self.pretrained_model_name = kwargs.get('pretrained_model_name', None)  # type: str
        self.max_depth = kwargs.get('max_depth', 512)  # type: int

        if self.pretrained_model_name == 'vgg16':
            model_class = VGG16ModelParams
        elif self.pretrained_model_name == 'resnet50':
            model_class = ResNetModelParams
        elif self.pretrained_model_name == 'unet':
            model_class = UNetModelParams
        else:
            raise NotImplementedError

        self.pretrained_model_file = kwargs.get('pretrained_model_file', model_class.PRETRAINED_MODEL_FILE)
        self.intermediate_conv = kwargs.get('intermediate_conv', model_class.INTERMEDIATE_CONV)
        self.upscale_params = kwargs.get('upscale_params', model_class.UPSCALE_PARAMS)
        self.selected_levels_upscaling = kwargs.get('selected_levels_upscaling', model_class.SELECTED_LAYERS_UPSCALING)
        self.correct_resnet_version = kwargs.get('correct_resnet_version', model_class.CORRECT_VERSION)
        self.check_params()

    def check_params(self):
        if self.upscale_params is not None and self.selected_levels_upscaling is not None:
            assert len(self.upscale_params) == len(self.selected_levels_upscaling), \
                'Upscaling levels and selection levels must have the same lengths (in model_params definition), ' \
                '{} != {}'.format(len(self.upscale_params),
                                  len(self.selected_levels_upscaling))

        if not os.path.isfile(self.pretrained_model_file):
            warnings.warn(f'WARNING - Default pretrained weights file in {self.pretrained_model_file} '
                          f'was not found. Have you changed the default pretrained file ?')


class TrainingParams(BaseParams):
    """Parameters to configure training process

    :ivar n_epochs: number of epoch for training
    :vartype n_epochs: int
    :ivar evaluate_every_epoch: the model will be evaluated every `n` epochs
    :vartype evaluate_every_epoch: int
    :ivar model_output_dir:  Directory to output the model
    :vartype model_output_dir: str
    :ivar restore_model: Set to true to continue training from last saved checkpoint
    :vartype restore_model: bool
    :ivar learning_rate: the starting learning rate value
    :vartype learning_rate: float
    :ivar exponential_learning: option to use exponential learning rate
    :vartype exponential_learning: bool
    :ivar batch_size: size of batch
    :vartype batch_size: int
    :ivar weights_labels: weight given to each label. Should be a list of length = number of classes
    :vartype weights_labels: list
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
        self.model_output_dir = kwargs.get('model_output_dir')
        self.restore_model = kwargs.get('restore_model', False)
        self.learning_rate = kwargs.get('learning_rate', 1e-5)
        self.exponential_learning = kwargs.get('exponential_learning', True)
        self.batch_size = kwargs.get('batch_size', 5)
        self.weights_labels = kwargs.get('weights_labels')
        self.weights_evaluation_miou = kwargs.get('weights_evaluation_miou', None)
        self.training_margin = kwargs.get('training_margin', 16)
        self.local_entropy_ratio = kwargs.get('local_entropy_ratio', 0.)
        self.local_entropy_sigma = kwargs.get('local_entropy_sigma', 3)
        self.focal_loss_gamma = kwargs.get('focal_loss_gamma', 0.)

        self.check_params()
        self.init_training_folder()

    def init_training_folder(self):
        # Create output directory
        if not os.path.isdir(self.model_output_dir):
            os.makedirs(self.model_output_dir)
        else:
            assert self.restore_model, \
                f'{self.model_output_dir} already exists, you cannot use it as output directory. ' \
                f'Set "training_params.restore_model=True" to continue training, ' \
                f'or delete dir "rm -r {self.model_output_dir}"'

        # Create export directory for saved models
        saved_model_dir = os.path.join(self.model_output_dir, 'export')
        if not os.path.isdir(saved_model_dir):
            os.makedirs(saved_model_dir)

    def check_params(self) -> None:
        """Checks if there is no parameter inconsistency
        """
        # assert self.training_margin*2 < min(self.patch_shape)
        if self.evaluate_every_epoch >= self.n_epochs:
            warnings.warn(f'Model will be trained for {self.n_epochs} but will not be evaluated until the end '
                          f'of the training because `evaluate_every_epoch` value is {self.evaluate_every_epoch}')


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

        self._get_classes_from_file()

    @staticmethod
    def _assign_prediction_type(prediction_type_string: str):
        return PredictionType.parse(prediction_type_string)

    def _get_classes_from_file(self):
        if not os.path.exists(self.classes_file):
            raise FileNotFoundError(self.classes_file)

        content = np.loadtxt(self.classes_file).astype(np.float32)

        if self.prediction_type == PredictionType.CLASSIFICATION:
            assert content.shape[1] == 3, "Color file should represent RGB values"
            self.color_codes = content
            self.onehot_labels = None
        elif self.prediction_type == PredictionType.MULTILABEL:
            assert content.shape[1] > 3, "The number of columns should be greater in multilabel framework"
            self.color_codes = content[:, :3]
            self.onehot_labels = content[:, 3:]

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


