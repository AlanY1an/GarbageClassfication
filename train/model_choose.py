import os
import logging
import torch
from typing import Optional, Dict, Callable

# 导入各种模型
from model.DenseNet import densenet201
from model.EfficientNet import efficientnet_b7
from model.GoogLeNet import GoogLeNet
from model.LeNet import LeNet
from model.MobileNetV2 import mobilenet_v2
from model.MobileNetV3 import mobilenet_v3_large
from model.ResNet import resnet50
from model.AlexNet import AlexNet
from model.ShuffleNetV2 import shufflenet_v2_x2_0
from model.VGGNet import vgg
from model.RegNet import regnet_x_32gf


class TrainingConfig:
    """Training configuration class to manage basic parameters for model training"""

    def __init__(
            self,
            model_name: str = 'ResNet',
            num_epochs: int = 5,
            learning_rate: float = 0.001,
            batch_size: int = 64,
            num_classes: int = 12,
            train_data_path: str = '../data/split-data/train',
            val_data_path: str = '../data/split-data/val',
            weights_dir: str = '../weights/ResNet',
            pretrained_weights: Optional[str] = None
    ):
        # Initialize training configuration parameters
        self.model_name = model_name
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.weights_dir = weights_dir
        self.pretrained_weights = pretrained_weights
        # Automatically detect and select the available device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelChoose:
    """Model selection and initialization class, supporting multiple deep learning models and pre-trained weight handling"""

    def __init__(self, config: TrainingConfig):
        """
        Initialize the model selector

        :param config: Training configuration object
        """
        self.config = config
        self.logger = self._setup_logger()
        # Define supported model mappings
        self.supported_models: Dict[str, Callable] = {
            "LeNet": self._create_lenet,
            "AlexNet": self._create_alexnet,
            "GoogLeNet": self._create_googlenet,
            "VGG": self._create_vgg19,
            "ResNet": self._create_resnet50,
            "RegNet": self._create_regnetx,
            "MobileNetV2": self._create_mobilenetv2,
            "MobileNetV3": self._create_mobilenetv3,
            "DenseNet": self._create_densenet201,
            "EfficientNet": self._create_efficientnet,
            "ShuffleNetV2": self._create_shufflenetv2,
        }

    def _create_model_factory(self, model_func: Callable) -> torch.nn.Module:
        """
        Generic model creation factory method

        :param model_func: Model creation function
        :return: Initialized model instance
        """
        return model_func(num_classes=self.config.num_classes).to(self.config.device)

    def _create_lenet(self) -> torch.nn.Module:
        return LeNet(num_classes=self.config.num_classes).to(self.config.device)

    def _create_alexnet(self) -> torch.nn.Module:
        return AlexNet(num_classes=self.config.num_classes, init_weights=True).to(self.config.device)

    def _create_googlenet(self) -> torch.nn.Module:
        return GoogLeNet(num_classes=self.config.num_classes).to(self.config.device)

    def _create_vgg19(self) -> torch.nn.Module:
        return vgg(model_name="vgg19", num_classes=self.config.num_classes, init_weights=True).to(
            self.config.device)

    def _create_resnet50(self) -> torch.nn.Module:
        return resnet50(num_classes=self.config.num_classes).to(self.config.device)

    def _create_regnetx(self) -> torch.nn.Module:
        return regnet_x_32gf(num_classes=self.config.num_classes).to(self.config.device)

    def _create_mobilenetv2(self) -> torch.nn.Module:
        return mobilenet_v2(num_classes=self.config.num_classes).to(self.config.device)

    def _create_mobilenetv3(self) -> torch.nn.Module:
        return mobilenet_v3_large(num_classes=self.config.num_classes).to(self.config.device)

    def _create_densenet201(self) -> torch.nn.Module:
        return densenet201(num_classes=self.config.num_classes).to(self.config.device)

    def _create_efficientnet(self) -> torch.nn.Module:
        return efficientnet_b7(num_classes=self.config.num_classes).to(self.config.device)

    def _create_shufflenetv2(self) -> torch.nn.Module:
        return shufflenet_v2_x2_0(num_classes=self.config.num_classes).to(self.config.device)

    def initialize_model(self) -> torch.nn.Module:
        """
        Initialize the model and load pre-trained weights
        """
        # Retrieve model creation function from the mapping dictionary
        model_creator = self.supported_models.get(
            self.config.model_name,
            self._create_resnet50  # Default to ResNet50
        )
        model = model_creator()

        # Load pre-trained weights if provided
        if self.config.pretrained_weights and os.path.exists(self.config.pretrained_weights):
            try:
                # Load weights and selectively update model parameters
                state_dict = torch.load(self.config.pretrained_weights, map_location=self.config.device,
                                        weights_only=True)
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and v.size() == model_dict[k].size()
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict)

                # Selectively freeze and unfreeze layers based on the model
                self._selective_layer_freezing(model)

                self.logger.info(f"Successfully loaded weights from {self.config.pretrained_weights}")
            except Exception as e:
                self.logger.warning(f"Failed to load weights: {e}")

        return model

    def _selective_layer_freezing(self, model: torch.nn.Module):
        """
        Selectively freeze and unfreeze model layers

        :param model: The model instance to process
        """
        # Define models requiring special handling and their freezing strategies
        special_models = {
            'AlexNet': ('classifier', 'classifier'),
            'VGG': ('classifier', 'classifier'),
            'MobileNetV2': ('classifier', 'classifier'),
            'MobileNetV3': ('classifier', 'classifier'),
            'EfficientNet': ('classifier', 'classifier'),
            'DenseNet': ('classifier', 'classifier'),
        }

        # Retrieve the specific freezing strategy for the current model
        target_layer, trainable_layer = special_models.get(
            self.config.model_name,
            ('fc', 'fc')
        )

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze specific layers
        try:
            trainable_params = getattr(model, trainable_layer).parameters()
            for param in trainable_params:
                param.requires_grad = True
        except AttributeError:
            self.logger.warning(f"Could not find {trainable_layer} layer")

    def _setup_logger(self):
        """
        Set up the logger

        :return: Configured logger instance
        """
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        return logging.getLogger(__name__)
