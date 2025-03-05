import os
from train.model_choose import TrainingConfig, ModelChoose
from train.model_train import ModelTrainer

def main():
    # List of model names to be trained (must match supported_models in ModelChoose)
    model_names = [
        # "LeNet", "AlexNet", "GoogLeNet", "VGG", "ResNet",
        "GoogLeNet",
        "RegNet", "MobileNetV2", "MobileNetV3", "DenseNet",
        "EfficientNet", "ShuffleNetV2"
    ]

    # Mapping of pretrained weight paths for each model (set to None if not available)
    pretrained_weights_paths = {
        # "ResNet": "../weights/ResNet/resnet50-0676ba61.pth",
        # "LeNet": None,  # Ensure this comma is present to avoid syntax errors
        # "AlexNet": "../weights/AlexNet/AlexNet.pth",
        "GoogLeNet": "../weights/GoogLeNet/GoogLeNet.pth",
        # "VGG": "../weights/VGGNet/vgg19.pth",
        "RegNet": "../weights/RegNet/regnet_x_32gf.pth",
        "MobileNetV2": "../weights/MobileNetV2/mobilenet_v2.pth",
        "MobileNetV3": "../weights/MobileNetV3/mobilenet_v3.pth",
        "DenseNet": "../weights/DenseNet/densenet201.pth",
        "EfficientNet": "../weights/EfficientNet/efficientnet_b7.pth",
        "ShuffleNetV2": "../weights/ShuffleNetV2/shufflenetv2_x2_0.pth",
    }

    # Iterate through each model name and train sequentially
    for model_name in model_names:
        print(f"\n{'='*20} Starting training for model: {model_name} {'='*20}")

        # Set a unique directory for saving weights of each model
        weights_dir = f"../weights/{model_name}"
        os.makedirs(weights_dir, exist_ok=True)

        # Retrieve the pretrained weight path (if available)
        pretrained_weight = pretrained_weights_paths.get(model_name, None)

        # Create a training configuration object (adjust parameters as needed)
        config = TrainingConfig(
            model_name=model_name,
            num_epochs=50,                  # Adjust the number of training epochs if necessary
            learning_rate=0.001,            # Learning rate
            batch_size=64,                  # Batch size
            num_classes=12,                 # Number of classes (modify based on your dataset)
            train_data_path="../data/split-data/train",
            val_data_path="../data/split-data/val",
            weights_dir=weights_dir,
            pretrained_weights=pretrained_weight
        )

        # Select and initialize the model based on the configuration
        model_choose = ModelChoose(config)
        model = model_choose.initialize_model()

        # Create a model trainer and start training
        trainer = ModelTrainer(config, model_choose.logger, model)
        trainer.train()

        print(f"{'='*20} Training completed for model: {model_name} {'='*20}\n")

if __name__ == "__main__":
    main()
