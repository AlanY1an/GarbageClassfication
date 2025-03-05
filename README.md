# Garbage Classification Model Training

This project is focused on training deep learning models for image classification. The repository contains various neural network architectures that can be used for training and evaluation.

## Available Models
The following models are available for training:
- AlexNet
- DenseNet
- EfficientNet
- GoogLeNet
- LeNet
- MobileNetV2
- MobileNetV3
- RegNet
- ResNet
- ShuffleNetV2
- VGGNet

## Workflow

### 1. Prepare the Dataset
To train a model, replace the images inside the `data/` directory with your dataset. Then, use the scripts inside `dataProcessing/` to process the dataset:
- `ImgRename.py`: Renames images if necessary.
- `dataPartitioner.py`: Splits the dataset into training (`train`), validation (`val`), and testing (`test`) sets.

### 2. Train a Model
You can train a single model or all models at once:
- **Train a single model:** Use `model_choose.py` to select and train a specific model.
- **Train all models:** Use `model_trainAll.py` to train all models sequentially. 
  - If you choose this option, ensure that parameters such as the number of classes (categories in your dataset) are correctly set.

### 3. Test the Model
After training, use the scripts in `predict/` to evaluate the trained models on test images.

## Getting Started
1. Set up your dataset.
2. Run the data preprocessing scripts.
3. Choose a model and start training.
4. Test your trained model.

Feel free to modify the parameters and models to best fit your dataset and use case. Happy training!
