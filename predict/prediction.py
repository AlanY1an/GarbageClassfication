# -*- coding: utf-8 -*-
# @Time : 2024-12-09 17:08
# @Author : Lin Feng
# @File : prediction.py

import logging
import json
import time
from io import BytesIO

import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from predict.model_choose import ModelChoose, TrainingConfig


class Prediction:
    def __init__(self, config: TrainingConfig, logger=None, model=None):
        """
        Initialize the prediction class.
        :param config: Training configuration
        :param logger: Logger
        :param model: Model to be tested
        """
        self.config = config
        self.logger = logger
        self.model = model
        self.classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                        'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']
        if self.model is None:
            raise ValueError("Model is not initialized. Please provide a valid model instance.")

    def preprocess_image(self):
        """
        Preprocess the image to match the model input requirements.
        :return: Preprocessed image tensor
        """
        try:
            # Load the image from the local path
            image = Image.open(self.config.image_path).convert('RGB')
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.1847, 0.1716, 0.1502],
                    std=[0.0678, 0.0615, 0.0552]
                )
            ])
            return transform(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            raise ValueError(f"Image processing failed: {e}")

    def predict(self, image_tensor):
        """
        Perform model prediction on the image.
        :param image_tensor: Image tensor
        :return: Predicted class and confidence score
        """
        try:
            self.model.eval()
            with torch.no_grad():
                image_tensor = image_tensor.to(self.config.device)
                output = self.model(image_tensor)
                _, predicted_idx = torch.max(output, dim=1)
                confidence = torch.softmax(output, dim=1)[0, predicted_idx].item()
                return self.classes[predicted_idx.item()], confidence
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")

    def run(self):
        """
        Execute prediction and return the result.
        :return: JSON-formatted result
        """
        try:
            start_time = time.time()
            image_tensor = self.preprocess_image()
            prediction, confidence = self.predict(image_tensor)
            total_time = time.time() - start_time

            result = {
                "status": 200,
                "message": "Prediction successful",
                "prediction": prediction,
                "confidence": f"{confidence*100:.2f}%",
                "total_time": f"{total_time:.3f} seconds"
            }
            self.logger.info(f"Prediction completed: {result}")
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            error_response = {
                "status": 400,
                "message": f"Error: {str(e)}"
            }
            self.logger.error(f"Prediction failed: {error_response}")
            return json.dumps(error_response, ensure_ascii=False)


def main():
    """
    Main program entry point.
    """
    try:
        config = TrainingConfig(
            model_name='ResNet50',
            num_classes=12,
            image_path='../predict/test.jpg',
            pretrained_weights='../weights/ResNet/ResNet_model_98.14%.pth'
        )
        model_choose = ModelChoose(config)
        model = model_choose.initialize_model()

        predictor = Prediction(config, model_choose.logger, model)
        result = predictor.run()
        print(result)
    except Exception as e:
        print(f"Main program execution failed: {e}")


if __name__ == '__main__':
    main()
