import logging
from datetime import datetime
import torch
import torchvision.transforms as transforms
import torch.utils.data as Data
from torchvision.datasets import ImageFolder
from predict.model_choose import ModelChoose, TrainingConfig


class ModelTest:
    def __init__(self, config: TrainingConfig, logger, model):
        """
        Initialize the test class.
        :param config: Training configuration
        :param logger: Logger
        :param model: The model to be tested
        """
        self.config = config
        self.logger = logger if logger else logging.getLogger(__name__)
        self.model = model
        self.classes = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
                        'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

    def _get_data_loaders(self):
        """Data preprocessing and loading test dataset"""
        # Define dataset transformation methods
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.18473272612791913, 0.17158856824516447, 0.15018492073688558],
                                 [0.06782158567666746, 0.061543981543858484, 0.05524772929753039]),  # Normalization
            transforms.Resize((224, 224))  # Resize images
        ])
        # Load dataset
        data = ImageFolder(self.config.test_data_path, transform=transform)
        print("Test Class-to-Index Mapping:", data.class_to_idx)
        test_dataloader = Data.DataLoader(dataset=data, batch_size=self.config.batch_size, shuffle=True, num_workers=0)
        return test_dataloader

    def test(self):
        """Test the model, compute accuracy, and yield logs"""
        # Log the start of testing
        self.logger.info('Starting model testing')

        test_dataloader = self._get_data_loaders()
        test_corrects = 0.0
        test_num = 0
        incorrect_predictions = []

        with torch.no_grad():
            self.model.eval()  # Set model to evaluation mode
            for test_data_x, test_data_y in test_dataloader:
                test_data_x = test_data_x.to(self.config.device)  # Move features to test device
                test_data_y = test_data_y.to(self.config.device)  # Move labels to test device

                output = self.model(test_data_x)  # Forward propagation
                pre_lab = torch.argmax(output, dim=1)  # Predicted labels
                test_corrects += torch.sum(pre_lab == test_data_y.data).item()  # Update correct prediction count
                test_num += test_data_x.size(0)  # Update total number of test samples

                # Collect incorrect predictions
                for i in range(test_data_x.size(0)):
                    if pre_lab[i] != test_data_y[i]:  # Record only incorrect predictions
                        incorrect_info = {
                            'predicted': self.classes[pre_lab[i].item()],
                            'actual': self.classes[test_data_y[i].item()]
                        }
                        incorrect_predictions.append(incorrect_info)
                        message = 'Predicted: ' + incorrect_info['predicted'] + ' | Actual: ' + incorrect_info['actual']
                        self.logger.info(message)

        test_acc = test_corrects / test_num  # Compute test accuracy
        message = f"Model testing completed! Accuracy: {test_acc:.2%}, Total samples: {test_num}, Correct predictions: {test_corrects}, Incorrect predictions: {len(incorrect_predictions)}"
        self.logger.info(message)

    def _log_message(self, log_type: str, message: str, **kwargs):
        """
        Helper method to generate log messages consistently
        :param log_type: Log type
        :param message: Log message
        :param kwargs: Additional parameters
        :return: Log dictionary
        """
        log_data = {
            'log_type': log_type,
            'message': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        log_data.update(kwargs)
        return log_data


if __name__ == '__main__':
    """
        Main program entry
    """
    config = TrainingConfig(model_name='ResNet',
                            pretrained_weights='../weights/ResNet/ResNet_model_98.14%.pth')  # Configure training parameters
    model_choose = ModelChoose(config)  # Initialize model selector
    model = model_choose.initialize_model()  # Initialize model
    tester = ModelTest(config, model_choose.logger, model)  # Create tester and start testing
    tester.test()