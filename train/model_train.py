import os
import time
import sys
import signal
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from train.model_choose import ModelChoose, TrainingConfig


class ModelTrainer:
    def __init__(self, config: TrainingConfig, logger, model):
        """
        Initializes the ModelTrainer class.
        :param config: Training configuration.
        :param logger: Logger instance.
        :param model: Model to be trained.
        """
        self.config = config
        self.logger = logger
        self.model = model

        # Store best model weights
        self.best_model_weights = self.model.state_dict()
        self.best_val_accuracy = 0.0
        self.best_val_loss = float("inf")
        self.early_stop_count = 0
        self.patience = 10  # Early stopping patience

        # Model save path
        self.model_save_path = os.path.join(
            self.config.weights_dir,
            f'{self.config.model_name}_model_{{:.2f}}%.pth'
        )

        # Training interruption flag
        self.training_interrupted = False

        # Compute dataset mean & std
        self.mean, self.std = [0.6581178307533264, 0.6164085865020752, 0.5857066512107849],[0.2586901783943176, 0.25833287835121155, 0.26854372024536133]

    def _compute_mean_std(self):
        """
        Computes the mean and standard deviation of the dataset.
        :return: mean, std (as lists)
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()  # Convert to tensor
        ])

        dataset = ImageFolder(root=self.config.train_data_path, transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        mean = torch.zeros(3)
        std = torch.zeros(3)
        num_samples = 0

        for images, _ in dataloader:
            batch_samples = images.size(0)
            images = images.view(batch_samples, 3, -1)
            mean += images.mean(dim=[0, 2]) * batch_samples
            std += images.std(dim=[0, 2]) * batch_samples
            num_samples += batch_samples

        mean /= num_samples
        std /= num_samples

        self.logger.info(f"Computed Mean: {mean.tolist()} | Computed Std: {std.tolist()}")
        return mean.tolist(), std.tolist()

    def _get_data_loaders(self):
        """
        Returns the train and validation data loaders.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        train_data = ImageFolder(self.config.train_data_path, transform=transform)
        val_data = ImageFolder(self.config.val_data_path, transform=transform)

        train_loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=1, shuffle=True)

        return train_loader, val_loader

    def _handle_interrupt(self, signum, frame):
        """
        Handles training interruption signal (Ctrl+C).
        """
        self.logger.warning("Training interrupted by user. Saving current best model...")
        self.training_interrupted = True

    def train(self):
        """
        Main training logic
        """
        # Register interrupt signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)

        start_time = time.time()  # Record start time
        train_loader, val_loader = self._get_data_loaders()  # Initialize data loaders
        criterion = nn.CrossEntropyLoss()  # Define loss function
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)  # Define optimizer

        train_metrics = {  # Initialize training and validation metrics
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': []
        }

        if self.config.pretrained_weights:
            self.logger.info(f"Loaded pretrained weights: {self.config.pretrained_weights}")

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")

        try:
            for epoch in range(self.config.num_epochs):
                if self.training_interrupted:
                    break

                epoch_start_time = time.time()  # Record start time for the current epoch
                epoch_log = f"{'=' * 25}Epoch {epoch + 1}/{self.config.num_epochs}{'=' * 25}"
                self.logger.info(epoch_log)

                # Training phase
                self.model.train()  # Set model to training mode
                total_train_loss, total_train_correct, total_train_samples = 0, 0, 0

                for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
                    if self.training_interrupted:
                        break

                    batch_x, batch_y = batch_x.to(self.config.device), batch_y.to(self.config.device)
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]  # Main output is at the first position
                        loss = criterion(main_output, batch_y)
                    elif hasattr(outputs, 'logits'):
                        loss = criterion(outputs.logits, batch_y)
                    else:
                        loss = criterion(outputs, batch_y)

                    loss.backward()
                    optimizer.step()

                    total_train_loss += loss.item() * batch_x.size(0)
                    total_train_correct += (outputs.argmax(1) == batch_y).sum().item()
                    total_train_samples += batch_x.size(0)

                    # Real-time progress display
                    progress = (batch_idx + 1) / len(train_loader) * 100
                    self._display_progress('Train', progress)

                if self.training_interrupted:
                    break

                train_loss = total_train_loss / total_train_samples if total_train_samples > 0 else 0
                train_accuracy = total_train_correct / total_train_samples if total_train_samples > 0 else 0

                # Validation phase
                self.model.eval()  # Set model to evaluation mode
                total_val_loss, total_val_correct, total_val_samples = 0, 0, 0

                with torch.no_grad():
                    for batch_idx, (batch_x, batch_y) in enumerate(val_loader):
                        if self.training_interrupted:
                            break

                        batch_x, batch_y = batch_x.to(self.config.device), batch_y.to(self.config.device)
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)

                        total_val_loss += loss.item() * batch_x.size(0)
                        total_val_correct += (outputs.argmax(1) == batch_y).sum().item()
                        total_val_samples += batch_x.size(0)

                        # Real-time progress display
                        progress = (batch_idx + 1) / len(val_loader) * 100
                        self._display_progress('Validation', progress)

                    if self.training_interrupted:
                        break

                val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else 0
                val_accuracy = total_val_correct / total_val_samples if total_val_samples > 0 else 0

                # If validation accuracy is higher, update the best model weights
                if val_accuracy > self.best_val_accuracy:
                    self.best_val_accuracy = val_accuracy  # Update best validation accuracy
                    self.best_model_weights = self.model.state_dict()  # Save best model weights
                    self.early_stop_count = 0  # Reset patience counter
                else:
                    self.early_stop_count += 1  # No improvement, increase counter

                # Stop if no improvement after `patience` epochs
                if self.early_stop_count >= self.patience:
                    self.logger.info(f"Early stopping triggered. No improvement in {self.patience} epochs.")
                    break  # Stop training early

                # Record metrics
                train_metrics['train_loss'].append(train_loss)
                train_metrics['val_loss'].append(val_loss)
                train_metrics['train_accuracy'].append(train_accuracy)
                train_metrics['val_accuracy'].append(val_accuracy)

                epoch_duration = time.time() - epoch_start_time  # Calculate epoch duration
                log_summary = (f"Epoch Summary: "
                               f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                               f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}, "
                               f"Epoch Duration: {epoch_duration:.2f} seconds")
                self.logger.info(log_summary)

                if self.training_interrupted:
                    break

        except Exception as e:
            self.logger.error(f"Training interrupted due to an error: {e}")
            self.training_interrupted = True

        finally:
            if self.training_interrupted:
                self.logger.warning("Training was interrupted. Saving best model so far...")

            self._save_best_model()  # Save the best model when training ends or an exception occurs

            total_time = time.time() - start_time
            final_status = "Interrupted" if self.training_interrupted else "Completed"
            self.logger.info(f"Training {final_status}. Total Time: {total_time:.2f} seconds")

            if not self.training_interrupted:
                self._plot_training_results(train_metrics)

    def _save_best_model(self):
        """
        Save the best model
        """
        os.makedirs(self.config.weights_dir, exist_ok=True)
        save_path = self.model_save_path.format(self.best_val_accuracy * 100)
        torch.save(self.best_model_weights, save_path)
        self.logger.info(f"Best model saved to {save_path}")

    @staticmethod
    def _plot_training_results(train_metrics):
        """
        Draw result
        :param train_metrics: Training and validation parameter
        """
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title('Loss Curves')
        plt.plot(train_metrics['train_loss'], label='Train Loss')
        plt.plot(train_metrics['val_loss'], label='Val Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Accuracy Curves')
        plt.plot(train_metrics['train_accuracy'], label='Train Accuracy')
        plt.plot(train_metrics['val_accuracy'], label='Val Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def _log_message(self, log_type: str, message: str, **kwargs):
        """
        Helper method to generate a unified log message.
        :param log_type: Type of log.
        :param message: Log message.
        :param kwargs: Additional parameters.
        :return: Log dictionary.
        """
        log_data = {
            'log_type': log_type,
            'message': message,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        log_data.update(kwargs)
        return log_data

    def _display_progress(self, stage: str, progress: float):
        """Display training/validation progress."""
        status_bar = f"{stage} Progress: [{'=' * int(progress / 5)}{'>' if progress < 100 else ''}{'.' * (20 - int(progress / 5))}] {progress:.2f}%"
        print(f"\r{status_bar}", end='', flush=True)
        if progress >= 100:
            print()  #

if __name__ == '__main__':
    """
    Main program entry point.
    """

    config = TrainingConfig(
        model_name='VGGNet',
        num_epochs=50,
        weights_dir='../weights/VGGNet',
        pretrained_weights='../weights/VGGNet/vgg19.pth')  # Configure training parameters
    model_choose = ModelChoose(config)  # Initialize model selector
    model = model_choose.initialize_model()  # Initialize model
    trainer = ModelTrainer(config, model_choose.logger, model)  # Create trainer and start training
    trainer.train()
