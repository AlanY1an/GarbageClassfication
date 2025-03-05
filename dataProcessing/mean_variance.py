import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def compute_mean_std(dataset_path):
    """
    Computes the mean and standard deviation of the dataset.

    :param dataset_path: Path to the dataset.
    :return: mean, std (as lists)
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize all images to 224x224
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    num_samples = 0

    for images, _ in dataloader:
        batch_samples = images.size(0)  # Number of images in batch
        images = images.view(batch_samples, 3, -1)  # Flatten the pixels
        mean += images.mean(dim=[0, 2]) * batch_samples  # Compute mean per channel
        std += images.std(dim=[0, 2]) * batch_samples  # Compute std per channel
        num_samples += batch_samples  # Update total number of samples

    mean /= num_samples  # Compute final mean
    std /= num_samples  # Compute final standard deviation

    print(f"Computed Mean: {mean.tolist()} | Computed Std: {std.tolist()}")
    return mean.tolist(), std.tolist()

# Example usage
dataset_path = "../data/split-data/train"  # Change to your dataset path
mean, std = compute_mean_std(dataset_path)
print("Mean:", mean)
print("Std:", std)
