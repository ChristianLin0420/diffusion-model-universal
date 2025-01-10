import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple

class MNISTDataset:
    """MNIST dataset loader with preprocessing."""
    
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 32,
        num_workers: int = 4
    ):
        """
        Initialize MNIST dataset loader.
        
        Args:
            data_dir (str): Directory to store the dataset
            image_size (int): Size to resize images to
            batch_size (int): Batch size for dataloaders
            num_workers (int): Number of workers for dataloaders
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to 3 channels
        ])
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Get train and test dataloaders.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Download and load training data
        train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Download and load test data
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, test_loader 