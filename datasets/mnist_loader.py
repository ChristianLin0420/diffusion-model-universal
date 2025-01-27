"""MNIST Dataset Loader.

This module provides a dataset loader for the MNIST handwritten digits dataset.
It handles downloading, preprocessing, and loading of MNIST data with configurable
transformations and batch processing.

Key Features:
    - Automatic download and caching
    - Configurable image size and preprocessing
    - Data augmentation for training
    - Train/val/test split support
    - Efficient batch loading with multiple workers
    - Automatic conversion to RGB format
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Optional, Tuple

class MNISTDataset:
    """MNIST dataset loader with preprocessing.
    
    This class provides functionality to load and preprocess the MNIST dataset,
    including resizing, normalization, and optional data augmentation. It converts
    the original grayscale images to RGB format for compatibility with RGB-based models.
    
    Args:
        data_dir (str): Directory to store the dataset. Defaults to "./data".
            The directory will be created if it doesn't exist.
        
        image_size (int): Size to resize images to. Defaults to 32.
            Both height and width will be resized to this value.
        
        batch_size (int): Batch size for dataloaders. Defaults to 32.
            Used for train, validation, and test dataloaders.
        
        num_workers (int): Number of workers for dataloaders. Defaults to 4.
            More workers can speed up data loading but use more memory.
        
        val_split (float): Fraction of training data to use for validation.
            Must be between 0 and 1. Defaults to 0.1 (10% validation).
    
    Attributes:
        transform (transforms.Compose): Transformation pipeline for training data.
        val_transform (transforms.Compose): Transformation pipeline for validation data.
        test_transform (transforms.Compose): Transformation pipeline for test data.
    
    Example:
        >>> dataset = MNISTDataset(image_size=64, batch_size=128)
        >>> train_loader, val_loader, test_loader = dataset.get_dataloaders()
        >>> for images, labels in train_loader:
        ...     # images will be RGB tensors of shape [128, 3, 64, 64]
        ...     pass
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 32,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.1
    ):
        """Initialize MNIST dataset loader.
        
        Sets up the data directory and transformation pipelines for training,
        validation, and testing.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        
        # Define transforms for training data (with augmentation)
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.Resize(image_size),  # Resize to specified size
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # Convert to RGB
        ])
        
        # Define transforms for validation/test data (no augmentation)
        self.val_transform = self.test_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test dataloaders.
        
        Downloads the MNIST dataset if not already present in data_dir,
        applies the specified transformations, splits the training data into
        train and validation sets, and creates DataLoader instances.
        
        Returns:
            tuple: A tuple containing:
                - train_loader (DataLoader): DataLoader for training data
                - val_loader (DataLoader): DataLoader for validation data
                - test_loader (DataLoader): DataLoader for test data
        
        Example:
            >>> train_loader, val_loader, test_loader = dataset.get_dataloaders()
            >>> print(f"Training batches: {len(train_loader)}")
            >>> print(f"Validation batches: {len(val_loader)}")
            >>> print(f"Test batches: {len(test_loader)}")
        """
        # Download and load training data
        full_train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        # Split into train and validation
        val_size = int(len(full_train_dataset) * self.val_split)
        train_size = len(full_train_dataset) - val_size
        
        train_dataset, val_dataset = random_split(
            full_train_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
        )
        
        # Override validation transform
        val_dataset.dataset.transform = self.val_transform
        
        # Download and load test data
        test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up data transfer to GPU
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
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
        
        return train_loader, val_loader, test_loader 