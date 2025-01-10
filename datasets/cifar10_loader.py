"""CIFAR-10 Dataset Loader.

This module provides a dataset loader for the CIFAR-10 natural images dataset.
It handles downloading, preprocessing, and loading of CIFAR-10 data with configurable
transformations and batch processing.

Key Features:
    - Automatic download and caching
    - Configurable image size and preprocessing
    - Data augmentation for training
    - Train/val/test split support
    - Efficient batch loading with multiple workers
    - Built-in normalization and augmentation
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from typing import Optional, Tuple

class CIFAR10Dataset:
    """CIFAR-10 dataset loader with preprocessing.
    
    This class provides functionality to load and preprocess the CIFAR-10 dataset,
    including resizing, normalization, and data augmentation. It handles the natural
    RGB images with appropriate transformations for training, validation, and testing.
    
    Args:
        data_dir (str): Directory to store the dataset.
        image_size (int): Size to resize images to.
        transforms (dict): Dictionary containing 'train' and 'eval' transform pipelines.
        split_ratios (dict): Dictionary containing train/val/test split ratios.
            Must contain 'train', 'val', and 'test' keys with float values that sum to 1.
    
    Attributes:
        train_dataset (Dataset): Training dataset with augmentation transforms
        val_dataset (Dataset): Validation dataset with basic transforms
        test_dataset (Dataset): Test dataset with basic transforms
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        transforms: dict,
        split_ratios: dict
    ):
        """Initialize CIFAR-10 dataset loader with separate transforms for training and evaluation."""
        self.data_dir = data_dir
        self.image_size = image_size
        self.transforms = transforms
        self.split_ratios = split_ratios
        
        # Validate transforms
        if not all(k in transforms for k in ['train', 'eval']):
            raise ValueError("transforms must contain both 'train' and 'eval' keys")
        
        # Validate split ratios
        if not all(k in split_ratios for k in ['train', 'val', 'test']):
            raise ValueError("split_ratios must contain 'train', 'val', and 'test' keys")
        if abs(sum(split_ratios.values()) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1")
        
        # Download and load the full training dataset
        train_val_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=None  # We'll set transforms after splitting
        )
        
        # Calculate split sizes
        total_size = len(train_val_dataset)
        train_size = int(total_size * split_ratios['train'])
        val_size = int(total_size * split_ratios['val'])
        test_size = total_size - train_size - val_size
        
        # Create train/val/test splits
        train_subset, val_subset, remaining = random_split(
            train_val_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create wrapped datasets with appropriate transforms
        self.train_dataset = TransformDataset(train_subset, transforms['train'])
        self.val_dataset = TransformDataset(val_subset, transforms['eval'])
        
        # Load the test dataset
        self.test_dataset = datasets.CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms['eval']
        )

class TransformDataset(Dataset):
    """Wrapper dataset that applies transforms to a subset."""
    
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test dataloaders.
        
        Downloads the CIFAR-10 dataset if not already present in data_dir,
        applies the specified transformations, splits the training data into
        train and validation sets, and creates DataLoader instances.
        
        Returns:
            tuple: A tuple containing:
                - train_loader (DataLoader): DataLoader for training data,
                  with shuffling and augmentation.
                - val_loader (DataLoader): DataLoader for validation data,
                  without shuffling or augmentation.
                - test_loader (DataLoader): DataLoader for test data,
                  without shuffling or augmentation.
        
        Example:
            >>> train_loader, val_loader, test_loader = dataset.get_dataloaders()
            >>> print(f"Training batches: {len(train_loader)}")
            >>> print(f"Validation batches: {len(val_loader)}")
            >>> print(f"Test batches: {len(test_loader)}")
        """
        # Create dataloaders with appropriate settings
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up data transfer to GPU
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader 