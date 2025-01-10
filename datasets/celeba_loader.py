"""CelebA Dataset Loader.

This module provides a dataset loader for the CelebA face dataset.
It handles downloading, preprocessing, and loading of CelebA data with configurable
transformations and batch processing.

Key Features:
    - Automatic download and caching
    - Configurable image size and preprocessing
    - Center cropping for face alignment
    - Data augmentation for training
    - Train/val/test split support
    - Efficient batch loading with multiple workers
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple

class CelebADataset:
    """CelebA dataset loader with preprocessing.
    
    This class provides functionality to load and preprocess the CelebA dataset,
    including center cropping, resizing, normalization, and data augmentation.
    It handles high-resolution face images with appropriate transformations for
    training, validation, and testing.
    
    Args:
        data_dir (str): Directory to store the dataset. Defaults to "./data".
            The directory will be created if it doesn't exist.
        
        image_size (int): Size to resize images to. Defaults to 64.
            Both height and width will be resized to this value.
            Recommended to be a power of 2 (32, 64, 128, etc.).
        
        batch_size (int): Batch size for dataloaders. Defaults to 32.
            Used for train, validation, and test dataloaders.
        
        num_workers (int): Number of workers for dataloaders. Defaults to 4.
            More workers can speed up data loading but use more memory.
        
        crop_size (int): Size of center crop. Defaults to 178.
            This is the standard center crop size for CelebA to focus on faces.
    
    Attributes:
        transform (transforms.Compose): Transformation pipeline for training data,
            including center crop, resize, and random horizontal flips.
        val_transform (transforms.Compose): Transformation pipeline for validation data,
            without augmentation.
        test_transform (transforms.Compose): Transformation pipeline for test data,
            without augmentation.
    
    Example:
        >>> dataset = CelebADataset(image_size=64, batch_size=32)
        >>> train_loader, val_loader, test_loader = dataset.get_dataloaders()
        >>> for images, _ in train_loader:
        ...     # images will be RGB tensors of shape [32, 3, 64, 64]
        ...     pass
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        image_size: int = 64,  # CelebA typically uses larger images
        batch_size: int = 32,
        num_workers: int = 4,
        crop_size: int = 178  # Default center crop size for CelebA
    ):
        """Initialize CelebA dataset loader.
        
        Sets up the data directory and transformation pipelines for training,
        validation, and testing. The training pipeline includes center cropping,
        resizing, and data augmentation, while the validation and test pipelines
        only include necessary preprocessing.
        
        The center crop is applied first to focus on the face region, followed by
        resizing to the specified image size. For training, random horizontal flips
        are applied for augmentation.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.crop_size = crop_size
        
        # Define transforms for training data (with augmentation)
        self.transform = transforms.Compose([
            transforms.CenterCrop(crop_size),  # Center crop to focus on face
            transforms.Resize(image_size),  # Resize to specified size
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),  # Convert to tensor and scale to [0, 1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])
        
        # Define transforms for validation/test data (no augmentation)
        self.val_transform = self.test_transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get train, validation, and test dataloaders.
        
        Downloads the CelebA dataset if not already present in data_dir,
        applies the specified transformations, and creates DataLoader
        instances for training, validation, and test sets.
        
        The training set uses data augmentation and shuffling, while the
        validation and test sets use only basic preprocessing without augmentation.
        
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
        # Download and load training data
        train_dataset = datasets.CelebA(
            root=self.data_dir,
            split='train',  # Use training split
            download=True,
            transform=self.transform
        )
        
        # Load validation data
        val_dataset = datasets.CelebA(
            root=self.data_dir,
            split='valid',  # Use validation split
            download=True,
            transform=self.val_transform
        )
        
        # Load test data
        test_dataset = datasets.CelebA(
            root=self.data_dir,
            split='test',  # Use test split
            download=True,
            transform=self.test_transform
        )
        
        # Create dataloaders with appropriate settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # Shuffle training data
            num_workers=self.num_workers,
            pin_memory=True  # Speeds up data transfer to GPU
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle validation data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Don't shuffle test data
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader 