"""Utility functions for dataset handling.

This module provides utility functions for creating datasets, data loaders,
and data transformations.
"""

import torch
import torchvision.transforms as T
from pathlib import Path
from torch.utils.data.distributed import DistributedSampler

from .registry import DATASET_REGISTRY
from utils.config_utils import load_data_config, print_config

def create_transforms(transform_configs, mean, std, is_train=False):
    """Create a composition of transforms from config.
    
    Args:
        transform_configs (list): List of transform configurations.
        mean (list): Normalization mean values.
        std (list): Normalization standard deviation values.
        is_train (bool): Whether to include training-specific transforms.
    
    Returns:
        torchvision.transforms.Compose: Composition of transforms.
    """
    transforms = []
    
    for transform in transform_configs:
        if transform['name'] == 'center_crop':
            transforms.append(T.CenterCrop(transform['size']))
        elif transform['name'] == 'resize':
            transforms.append(T.Resize(transform['size']))
        elif transform['name'] == 'random_horizontal_flip':
            if is_train and transform.get('probability', 0.5) > 0:
                transforms.append(T.RandomHorizontalFlip(p=transform.get('probability', 0.5)))
        elif transform['name'] == 'random_vertical_flip':
            if is_train and transform.get('probability', 0.5) > 0:
                transforms.append(T.RandomVerticalFlip(p=transform.get('probability', 0.5)))
        elif transform['name'] == 'random_rotation':
            if is_train:
                transforms.append(T.RandomRotation(transform.get('degrees', 10)))
        elif transform['name'] == 'color_jitter':
            if is_train:
                transforms.append(T.ColorJitter(
                    brightness=transform.get('brightness', 0),
                    contrast=transform.get('contrast', 0),
                    saturation=transform.get('saturation', 0),
                    hue=transform.get('hue', 0)
                ))
        elif transform['name'] == 'random_crop':
            if is_train:
                transforms.append(T.RandomCrop(
                    transform['size'],
                    padding=transform.get('padding', None),
                    padding_mode=transform.get('padding_mode', 'constant')
                ))
        elif transform['name'] == 'normalize':
            transforms.append(T.Normalize(mean=mean, std=std))
        elif transform['name'] == 'to_tensor':
            transforms.append(T.ToTensor())
        elif transform['name'] == 'grayscale':
            transforms.append(T.Grayscale(num_output_channels=transform.get('num_channels', 1)))
    
    # Always ensure ToTensor is applied first if not explicitly specified
    if not any(t['name'] == 'to_tensor' for t in transform_configs):
        transforms.insert(0, T.ToTensor())
    
    return T.Compose(transforms)

def get_dataset(config: dict, world_size: int = 1, rank: int = 0):
    """Get dataset based on configuration.
    
    Creates and returns a dataset instance based on the configuration.
    The dataset class is looked up in the DATASET_REGISTRY.
    
    Args:
        config (dict): Configuration dictionary containing:
            - data: Data configuration with dataset name and parameters
            - training: Training configuration
        world_size (int): Total number of processes for distributed training.
        rank (int): Process rank for distributed training.
    
    Returns:
        Dataset: Instantiated dataset object.
        
    Raises:
        ValueError: If the specified dataset is not supported.
    """
    dataset_name = config['data']['dataset'].lower()
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    
    if dataset_class is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    # Load dataset-specific configuration from data_config.yaml
    data_config_path = Path('configs/data_config.yaml')
    data_config = load_data_config(str(data_config_path), dataset_name)
    
    # Print dataset configuration
    if rank == 0:  # Only print from main process
        print_config("Dataset Configuration", data_config)
        print_config("Data Configuration from Main Config", config['data'])
    
    # Create transforms for training and evaluation
    train_transforms = create_transforms(
        data_config['transforms'],
        data_config['mean'],
        data_config['std'],
        is_train=True
    )
    
    eval_transforms = create_transforms(
        data_config['transforms'],
        data_config['mean'],
        data_config['std'],
        is_train=False
    )
    
    # Initialize dataset with all required parameters
    dataset_params = {
        'data_dir': Path(config['data']['data_dir']) / dataset_name.lower(),
        'image_size': config['data']['image_size'],
        'transforms': {
            'train': train_transforms,
            'eval': eval_transforms
        },
        'split_ratios': data_config['splits']
    }
    
    # Add optional parameters if they exist
    if 'crop_size' in data_config:
        dataset_params['crop_size'] = data_config['crop_size']
    
    dataset = dataset_class(**dataset_params)
    
    # Create dataloaders with the specified configuration
    if world_size > 1:
        train_sampler = DistributedSampler(
            dataset.train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            dataset.val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        test_sampler = DistributedSampler(
            dataset.test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
        
        train_loader = torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=train_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            dataset.val_dataset,
            batch_size=config['training']['batch_size'],
            sampler=val_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=config['training']['batch_size'],
            sampler=test_sampler,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    # For non-distributed training
    return {
        'train': torch.utils.data.DataLoader(
            dataset.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            dataset.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            dataset.test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=True
        )
    } 