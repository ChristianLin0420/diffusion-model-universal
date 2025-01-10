"""Training Script for Diffusion Models.

This script provides a command-line interface for training various diffusion models
on different datasets. It supports multiple model architectures (DDPM, DDIM,
Score-based, Energy-based) and datasets (MNIST, CIFAR-10, CelebA).

Key Features:
    - Command-line interface for model training
    - YAML configuration file support
    - Multiple model architectures
    - Multiple dataset options
    - Train/val/test split support
    - Checkpoint saving and loading
    - Device (CPU/GPU) configuration
    - Training progress logging
    - Model benchmarking
    - Multi-GPU training support
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import yaml
import argparse
from pathlib import Path
import sys
import json
import os
import torchvision.transforms as T

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models import DDPM, DDIM, ScoreBasedDiffusion, EnergyBasedDiffusion
from datasets import DATASET_REGISTRY
from trainers import TRAINER_REGISTRY
from utils.benchmarks import DiffusionBenchmark

# Model registry
MODEL_REGISTRY = {
    'ddpm': DDPM,
    'ddim': DDIM,
    'score_based': ScoreBasedDiffusion,
    'energy_based': EnergyBasedDiffusion
}

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

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
            The file should contain dataset, model, training, and logging sections.
    
    Returns:
        dict: Loaded configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data_config(config_path: str, dataset_name: str) -> dict:
    """Load dataset configuration from the data config file.
    
    Args:
        config_path (str): Path to the data configuration file.
        dataset_name (str): Name of the dataset to load configuration for.
    
    Returns:
        dict: Dataset configuration dictionary.
        
    Raises:
        ValueError: If the dataset is not found in the config file.
    """
    with open(config_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    if dataset_name not in data_config['datasets']:
        raise ValueError(f"Dataset {dataset_name} not found in {config_path}")
    
    return data_config['datasets'][dataset_name]

def print_config(title: str, config: dict):
    """Print configuration in a formatted way.
    
    Args:
        title (str): Title of the configuration section
        config (dict): Configuration dictionary to print
    """
    print("\n" + "="*50)
    print(f"{title}:")
    print("="*50)
    
    def _print_dict(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("\n" + " "*indent + f"{key}:")
                _print_dict(value, indent + 2)
            else:
                print(" "*indent + f"{key}: {value}")
    
    _print_dict(config)
    print("="*50 + "\n")

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
    data_config_path = Path(project_root) / 'configs' / 'data_config.yaml'
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

def setup_distributed(rank: int, world_size: int):
    """Set up distributed training environment.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def train_process(rank: int, world_size: int, args: argparse.Namespace):
    """Training process for distributed training.
    
    Args:
        rank (int): Process rank.
        world_size (int): Total number of processes.
        args (argparse.Namespace): Command line arguments.
    """
    # Set up distributed environment
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Load configuration
    config = load_config(args.config)
    
    # Print configuration from main process only
    if rank == 0:
        print_config("Main Configuration", config)
    
    # Set device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Create dataset
    train_loader, val_loader, test_loader = get_dataset(config, world_size, rank)
    
    # Create model
    model_class = MODEL_REGISTRY.get(args.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model_class(config['model_config'])
    
    # Create trainer
    trainer_class = TRAINER_REGISTRY.get(args.model_type)
    if trainer_class is None:
        raise ValueError(f"No trainer found for model type: {args.model_type}")
    
    trainer = trainer_class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={**config['model_config'], **config['training'], **config['logging']},
        device=device,
        rank=rank,
        world_size=world_size
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Run evaluation or training
    if args.eval_only:
        print("Running evaluation...")
        test_loss = trainer.test()
        if rank == 0:
            print(f"Test Loss: {test_loss:.4f}")
        
        if args.benchmark and rank == 0:
            print("Running benchmarks...")
            benchmark = DiffusionBenchmark(
                device=device,
                n_samples=config.get('benchmark', {}).get('n_samples', 50000),
                batch_size=config['training']['batch_size']
            )
            
            metrics = benchmark.evaluate(model, test_loader)
            
            # Save benchmark results
            output_dir = Path(config['logging']['output_dir'])
            benchmark_path = output_dir / 'benchmark_results.json'
            
            print("\nBenchmark Results:")
            print(f"FID Score: {metrics['fid']:.2f}")
            print(f"Inception Score: {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
            print(f"SSIM: {metrics['ssim']:.4f}")
            print(f"PSNR: {metrics['psnr']:.2f}")
            
            with open(benchmark_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"\nBenchmark results saved to {benchmark_path}")
    else:
        # Train model
        trainer.train(config['training']['num_epochs'] - start_epoch)
        
        # Final evaluation
        if rank == 0:
            print("Running final evaluation...")
        test_loss = trainer.test()
        if rank == 0:
            print(f"Final Test Loss: {test_loss:.4f}")
        
        if args.benchmark and rank == 0:
            print("Running benchmarks...")
            benchmark = DiffusionBenchmark(
                device=device,
                n_samples=config.get('benchmark', {}).get('n_samples', 50000),
                batch_size=config['training']['batch_size']
            )
            
            metrics = benchmark.evaluate(model, test_loader)
            
            # Save benchmark results
            output_dir = Path(config['logging']['output_dir'])
            benchmark_path = output_dir / 'benchmark_results.json'
            
            print("\nBenchmark Results:")
            print(f"FID Score: {metrics['fid']:.2f}")
            print(f"Inception Score: {metrics['is_mean']:.2f} ± {metrics['is_std']:.2f}")
            print(f"SSIM: {metrics['ssim']:.4f}")
            print(f"PSNR: {metrics['psnr']:.2f}")
            
            with open(benchmark_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            print(f"\nBenchmark results saved to {benchmark_path}")
    
    # Clean up distributed environment
    if world_size > 1:
        cleanup_distributed()

def main():
    """Main entry point for training.
    
    Parses command-line arguments and launches distributed training
    if multiple GPUs are available.
    
    Command-line Arguments:
        --config (str): Path to the configuration file.
        --model_type (str): Type of diffusion model to train.
            Must be one of: ddpm, ddim, score_based, energy_based.
        --resume (str, optional): Path to checkpoint to resume training from.
        --eval_only (bool, optional): Only run evaluation on test set.
        --benchmark (bool, optional): Run benchmarking during evaluation.
        --num_gpus (int, optional): Number of GPUs to use. Defaults to all available.
    """
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=list(MODEL_REGISTRY.keys()),
                      help='Type of diffusion model to train')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarking during evaluation')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Determine number of GPUs to use
    num_gpus = args.num_gpus if args.num_gpus is not None else torch.cuda.device_count()
    
    if num_gpus > 1:
        # Launch distributed training
        mp.spawn(
            train_process,
            args=(num_gpus, args),
            nprocs=num_gpus,
            join=True
        )
    else:
        # Single GPU or CPU training
        train_process(0, 1, args)

if __name__ == '__main__':
    main() 