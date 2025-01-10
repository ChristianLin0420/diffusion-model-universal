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
from datasets import get_dataset
from trainers import TRAINER_REGISTRY
from utils.benchmarks import DiffusionBenchmark
from utils.config_utils import print_config, load_config

# Model registry
MODEL_REGISTRY = {
    'ddpm': DDPM,
    'ddim': DDIM,
    'score_based': ScoreBasedDiffusion,
    'energy_based': EnergyBasedDiffusion
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