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
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models import DDPM, DDIM, ScoreBasedDiffusion, EnergyBasedDiffusion
from datasets import DATASET_REGISTRY
from trainers import TRAINER_REGISTRY

# Model registry
MODEL_REGISTRY = {
    'ddpm': DDPM,
    'ddim': DDIM,
    'score_based': ScoreBasedDiffusion,
    'energy_based': EnergyBasedDiffusion
}

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

def get_dataset(config: dict):
    """Get dataset based on configuration.
    
    Creates and returns a dataset instance based on the configuration.
    The dataset class is looked up in the DATASET_REGISTRY.
    
    Args:
        config (dict): Configuration dictionary containing:
            - dataset: Dataset configuration with name, data_dir, image_size
            - training: Training configuration with batch_size, num_workers
            - val_split: Validation split ratio (for datasets that need splitting)
    
    Returns:
        Dataset: Instantiated dataset object.
        
    Raises:
        ValueError: If the specified dataset is not supported.
    """
    dataset_name = config['dataset']['name'].lower()
    dataset_class = DATASET_REGISTRY.get(dataset_name)
    
    if dataset_class is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    return dataset_class(
        data_dir=config['dataset']['data_dir'],
        image_size=config['dataset']['image_size'],
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        val_split=config['dataset'].get('val_split', 0.1)  # Default to 10% validation
    )

def main():
    """Main training function.
    
    Parses command-line arguments, sets up the model, dataset, and trainer,
    and runs the training loop. Supports resuming training from checkpoints.
    
    Command-line Arguments:
        --config (str): Path to the configuration file.
        --model_type (str): Type of diffusion model to train.
            Must be one of: ddpm, ddim, score_based, energy_based.
        --resume (str, optional): Path to checkpoint to resume training from.
        --eval_only (bool, optional): Only run evaluation on test set.
    
    The function performs the following steps:
    1. Parse command-line arguments
    2. Load configuration from YAML file
    3. Set up device (CPU/GPU)
    4. Create dataset and dataloaders
    5. Initialize model and trainer
    6. Resume from checkpoint if specified
    7. Run training loop or evaluation
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=list(MODEL_REGISTRY.keys()),
                      help='Type of diffusion model to train')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    dataset = get_dataset(config)
    train_loader, val_loader, test_loader = dataset.get_dataloaders()
    
    # Create model
    model_class = MODEL_REGISTRY.get(args.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model_class(config['model'])
    
    # Create trainer
    trainer_class = TRAINER_REGISTRY.get(args.model_type)
    if trainer_class is None:
        raise ValueError(f"No trainer found for model type: {args.model_type}")
    
    trainer = trainer_class(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config={**config['model'], **config['training'], **config['logging']},
        device=device
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
        print(f"Test Loss: {test_loss:.4f}")
    else:
        # Train model
        trainer.train(config['training']['num_epochs'] - start_epoch)
        
        # Final evaluation
        print("Running final evaluation...")
        test_loss = trainer.test()
        print(f"Final Test Loss: {test_loss:.4f}")

if __name__ == '__main__':
    main() 