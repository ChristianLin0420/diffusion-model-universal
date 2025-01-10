import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models import DDPM, DDIM, ScoreBasedDiffusion, EnergyBasedDiffusion
from datasets.mnist_loader import MNISTDataset
from trainers import TRAINER_REGISTRY

# Model registry
MODEL_REGISTRY = {
    'ddpm': DDPM,
    'ddim': DDIM,
    'score_based': ScoreBasedDiffusion,
    'energy_based': EnergyBasedDiffusion
}

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_dataset(config: dict):
    """Get dataset based on configuration."""
    dataset_name = config['dataset']['name'].lower()
    
    if dataset_name == 'mnist':
        return MNISTDataset(
            data_dir=config['dataset']['data_dir'],
            image_size=config['dataset']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers']
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train diffusion models')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=['ddpm', 'ddim', 'score_based', 'energy_based'],
                      help='Type of diffusion model to train')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataset
    dataset = get_dataset(config)
    train_loader, test_loader = dataset.get_dataloaders()
    
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
        test_loader=test_loader,
        config={**config['model'], **config['training'], **config['logging']},
        device=device
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(config['training']['num_epochs'] - start_epoch)

if __name__ == '__main__':
    main() 