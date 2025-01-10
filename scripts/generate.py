"""Sample Generation Script for Diffusion Models.

This script provides a command-line interface for generating samples from trained
diffusion models. It loads a trained model checkpoint and generates a specified
number of samples, saving them both individually and as a grid.

Key Features:
    - Command-line interface for sample generation
    - YAML configuration file support
    - Checkpoint loading
    - Individual sample saving
    - Grid visualization
    - Device (CPU/GPU) configuration
    - Configurable number of samples
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys
from torchvision.utils import save_image
import os

# Add project root to path
project_root = str(Path(__file__).parent.parent)
sys.path.append(project_root)

from models import DDPM, DDIM, ScoreBasedDiffusion, EnergyBasedDiffusion

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
            The file should contain model and device configuration sections.
    
    Returns:
        dict: Loaded configuration dictionary.
        
    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the config file is invalid.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main generation function.
    
    Parses command-line arguments, loads the model checkpoint, and generates
    samples. The samples are saved both individually and as a grid visualization.
    
    Command-line Arguments:
        --config (str): Path to the configuration file.
        --model_type (str): Type of diffusion model to use.
            Must be one of: ddpm, ddim, score_based, energy_based.
        --checkpoint (str): Path to model checkpoint.
        --num_samples (int, optional): Number of samples to generate.
            Defaults to 16.
        --output_dir (str, optional): Output directory for samples.
            Defaults to 'generated_samples'.
    
    The function performs the following steps:
    1. Parse command-line arguments
    2. Load configuration from YAML file
    3. Set up device (CPU/GPU)
    4. Load model from checkpoint
    5. Generate specified number of samples
    6. Save samples individually and as a grid
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate samples from trained diffusion model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_type', type=str, required=True,
                      choices=list(MODEL_REGISTRY.keys()),
                      help='Type of diffusion model to use')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--output_dir', type=str, default='generated_samples', help='Output directory for samples')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['cuda'] else 'cpu')
    print(f'Using device: {device}')
    
    # Create model and load checkpoint
    model_class = MODEL_REGISTRY.get(args.model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model_class(config['model'])
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f'Generating {args.num_samples} samples...')
    with torch.no_grad():
        # Generate samples
        samples = model.sample(args.num_samples, device)
        
        # Save individual samples
        for i, sample in enumerate(samples):
            save_image(
                sample,
                os.path.join(args.output_dir, f'sample_{i}.png'),
                normalize=True,
                value_range=(-1, 1)
            )
        
        # Save grid of samples
        save_image(
            samples,
            os.path.join(args.output_dir, 'samples_grid.png'),
            nrow=int(args.num_samples ** 0.5),
            normalize=True,
            value_range=(-1, 1)
        )
    
    print(f'Samples saved to {args.output_dir}')

if __name__ == '__main__':
    main() 