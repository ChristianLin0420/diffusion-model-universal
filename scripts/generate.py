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

from models.ddpm import DDPM

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Generate samples from trained DDPM model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
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
    model = DDPM(config['model'])
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