# Diffusion Models Universal

A comprehensive PyTorch-based framework for training and experimenting with various diffusion models. This project provides a modular and flexible implementation of multiple diffusion model variants, including DDPM, DDIM, Score-based, and Energy-based models.

## Features

### Multiple Model Implementations
- **[DDPM (Denoising Diffusion Probabilistic Models)](docs/ddpm.md)**
  - Standard diffusion model with forward and reverse processes
  - Configurable noise schedule
  - [Detailed Documentation](docs/ddpm.md)
  
- **DDIM (Denoising Diffusion Implicit Models)**
  - Accelerated sampling with fewer steps
  - Deterministic or stochastic sampling options
  
- **Score-based Diffusion**
  - Score matching with Langevin dynamics
  - Continuous noise schedule
  - Configurable temperature parameters
  
- **Energy-based Diffusion**
  - Energy-based modeling with annealed Langevin dynamics
  - Gradient penalty regularization
  - Time conditioning options

### Supported Datasets
- **MNIST**
  - Standard 28x28 grayscale images
  - Automatically converted to RGB and resized
  - Basic augmentation with normalization
  
- **CIFAR-10**
  - 32x32 RGB natural images
  - 10 classes of objects
  - Includes random horizontal flips
  - Normalized to [-1, 1] range
  
- **CelebA**
  - High-quality celebrity face images
  - Center-cropped and resized
  - Supports different image sizes (default: 64x64)
  - Includes standard preprocessing and augmentation

### Flexible Loss Functions
All models support multiple loss functions that can be configured via YAML:
- MSE Loss
- L1 Loss
- Huber Loss
- Hybrid Loss (weighted combination)
- Time-dependent weighting
- Model-specific losses (Score Matching, Energy-based)

### Dataset Support
- MNIST (default)
- Extensible for other datasets (CIFAR-10, CelebA, etc.)
- Easy-to-add custom datasets

### Training Features
- Configurable training parameters
- Checkpoint saving and loading
- Sample generation during training
- Wandb integration for experiment tracking
- Multi-GPU support

## Project Structure
```
├── models/          # Model implementations
│   ├── ddpm.py
│   ├── ddim.py
│   ├── score_based.py
│   └── energy_based.py
├── datasets/        # Dataset loaders
├── trainers/        # Training implementations
├── utils/          # Helper functions
├── configs/        # Configuration files
├── scripts/        # Training and generation scripts
└── tests/          # Unit tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diffusion-model-universal.git
cd diffusion-model-universal
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training Models

1. Choose or modify a configuration file from `configs/`:
   - `ddpm_config.yaml`
   - `ddim_config.yaml`
   - `score_based_config.yaml`
   - `energy_based_config.yaml`

2. Start training:
```bash
# Train DDPM
python scripts/train.py --config configs/ddpm_config.yaml --model_type ddpm

# Train DDIM
python scripts/train.py --config configs/ddim_config.yaml --model_type ddim

# Train Score-based model
python scripts/train.py --config configs/score_based_config.yaml --model_type score_based

# Train Energy-based model
python scripts/train.py --config configs/energy_based_config.yaml --model_type energy_based

# Resume training from checkpoint
python scripts/train.py --config configs/ddpm_config.yaml --model_type ddpm --resume path/to/checkpoint.pt
```

### Generating Samples
```bash
python scripts/generate.py --config configs/ddpm_config.yaml --model_type ddpm --checkpoint path/to/checkpoint.pt --num_samples 16
```

### Configuration Guide

#### Loss Configuration
Each model supports flexible loss functions that can be configured in the YAML files:

1. **Basic Loss Types**:
```yaml
model:
  loss_type: 'mse'  # Options: 'mse', 'l1', 'huber'
  loss_config:
    reduction: 'mean'  # Options: 'mean', 'sum', 'none'
```

2. **Hybrid Loss**:
```yaml
model:
  loss_type: 'hybrid'
  loss_config:
    weights:
      mse: 0.6
      l1: 0.3
      huber: 0.1
```

3. **Time-weighted Loss**:
```yaml
model:
  loss_config:
    time_weights:
      type: 'linear'  # or 'exponential'
      max_timesteps: 1000
      beta: 0.1  # for exponential weighting
```

4. **Model-specific Losses**:
```yaml
# Score-based model
model:
  loss_type: 'score_matching'
  loss_config:
    sigma_min: 0.01
    sigma_max: 50.0

# Energy-based model
model:
  loss_type: 'energy_based'
  loss_config:
    energy_scale: 1.0
    regularization_weight: 0.1
```

#### Dataset-specific Configurations
Example configurations are provided for each dataset:

1. **MNIST Configuration**:
```yaml
dataset:
  name: "mnist"
  data_dir: "./data"
  image_size: 32
```

2. **CIFAR-10 Configuration**:
```yaml
dataset:
  name: "cifar10"
  data_dir: "./data"
  image_size: 32  # Native CIFAR-10 size
```

3. **CelebA Configuration**:
```yaml
dataset:
  name: "celeba"
  data_dir: "./data"
  image_size: 64  # Can be adjusted based on needs
  crop_size: 178  # CelebA-specific center crop
```

## Extending the Framework

### Adding New Models
1. Create a new model file in `models/`
2. Inherit from `BaseDiffusion`
3. Implement required methods: `forward`, `loss_function`, `sample`
4. Add model to `MODEL_REGISTRY` in `train.py`
5. Create corresponding configuration file

### Adding New Datasets
1. Create a new dataset loader in `datasets/`
2. Implement data preprocessing and augmentation
3. Add dataset to `get_dataset()` in `train.py`

### Adding New Loss Functions
1. Add new loss implementation to `utils/losses.py`
2. Update `DiffusionLoss` class with new loss type
3. Add corresponding configuration options

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 