# Denoising Diffusion Probabilistic Models (DDPM)

This document provides detailed information about our DDPM implementation, including architecture, training process, and configuration options.


![DDPM](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/DDPM.png)

## Table of Contents
- [Overview](#overview)
- [Theory and Background](#theory-and-background)
- [Architecture](#architecture)
- [Training Process](#training-process)
- [Configuration](#configuration)
- [Layers](#layers)
- [Loss Functions](#loss-functions)
- [Benchmarking](#benchmarking)
- [Examples and Results](#examples-and-results)

## Overview

Our DDPM implementation follows the original paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) with several enhancements:
- Flexible U-Net architecture with configurable channels and blocks
- Advanced time embedding with transformer-style positional encoding
- Multiple loss function options with time-dependent weighting
- EMA (Exponential Moving Average) for stable training
- Comprehensive logging and benchmarking

## Theory and Background

### Diffusion Process
The diffusion process consists of two main parts:
1. **Forward Process (q)**: Gradually adds Gaussian noise to images
   ```
   q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_tI)
   ```

2. **Reverse Process (p)**: Learns to denoise images
   ```
   p(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
   ```

### Key Concepts
1. **Noise Schedule**:
   - Linear schedule: β_t increases linearly from β_1 to β_T
   - Cosine schedule: Alternative schedule for better sampling

2. **Time Steps**:
   - Default: T=1000 steps
   - Can be adjusted based on quality/speed trade-off

## Architecture

### U-Net Backbone
The U-Net architecture consists of:
- Initial convolution layer
- Time embedding module
- Downsampling path with attention
- Bottleneck with attention
- Upsampling path with attention
- Output convolution

Key components:
```python
# Initial convolution
self.initial_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding='same')

# Time embedding
self.time_embedding = TimeEmbedding(model_channels, model_channels * 4)

# Downsampling path
self.down_blocks = nn.ModuleList([
    ConvDownBlock(model_channels, model_channels, model_channels * 4),
    ConvDownBlock(model_channels, model_channels, model_channels * 4),
    ConvDownBlock(model_channels, model_channels * 2, model_channels * 4),
    AttentionDownBlock(model_channels * 2, model_channels * 2, model_channels * 4),
    ConvDownBlock(model_channels * 2, model_channels * 4, model_channels * 4)
])
```

### Shared Layers
The model uses several reusable layers:

1. **Attention Layers** (`models/layers/attention.py`):
   - `SelfAttentionBlock`: Multi-head self-attention with group normalization

2. **Residual Blocks** (`models/layers/residual.py`):
   - `ResidualBlock`: Basic building block with time embedding injection
   - `ConvDownBlock`: Downsampling with residual connections
   - `ConvUpBlock`: Upsampling with residual connections
   - `AttentionDownBlock`: Downsampling with attention
   - `AttentionUpBlock`: Upsampling with attention

3. **Embedding Layers** (`models/layers/embeddings.py`):
   - `TransformerPositionalEmbedding`: Sinusoidal position encoding
   - `TimeEmbedding`: Time step embedding with MLP projection

## Training Process

The training process involves:

1. **Forward Process**:
   - Add noise to images according to noise schedule
   - Sample random timesteps
   - Predict noise using U-Net model

2. **Loss Calculation**:
   - Support for multiple loss types (MSE, L1, Huber)
   - Time-dependent loss weighting
   - Optional perceptual and adversarial losses

3. **Optimization**:
   - Adam optimizer with configurable learning rate
   - Learning rate scheduling (cosine, linear, step, exponential)
   - EMA model updates for stable sampling

Example training configuration:
```yaml
training:
  num_epochs: 2000
  batch_size: 128
  learning_rate: 2e-4
  beta1: 0.9
  beta2: 0.999
  ema_decay: 0.9999
  scheduler:
    type: "cosine"
    warmup_steps: 1000
    min_lr: 1e-6
```

## Configuration

The DDPM model is highly configurable through YAML files. Key configuration sections:

### Model Configuration
```yaml
model_config:
  time_steps: 1000
  beta_start: 0.0001
  beta_end: 0.02
  image_size: 32
  image_channels: 3
  hidden_channels: 128
  num_res_blocks: 3
  dropout: 0.1
```

### Loss Configuration
```yaml
loss_config:
  loss_type: "mse"
  mse_weight: 1.0
  use_time_weighting: true
  time_weight_type: "snr"
  time_weight_params:
    min_weight: 0.1
    max_weight: 1.0
```

## Loss Functions

The implementation supports multiple loss functions through the `DiffusionLoss` class:

1. **Basic Losses**:
   - MSE Loss: L = ||x - x̂||²
   - L1 Loss: L = |x - x̂|
   - Huber Loss: Combination of L1 and L2

2. **Hybrid Loss**:
   - Combination of multiple loss types
   - Configurable weights for each component

3. **Time-dependent Weighting**:
   - SNR-based weighting
   - Linear weighting
   - Inverse time weighting

## Benchmarking

The model includes comprehensive benchmarking capabilities:

1. **Metrics**:
   - Fréchet Inception Distance (FID)
   - Inception Score (IS)
   - Structural Similarity (SSIM)
   - Peak Signal-to-Noise Ratio (PSNR)

2. **Sample Generation**:
   - Configurable number of samples
   - Intermediate step visualization
   - Noise-to-image process tracking

Example benchmark configuration:
```yaml
benchmark:
  n_samples: 50000
  batch_size: 128
  metrics:
    fid: true
    inception_score: true
    ssim: true
    psnr: true
  save_samples: true
  save_metrics: true
```

## Examples and Results

### Training Progress
The model shows consistent improvement across different datasets:

1. **MNIST**:
   - Clear digit formation by 100 epochs
   - Sharp, distinct digits by 500 epochs
   - High-quality samples by 1000 epochs

2. **CIFAR-10**:
   - Object shapes emerge by 200 epochs
   - Clear object boundaries by 1000 epochs
   - Detailed textures by 2000 epochs

3. **CelebA**:
   - Face structures visible by 300 epochs
   - Clear facial features by 1500 epochs
   - Fine details and realistic textures by 3000 epochs

### Sample Quality
Quality progression during training:

1. **Early Training (50 epochs)**:
   - Basic shapes and structures
   - Blurry details
   - Limited color diversity

2. **Mid Training (500 epochs)**:
   - Clear object boundaries
   - Improved textures
   - Better color distribution

3. **Final Results (2000 epochs)**:
   - Sharp, detailed features
   - Realistic textures
   - Natural color variations

### Interpolation
The model supports smooth interpolation between samples:
- Consistent feature transitions
- Stable intermediate representations
- Meaningful attribute interpolation

## Usage Examples

1. **Training**:
```bash
python scripts/train.py --config configs/ddpm_config.yaml --model_type ddpm
```

2. **Evaluation**:
```bash
python scripts/train.py --config configs/ddpm_config.yaml --model_type ddpm --eval_only --benchmark
```

3. **Sample Generation**:
```bash
python scripts/generate.py --config configs/ddpm_config.yaml --model_type ddpm --num_samples 16
```

## Common Issues and Solutions

1. **Out of Memory**:
   - Reduce batch size
   - Use gradient checkpointing
   - Optimize model size

2. **Training Instability**:
   - Adjust learning rate
   - Use EMA
   - Modify noise schedule

3. **Poor Sample Quality**:
   - Increase number of timesteps
   - Adjust model architecture
   - Fine-tune loss weights 