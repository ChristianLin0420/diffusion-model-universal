"""Energy-based Diffusion Model Implementation.

This module implements the Energy-based Diffusion Model, which combines ideas from
energy-based models and diffusion models. It uses an energy network to model the
data distribution and implements Langevin dynamics sampling.

Key Features:
    - Energy-based modeling
    - Langevin dynamics sampling
    - Optional time conditioning
    - Flexible loss functions
    - Configurable sampling parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import EnergyBasedLoss, DiffusionLoss

class EnergyNet(nn.Module):
    """Energy network architecture.
    
    This class implements a neural network that models the energy function
    of the data distribution. It can optionally be conditioned on time
    for diffusion-like behavior.
    
    Args:
        in_channels (int): Number of input channels.
            Includes time embedding channels if time conditioning is used.
        model_channels (int): Base channel count for the model.
            The number of channels in intermediate layers will be multiples of this.
    
    Attributes:
        conv1, conv2, conv3 (nn.Conv2d): Convolutional layers
        norm1, norm2 (nn.GroupNorm): Group normalization layers
        dense (nn.Linear): Final dense layer for energy prediction
    """
    
    def __init__(self, in_channels: int, model_channels: int):
        """Initialize Energy network.
        
        Sets up the network architecture with convolutional layers and
        a final dense layer for energy prediction.
        """
        super().__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(model_channels, model_channels*2, 3, padding=1)
        self.conv3 = nn.Conv2d(model_channels*2, model_channels*4, 3, padding=1)
        
        # Normalization layers
        self.norm1 = nn.GroupNorm(8, model_channels)
        self.norm2 = nn.GroupNorm(8, model_channels*2)
        
        # Final dense layer for energy prediction
        self.dense = nn.Linear(model_channels*4, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing the energy.
        
        Processes the input through the network to compute the energy value.
        A lower energy indicates higher probability under the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The images to compute energy for.
        
        Returns:
            torch.Tensor: Energy values of shape [B].
                The computed energy for each input sample.
        """
        # Convolutional layers with activations
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.conv3(h))
        
        # Global average pooling
        h = h.mean(dim=[2, 3])
        
        # Final energy prediction
        return self.dense(h).squeeze(-1)

class EnergyBasedDiffusion(BaseDiffusion):
    """Energy-based diffusion model implementation.
    
    This class implements the energy-based diffusion model training and sampling
    procedures. It uses an energy network to model the data distribution and
    implements Langevin dynamics for sampling.
    
    Args:
        config (Dict): Model configuration containing:
            - num_timesteps (int): Number of diffusion steps
            - beta_start (float): Starting value for noise schedule
            - beta_end (float): Ending value for noise schedule
            - use_time_conditioning (bool): Whether to condition on time
            - in_channels (int): Number of input channels
            - model_channels (int): Base channel count for energy network
            - loss_type (str): Type of loss function to use
            - loss_config (Dict): Configuration for loss function
            - langevin_steps (int): Number of Langevin steps per noise level
            - langevin_step_size (float): Step size for Langevin dynamics
    
    Attributes:
        betas (torch.Tensor): Noise schedule
        alphas (torch.Tensor): 1 - betas
        alphas_cumprod (torch.Tensor): Cumulative product of alphas
        model (EnergyNet): Energy network
        loss_fn (EnergyBasedLoss or DiffusionLoss): Loss function
        langevin_steps (int): Number of Langevin steps
        langevin_step_size (float): Langevin step size
    """
    
    def __init__(self, config: Dict):
        """Initialize Energy-based diffusion model.
        
        Sets up the energy network and loss function according to the configuration.
        Supports both energy-based and diffusion-style losses.
        """
        super().__init__(config)
        
        # Model parameters
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        
        # Create noise schedule
        self.register_buffer('betas', torch.linspace(self.beta_start, self.beta_end, self.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Create energy network
        in_channels = config.get('in_channels', 3)
        if config.get('use_time_conditioning', True):
            in_channels += config.get('model_channels', 64)
            
        self.model = EnergyNet(
            in_channels=in_channels,
            model_channels=config.get('model_channels', 64)
        )
        
        # Setup loss functions
        loss_type = config.get('loss_type', 'energy_based')
        if loss_type == 'energy_based':
            self.loss_fn = EnergyBasedLoss(
                energy_scale=config.get('energy_scale', 1.0),
                regularization_weight=config.get('regularization_weight', 0.1)
            )
        else:
            self.loss_fn = DiffusionLoss(
                loss_type=loss_type,
                loss_config=config.get('loss_config', None)
            )
        
        # Langevin dynamics parameters
        self.langevin_steps = config.get('langevin_steps', 10)
        self.langevin_step_size = config.get('langevin_step_size', 0.01)
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute energy of the input.
        
        Evaluates the energy function on the input images, optionally
        conditioned on timesteps.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The images to compute energy for.
            t (Optional[torch.Tensor]): Optional timestep tensor of shape [B].
                Used for time conditioning if enabled.
        
        Returns:
            torch.Tensor: Energy values of shape [B].
        """
        return self.model(x, t)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the energy-based loss.
        
        Computes the loss using either energy-based contrastive divergence
        or a diffusion-style loss function.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W].
                The clean images to train on.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = x.shape[0]
        
        # Sample timestep
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
        # Get noisy samples
        noise = torch.randn_like(x)
        x_noisy = self._add_noise(x, t, noise)
        
        # Generate samples using Langevin dynamics
        x_fake = self._langevin_sampling(x_noisy, t)
        
        if isinstance(self.loss_fn, EnergyBasedLoss):
            # Use energy-based loss
            return self.loss_fn(self.forward, x, x_fake)
        else:
            # For other loss types, compute energy difference
            energy_real = self.forward(x, t)
            energy_fake = self.forward(x_fake, t)
            return self.loss_fn(energy_real, energy_fake, t)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples using annealed Langevin dynamics.
        
        Implements the annealed Langevin dynamics sampling procedure,
        which gradually denoises random noise using the energy function
        at decreasing noise levels.
        
        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
        
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W].
        """
        # Start from random noise
        x = torch.randn((batch_size, self.config['in_channels'],
                        self.config['image_size'], self.config['image_size']),
                       device=device)
        
        # Gradually denoise the samples
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device)
            
            # Langevin dynamics sampling
            x = self._langevin_sampling(x, t_batch)
            
            # Add noise for next step if not the last step
            if t > 0:
                noise = torch.randn_like(x)
                alpha_next = self.alphas_cumprod[t-1]
                alpha = self.alphas_cumprod[t]
                sigma = torch.sqrt((1 - alpha_next) / (1 - alpha)) * \
                       torch.sqrt(1 - alpha / alpha_next)
                x = torch.sqrt(alpha_next / alpha) * x + sigma * noise
        
        return x
    
    def _langevin_sampling(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform Langevin dynamics sampling.
        
        Runs multiple steps of Langevin dynamics to generate samples from
        the energy-based model at a given noise level.
        
        Args:
            x (torch.Tensor): Initial samples of shape [B, C, H, W].
            t (torch.Tensor): Timestep tensor of shape [B].
        
        Returns:
            torch.Tensor: Updated samples after Langevin dynamics.
        """
        x.requires_grad_(True)
        
        for _ in range(self.langevin_steps):
            # Compute energy gradient
            energy = self.forward(x, t)
            grad = torch.autograd.grad(energy.sum(), x)[0]
            
            # Langevin update
            noise = torch.randn_like(x)
            x = x - self.langevin_step_size * grad + \
                torch.sqrt(2 * self.langevin_step_size) * noise
            
            x = x.detach().requires_grad_(True)
        
        return x.detach()
    
    def _add_noise(self, x: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to the input according to the noise schedule.
        
        Args:
            x (torch.Tensor): Input images
            t (torch.Tensor): Timesteps
            noise (Optional[torch.Tensor]): Optional pre-generated noise
        
        Returns:
            torch.Tensor: Noisy images
        """
        if noise is None:
            noise = torch.randn_like(x)
            
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise 