import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import EnergyBasedLoss, DiffusionLoss

class EnergyNet(nn.Module):
    """Energy network architecture."""
    
    def __init__(self, in_channels: int, model_channels: int):
        super().__init__()
        
        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, 3, padding=1),
            nn.GroupNorm(8, model_channels),
            nn.SiLU(),
            
            nn.Conv2d(model_channels, model_channels * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, model_channels * 2),
            nn.SiLU(),
            
            nn.Conv2d(model_channels * 2, model_channels * 4, 4, stride=2, padding=1),
            nn.GroupNorm(8, model_channels * 4),
            nn.SiLU(),
            
            nn.Conv2d(model_channels * 4, model_channels * 8, 4, stride=2, padding=1),
            nn.GroupNorm(8, model_channels * 8),
            nn.SiLU(),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(model_channels * 8, 1)
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
    
    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute energy of the input.
        
        Args:
            x: Input tensor
            t: Optional timestep tensor
            
        Returns:
            Energy values
        """
        if t is not None:
            # Time embedding
            t_emb = self.time_embed(t.view(-1, 1))
            t_emb = t_emb.view(t_emb.shape[0], -1, 1, 1)
            x = torch.cat([x, t_emb.expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
        
        return self.encoder(x)

class EnergyBasedDiffusion(BaseDiffusion):
    """Energy-based diffusion model implementation."""
    
    def __init__(self, config: Dict):
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
        """Compute energy of the input."""
        return self.model(x, t)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the energy-based loss."""
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
        """Generate samples using annealed Langevin dynamics."""
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
        """Perform Langevin dynamics sampling."""
        x.requires_grad_(True)
        
        for _ in range(self.langevin_steps):
            # Compute energy gradient
            energy = self.forward(x, t)
            grad = torch.autograd.grad(energy.sum(), x)[0]
            
            # Update samples
            noise = torch.randn_like(x)
            x = x - self.langevin_step_size * grad + \
                torch.sqrt(2 * self.langevin_step_size) * noise
            
            x = x.detach().requires_grad_(True)
        
        return x.detach()
    
    def _add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to the input according to the noise schedule."""
        if noise is None:
            noise = torch.randn_like(x)
            
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise 