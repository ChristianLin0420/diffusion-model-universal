import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import ScoreMatchingLoss, DiffusionLoss

class ScoreNet(UNet):
    """Score network architecture based on U-Net."""
    
    def __init__(self, in_channels: int, model_channels: int, out_channels: int,
                 num_scales: int = 1000):
        super().__init__(in_channels, model_channels, out_channels)
        
        # Modify time embedding for continuous sigma
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels * 4)
        )
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing the score.
        
        Args:
            x: Input tensor
            sigma: Noise level
        """
        # Embed noise level (sigma)
        sigma_embed = torch.log(sigma).view(-1, 1)
        t_emb = self.time_embed(sigma_embed)[:, :, None, None]
        
        # Initial convolution
        h = self.conv_in(x)
        h1 = F.silu(h)
        
        # Downsampling
        h2 = F.silu(self.down1(h1))
        h3 = F.silu(self.down2(h2))
        
        # Middle
        h3 = h3 + t_emb
        h3 = self.middle(h3)
        
        # Upsampling with skip connections
        h = F.silu(self.up1(torch.cat([h3, h2], dim=1)))
        h = F.silu(self.up2(torch.cat([h, h1], dim=1)))
        
        return self.conv_out(h)

class ScoreBasedDiffusion(BaseDiffusion):
    """Score-based diffusion model implementation."""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Model parameters
        self.sigma_min = config.get('sigma_min', 0.01)
        self.sigma_max = config.get('sigma_max', 50.0)
        self.num_scales = config.get('num_scales', 1000)
        self.beta = config.get('beta', 1.0)  # Temperature parameter
        
        # Create score network
        self.model = ScoreNet(
            in_channels=config.get('in_channels', 3),
            model_channels=config.get('model_channels', 64),
            out_channels=config.get('in_channels', 3),
            num_scales=self.num_scales
        )
        
        # Setup loss functions
        loss_type = config.get('loss_type', 'score_matching')
        if loss_type == 'score_matching':
            self.loss_fn = ScoreMatchingLoss(
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max
            )
        else:
            self.loss_fn = DiffusionLoss(
                loss_type=loss_type,
                loss_config=config.get('loss_config', None)
            )
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Compute score function."""
        return self.model(x, sigma)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the score matching loss."""
        batch_size = x.shape[0]
        
        # Sample noise levels
        log_sigma = torch.rand(batch_size, device=x.device) * \
                   (np.log(self.sigma_max) - np.log(self.sigma_min)) + np.log(self.sigma_min)
        sigma = torch.exp(log_sigma)
        
        # Compute score
        score = self.forward(x, sigma)
        
        if isinstance(self.loss_fn, ScoreMatchingLoss):
            return self.loss_fn(score, x, sigma)
        else:
            # For other loss types, we need to generate noisy samples
            noise = torch.randn_like(x)
            x_noisy = x + sigma.view(-1, 1, 1, 1) * noise
            score_pred = self.forward(x_noisy, sigma)
            target = -noise / sigma.view(-1, 1, 1, 1)
            return self.loss_fn(score_pred, target, None)  # timesteps not used for non-score matching losses
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples using Langevin dynamics."""
        # Initialize from noise
        x = torch.randn((batch_size, self.config['in_channels'],
                        self.config['image_size'], self.config['image_size']),
                       device=device) * self.sigma_max
        
        # Annealing schedule
        sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_max), np.log(self.sigma_min),
            self.num_scales, device=device
        ))
        
        # Langevin dynamics
        for sigma in sigmas:
            sigma_batch = sigma.repeat(batch_size)
            
            # Multiple steps of Langevin dynamics for each noise level
            for _ in range(self.config.get('langevin_steps', 10)):
                # Compute score
                score = self.forward(x, sigma_batch)
                
                # Langevin dynamics step
                noise = torch.randn_like(x)
                step_size = (sigma ** 2) * 2 * self.beta
                x = x + step_size * score + torch.sqrt(2 * step_size) * noise
        
        return x
    
    def _get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert discrete timesteps to continuous noise levels."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t.float() / self.num_scales) 