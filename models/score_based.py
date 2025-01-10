"""Score-based Diffusion Model Implementation.

This module implements the Score-based Diffusion Model as described in the paper
"Score-Based Generative Modeling through Stochastic Differential Equations" by
Song et al. It provides a continuous-time formulation of diffusion models using
score matching and stochastic differential equations.

Key Features:
    - Continuous noise schedule
    - Score matching training objective
    - Noise-conditional score network
    - Annealed Langevin dynamics sampling
    - Flexible noise level range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import ScoreMatchingLoss, DiffusionLoss

class ScoreNet(UNet):
    """Score network architecture based on U-Net.
    
    This class extends the U-Net architecture to predict score functions
    (gradients of log density) conditioned on continuous noise levels.
    The time embedding is modified to handle continuous sigma values.
    
    Args:
        in_channels (int): Number of input channels.
            Usually 3 for RGB images.
        model_channels (int): Base channel count for the model.
            The number of channels in intermediate layers will be multiples of this.
        out_channels (int): Number of output channels.
            Usually same as in_channels.
        num_scales (int): Number of noise scales. Defaults to 1000.
            Used to determine the range of noise levels.
    
    Attributes:
        time_embed (nn.Sequential): Modified time embedding network for continuous sigma.
            Takes log(sigma) as input and outputs time embeddings.
    """
    
    def __init__(self, in_channels: int, model_channels: int, out_channels: int,
                 num_scales: int = 1000):
        """Initialize Score network.
        
        Sets up the U-Net architecture with a modified time embedding
        network that handles continuous noise levels (sigma).
        """
        super().__init__(in_channels, model_channels, out_channels)
        
        # Modify time embedding for continuous sigma
        self.time_embed = nn.Sequential(
            nn.Linear(1, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels * 4)
        )
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Forward pass computing the score.
        
        Predicts the score function (gradient of log density) for the input
        at the given noise levels.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The noisy images.
            sigma (torch.Tensor): Noise level tensor of shape [B].
                The standard deviation of noise for each sample.
        
        Returns:
            torch.Tensor: Predicted score of shape [B, C, H, W].
                The score function evaluated at the input.
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
    """Score-based diffusion model implementation.
    
    This class implements the score-based diffusion model training and sampling
    procedures. It uses a score network to predict the score function at
    different noise levels and implements annealed Langevin dynamics sampling.
    
    Args:
        config (Dict): Model configuration containing:
            - sigma_min (float): Minimum noise level
            - sigma_max (float): Maximum noise level
            - num_scales (int): Number of noise scales
            - beta (float): Temperature parameter for sampling
            - in_channels (int): Number of input channels
            - model_channels (int): Base channel count for score network
            - loss_type (str): Type of loss function to use
            - loss_config (Dict): Configuration for loss function
    
    Attributes:
        sigma_min (float): Minimum noise level
        sigma_max (float): Maximum noise level
        num_scales (int): Number of noise scales
        beta (float): Temperature parameter
        model (ScoreNet): Score network
        loss_fn (ScoreMatchingLoss or DiffusionLoss): Loss function
    """
    
    def __init__(self, config: Dict):
        """Initialize Score-based diffusion model.
        
        Sets up the score network and loss function according to the configuration.
        Supports both score matching and alternative loss functions.
        """
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
        """Compute score function.
        
        Predicts the score function (gradient of log density) for the input
        at the given noise levels.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The noisy images.
            sigma (torch.Tensor): Noise level tensor of shape [B].
                The standard deviation of noise for each sample.
        
        Returns:
            torch.Tensor: Predicted score of shape [B, C, H, W].
        """
        return self.model(x, sigma)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the score matching loss.
        
        Computes the denoising score matching loss by adding noise at
        random scales and predicting the score function.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W].
                The clean images to train on.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = x.shape[0]
        
        # Sample random noise levels
        u = torch.rand(batch_size, device=x.device)
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** u
        
        # Add noise
        noise = torch.randn_like(x)
        noisy_x = x + sigma[:, None, None, None] * noise
        
        # Predict score
        score = self.forward(noisy_x, sigma)
        
        # Compute loss
        return self.loss_fn(score, noise, sigma)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples using annealed Langevin dynamics.
        
        Implements the annealed Langevin dynamics sampling procedure,
        which gradually denoises random noise using the predicted score
        function at decreasing noise levels.
        
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
        
        # Create noise schedule
        sigmas = torch.exp(torch.linspace(
            np.log(self.sigma_max), np.log(self.sigma_min),
            self.num_scales, device=device
        ))
        
        # Annealed Langevin dynamics
        for sigma in sigmas:
            sigma_batch = torch.full((batch_size,), sigma, device=device)
            step_size = (sigma * self.beta) ** 2 * 2
            
            for _ in range(self.config.get('langevin_steps', 10)):
                # Predict score
                score = self.forward(x, sigma_batch)
                
                # Langevin update
                noise = torch.randn_like(x)
                x = x + step_size * score + torch.sqrt(step_size * 2) * noise
        
        return x
    
    def _get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert discrete timesteps to continuous noise levels."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** (t.float() / self.num_scales) 