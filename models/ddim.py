"""Denoising Diffusion Implicit Model Implementation.

This module implements the Denoising Diffusion Implicit Model (DDIM) as described
in the paper "Denoising Diffusion Implicit Models" by Song et al. It provides
a deterministic sampling process that achieves high-quality generation with fewer
steps than DDPM.

Key Features:
    - Deterministic sampling process
    - Configurable number of sampling steps
    - Adjustable stochasticity through eta parameter
    - Faster generation than DDPM
    - Compatible with pretrained DDPM models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import DiffusionLoss

class DDIM(BaseDiffusion):
    """Denoising Diffusion Implicit Model implementation.
    
    This class implements the DDIM training and sampling procedures. It uses
    a deterministic sampling process that can generate high-quality samples
    with significantly fewer steps than DDPM.
    
    Args:
        config (Dict): Model configuration containing:
            - beta_start (float): Starting value for noise schedule
            - beta_end (float): Ending value for noise schedule
            - num_timesteps (int): Number of diffusion steps
            - ddim_sampling_eta (float): Controls the stochasticity of sampling
            - ddim_steps (int): Number of steps for DDIM sampling
            - in_channels (int): Number of input channels
            - model_channels (int): Base channel count for U-Net
            - loss_type (str): Type of loss function to use
            - loss_config (Dict): Configuration for loss function
    
    Attributes:
        betas (torch.Tensor): Noise schedule
        alphas (torch.Tensor): 1 - betas
        alphas_cumprod (torch.Tensor): Cumulative product of alphas
        model (UNet): U-Net model for noise prediction
        loss_fn (DiffusionLoss): Loss function
        ddim_sampling_eta (float): Stochasticity parameter (0 = deterministic)
        ddim_steps (int): Number of sampling steps
    """
    
    def __init__(self, config: Dict):
        """Initialize DDIM model.
        
        Sets up the noise schedule, creates the U-Net model, and initializes
        the loss function according to the configuration. Also configures
        DDIM-specific parameters for sampling.
        """
        super().__init__(config)
        
        # Model parameters
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.num_timesteps = config.get('num_timesteps', 1000)
        self.ddim_sampling_eta = config.get('ddim_sampling_eta', 0.0)
        self.ddim_steps = config.get('ddim_steps', 50)
        
        # Create noise schedule
        self.register_buffer('betas', torch.linspace(self.beta_start, self.beta_end, self.num_timesteps))
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # Create U-Net model
        self.model = UNet(
            in_channels=config.get('in_channels', 3),
            model_channels=config.get('model_channels', 64),
            out_channels=config.get('in_channels', 3)
        )
        
        # Setup loss function
        self.loss_fn = DiffusionLoss(
            loss_type=config.get('loss_type', 'mse'),
            loss_config=config.get('loss_config', None)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass predicting noise.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The noisy images.
            t (torch.Tensor): Timestep tensor of shape [B].
                The timesteps for each sample.
        
        Returns:
            torch.Tensor: Predicted noise of shape [B, C, H, W].
        """
        return self.model(x, t)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the diffusion loss.
        
        Computes the loss by adding noise to the input at random timesteps
        and predicting that noise. Uses the same loss computation as DDPM
        since the training process is identical.
        
        Args:
            x (torch.Tensor): Input images of shape [B, C, H, W].
                The clean images to train on.
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        batch_size = x.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x.device)
        
        # Generate noise
        noise = torch.randn_like(x)
        
        # Get noisy image
        noisy_x = self._add_noise(x, t, noise)
        
        # Predict noise
        noise_pred = self.forward(noisy_x, t)
        
        # Use flexible loss function with timestep information
        return self.loss_fn(noise_pred, noise, t)
    
    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples using DDIM sampling.
        
        Implements the deterministic DDIM sampling process, which can generate
        high-quality samples in fewer steps than DDPM. The stochasticity can
        be controlled through the ddim_sampling_eta parameter.
        
        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
        
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W].
        """
        # Initial noise
        x = torch.randn((batch_size, self.config['in_channels'],
                        self.config['image_size'], self.config['image_size']),
                       device=device)
        
        # Get sampling timesteps
        timesteps = self._get_sampling_timesteps(self.ddim_steps)
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Current alphas
            alpha = self.alphas_cumprod[t]
            alpha_next = self.alphas_cumprod[t_next]
            
            # Predict noise
            t_batch = torch.tensor([t] * batch_size, device=device)
            noise_pred = self.forward(x, t_batch)
            
            # DDIM sampling step
            x = self._ddim_step(x, noise_pred, t, t_next, alpha, alpha_next)
        
        return x
    
    def _get_sampling_timesteps(self, num_steps: int) -> List[int]:
        """Get timesteps for DDIM sampling.
        
        Creates a sequence of timesteps for the DDIM sampling process.
        The timesteps are evenly spaced in the original diffusion process.
        
        Args:
            num_steps (int): Number of desired sampling steps.
        
        Returns:
            List[int]: List of timesteps in descending order.
        """
        steps = torch.linspace(0, self.num_timesteps - 1, num_steps + 1).round().long()
        return list(reversed(steps.tolist()))
    
    def _ddim_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_next: int,
        alpha: torch.Tensor,
        alpha_next: torch.Tensor
    ) -> torch.Tensor:
        """Perform one DDIM sampling step.
        
        Implements the deterministic DDIM update step with optional
        stochasticity controlled by ddim_sampling_eta.
        
        Args:
            x (torch.Tensor): Current noisy images
            noise_pred (torch.Tensor): Predicted noise
            t (int): Current timestep
            t_next (int): Next timestep
            alpha (torch.Tensor): Current cumulative alpha
            alpha_next (torch.Tensor): Next cumulative alpha
        
        Returns:
            torch.Tensor: Updated samples after one step
        """
        # Extract predicted x_0
        pred_x0 = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_next - self.ddim_sampling_eta ** 2 * (1 - alpha_next)) * noise_pred
        
        # Random noise for stochasticity
        if self.ddim_sampling_eta > 0:
            noise = torch.randn_like(x)
            noise_contribution = self.ddim_sampling_eta * torch.sqrt(1 - alpha_next) * noise
        else:
            noise_contribution = 0
        
        # Compute x_{t-1}
        x_next = torch.sqrt(alpha_next) * pred_x0 + dir_xt + noise_contribution
        
        return x_next
    
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