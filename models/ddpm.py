"""Denoising Diffusion Probabilistic Model Implementation.

This module implements the Denoising Diffusion Probabilistic Model (DDPM) as described
in the paper "Denoising Diffusion Probabilistic Models" by Ho et al. It provides
both the U-Net backbone architecture and the DDPM training and sampling logic.

Key Features:
    - U-Net architecture with time embedding
    - Linear noise schedule
    - Forward diffusion process
    - Reverse denoising process
    - Configurable timesteps and noise levels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from .base_model import BaseDiffusion
from utils.losses import DiffusionLoss

class UNet(nn.Module):
    """U-Net architecture for noise prediction.
    
    This class implements a U-Net model with time conditioning for noise prediction
    in the DDPM framework. It includes skip connections, residual blocks, and
    time embeddings.
    
    Args:
        in_channels (int): Number of input channels.
            Usually 3 for RGB images.
        model_channels (int): Base channel count for the model.
            The number of channels in intermediate layers will be multiples of this.
        out_channels (int): Number of output channels.
            Usually same as in_channels.
    
    Attributes:
        conv_in (nn.Conv2d): Initial convolution layer
        down1, down2 (nn.Conv2d): Downsampling convolutions
        middle (nn.Sequential): Middle block with residual connections
        up1, up2 (nn.ConvTranspose2d): Upsampling convolutions
        conv_out (nn.Conv2d): Final convolution layer
        time_embed (nn.Sequential): Time embedding network
    """
    
    def __init__(self, in_channels: int, model_channels: int, out_channels: int):
        """Initialize U-Net architecture.
        
        Sets up the U-Net architecture with specified channel dimensions and
        initializes all neural network layers.
        """
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # Downsampling
        self.down1 = nn.Conv2d(model_channels, model_channels*2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(model_channels*2, model_channels*4, kernel_size=4, stride=2, padding=1)
        
        # Middle block with residual connections
        self.middle = nn.Sequential(
            nn.Conv2d(model_channels*4, model_channels*4, kernel_size=3, padding=1),
            nn.GroupNorm(8, model_channels*4),
            nn.SiLU(),
            nn.Conv2d(model_channels*4, model_channels*4, kernel_size=3, padding=1)
        )
        
        # Upsampling with skip connections
        self.up1 = nn.ConvTranspose2d(model_channels*8, model_channels*2, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(model_channels*4, model_channels, kernel_size=4, stride=2, padding=1)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels*4),
            nn.SiLU(),
            nn.Linear(model_channels*4, model_channels*4)
        )
        
        # Output convolution
        self.conv_out = nn.Conv2d(model_channels*2, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Processes the input through the U-Net architecture with time conditioning.
        Includes skip connections between corresponding layers in the encoder
        and decoder paths.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
                The noisy images to denoise.
            t (torch.Tensor): Time embedding tensor of shape [B].
                The timesteps for each sample in the batch.
        
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
                The predicted noise in the input.
        """
        # Time embedding
        t_emb = self.time_embed(t)[:, :, None, None]
        
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

class DDPM(BaseDiffusion):
    """Denoising Diffusion Probabilistic Model implementation.
    
    This class implements the DDPM training and sampling procedures. It uses
    a U-Net model for noise prediction and implements the forward and reverse
    diffusion processes.
    
    Args:
        config (Dict): Model configuration containing:
            - beta_start (float): Starting value for noise schedule
            - beta_end (float): Ending value for noise schedule
            - num_timesteps (int): Number of diffusion steps
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
    """
    
    def __init__(self, config: Dict):
        """Initialize DDPM model.
        
        Sets up the noise schedule, creates the U-Net model, and initializes
        the loss function according to the configuration.
        """
        super().__init__(config)
        
        # Model parameters
        self.beta_start = config.get('beta_start', 0.0001)
        self.beta_end = config.get('beta_end', 0.02)
        self.num_timesteps = config.get('num_timesteps', 1000)
        
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
        and predicting that noise.
        
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
        """Generate samples through the reverse diffusion process.
        
        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
        
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W].
        """
        shape = (batch_size, self.config['in_channels'], 
                self.config['image_size'], self.config['image_size'])
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self._reverse_diffusion_step(x, t_batch)
            
        return x
    
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
    
    def _reverse_diffusion_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform one step of the reverse diffusion process.
        
        Args:
            x (torch.Tensor): Current noisy images
            t (torch.Tensor): Current timesteps
        
        Returns:
            torch.Tensor: Less noisy images
        """
        betas_t = self.betas[t][:, None, None, None]
        alphas_t = self.alphas[t][:, None, None, None]
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        
        # Predict noise
        noise_pred = self.forward(x, t)
        
        # Compute reverse step
        mean = (1 / torch.sqrt(alphas_t)) * (x - (betas_t * noise_pred) / 
                                           torch.sqrt(1 - alphas_cumprod_t))
        
        if t[0] > 0:
            noise = torch.randn_like(x)
            variance = torch.sqrt(betas_t) * noise
            x = mean + variance
        else:
            x = mean
            
        return x 