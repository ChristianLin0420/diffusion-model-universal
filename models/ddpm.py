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
import math

class TransformerPositionalEmbedding(nn.Module):
    """Transformer-style positional embedding for time steps.
    
    As described in the DDPM paper, we use the transformer's sinusoidal position
    embedding to encode diffusion time steps.
    
    Args:
        dimension (int): Dimension of the embedding.
    """
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dimension // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ResidualBlock(nn.Module):
    """Residual block with time embedding.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_channels (int): Number of time embedding channels
        num_groups (int): Number of groups for group normalization
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_channels: int, num_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection handling
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding injection."""
        h = self.norm1(x)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[..., None, None]
        
        h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)

class ConvDownBlock(nn.Module):
    """Convolutional block with downsampling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_channels (int): Number of time embedding channels
        num_layers (int): Number of residual blocks
        num_groups (int): Number of groups for group normalization
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_channels: int, num_layers: int = 2, num_groups: int = 32):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_channels,
                num_groups
            )
            for i in range(num_layers)
        ])
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding injection."""
        h = x
        for block in self.res_blocks:
            h = block(h, time_emb)
        return self.downsample(h)

class ConvUpBlock(nn.Module):
    """Convolutional block with upsampling.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_channels (int): Number of time embedding channels
        num_layers (int): Number of residual blocks
        num_groups (int): Number of groups for group normalization
    """
    def __init__(self, in_channels: int, out_channels: int, time_emb_channels: int, num_layers: int = 2, num_groups: int = 32):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_channels,
                num_groups
            )
            for i in range(num_layers)
        ])
        self.upsample = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding injection."""
        h = x
        for block in self.res_blocks:
            h = block(h, time_emb)
        return self.upsample(h)

class SelfAttentionBlock(nn.Module):
    """Self-attention block as described in the reference implementation."""
    def __init__(self, in_channels: int, embedding_dim: int, num_heads: int = 4, num_groups: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.d_model = embedding_dim
        self.d_keys = embedding_dim // num_heads
        
        # Linear projections for Q, K, V
        self.query_projection = nn.Linear(in_channels, embedding_dim)
        self.key_projection = nn.Linear(in_channels, embedding_dim)
        self.value_projection = nn.Linear(in_channels, embedding_dim)
        self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        
        # Layer norm
        self.norm = nn.GroupNorm(num_groups, embedding_dim)
    
    def split_features_for_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split features between attention heads."""
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.num_heads, self.d_keys)
        x = x.transpose(1, 2)  # [batch, num_heads, seq_len, d_keys]
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with multi-head self-attention."""
        input_tensor = x
        batch, c, h, w = x.shape
        
        # Reshape input: [batch, channels, height, width] -> [batch, height*width, channels]
        x = x.view(batch, c, h * w).transpose(1, 2)
        
        # Get Q, K, V projections
        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)
        
        # Split between heads
        queries = self.split_features_for_heads(queries)
        keys = self.split_features_for_heads(keys)
        values = self.split_features_for_heads(values)
        
        # Scaled dot-product attention
        scale = self.d_keys ** -0.5
        attention_scores = torch.softmax(torch.matmul(queries, keys.transpose(-1, -2)) * scale, dim=-1)
        attention_scores = torch.matmul(attention_scores, values)
        
        # Reshape attention scores
        attention_scores = attention_scores.permute(0, 2, 1, 3).contiguous()
        attention_scores = attention_scores.view(batch, h * w, self.d_model)
        
        # Final projection and reshape
        linear_projection = self.final_projection(attention_scores)
        linear_projection = linear_projection.transpose(-1, -2).reshape(batch, self.d_model, h, w)
        
        # Residual connection + norm
        x = self.norm(linear_projection + input_tensor)
        return x

class AttentionDownBlock(nn.Module):
    """Attention-enhanced downsampling block.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_channels (int): Number of time embedding channels
        num_layers (int): Number of residual blocks
        num_groups (int): Number of groups for group normalization
        num_att_heads (int): Number of attention heads
        downsample (bool): Whether to downsample the features
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_channels: int,
        num_layers: int = 2,
        num_groups: int = 32,
        num_att_heads: int = 4,
        downsample: bool = True
    ):
        super().__init__()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_channels,
                num_groups
            )
            for i in range(num_layers)
        ])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(
                in_channels=out_channels,
                embedding_dim=out_channels,
                num_heads=num_att_heads,
                num_groups=num_groups
            )
            for _ in range(num_layers)
        ])
        
        # Optional downsampling
        self.downsample = (
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if downsample
            else None
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual blocks and self-attention."""
        h = x
        for res_block, att_block in zip(self.res_blocks, self.attention_blocks):
            h = res_block(h, time_emb)
            h = att_block(h)
        
        if self.downsample:
            h = self.downsample(h)
        return h

class AttentionUpBlock(nn.Module):
    """Attention-enhanced upsampling block.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        time_emb_channels (int): Number of time embedding channels
        num_layers (int): Number of residual blocks
        num_groups (int): Number of groups for group normalization
        num_att_heads (int): Number of attention heads
        upsample (bool): Whether to upsample the features
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_channels: int,
        num_layers: int = 2,
        num_groups: int = 32,
        num_att_heads: int = 4,
        upsample: bool = True
    ):
        super().__init__()
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_channels,
                num_groups
            )
            for i in range(num_layers)
        ])
        
        # Attention blocks
        self.attention_blocks = nn.ModuleList([
            SelfAttentionBlock(
                in_channels=out_channels,
                embedding_dim=out_channels,
                num_heads=num_att_heads,
                num_groups=num_groups
            )
            for _ in range(num_layers)
        ])
        
        # Optional upsampling
        self.upsample = (
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
            if upsample
            else None
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual blocks and self-attention."""
        h = x
        for res_block, att_block in zip(self.res_blocks, self.attention_blocks):
            h = res_block(h, time_emb)
            h = att_block(h)
        
        if self.upsample:
            h = self.upsample(h)
        return h

class UNet(nn.Module):
    """U-Net architecture for noise prediction.
    
    This class implements a U-Net model with time conditioning for noise prediction
    in the DDPM framework. It includes skip connections, residual blocks, and
    time embeddings as described in the DDPM paper.
    
    Args:
        in_channels (int): Number of input channels.
        model_channels (int): Base channel count for the model.
        out_channels (int): Number of output channels.
    """
    
    def __init__(self, in_channels: int, model_channels: int, out_channels: int):
        super().__init__()
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding='same')
        
        # Time embedding
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(model_channels),
            nn.Linear(model_channels, model_channels * 4),
            nn.GELU(),
            nn.Linear(model_channels * 4, model_channels * 4)
        )
        
        # Downsampling path with attention
        self.down_blocks = nn.ModuleList([
            ConvDownBlock(model_channels, model_channels, model_channels * 4),
            ConvDownBlock(model_channels, model_channels, model_channels * 4),
            ConvDownBlock(model_channels, model_channels * 2, model_channels * 4),
            AttentionDownBlock(
                model_channels * 2, model_channels * 2, model_channels * 4,
                num_att_heads=4
            ),
            ConvDownBlock(model_channels * 2, model_channels * 4, model_channels * 4)
        ])
        
        # Bottleneck with attention
        self.bottleneck = nn.Sequential(
            ResidualBlock(model_channels * 4, model_channels * 4, model_channels * 4),
            SelfAttentionBlock(model_channels * 4, model_channels * 4, num_heads=4),
            ResidualBlock(model_channels * 4, model_channels * 4, model_channels * 4)
        )
        
        # Upsampling path with attention
        self.up_blocks = nn.ModuleList([
            ConvUpBlock(model_channels * 8, model_channels * 4, model_channels * 4),
            AttentionUpBlock(
                model_channels * 6, model_channels * 2, model_channels * 4,
                num_att_heads=4, upsample=True
            ),
            ConvUpBlock(model_channels * 4, model_channels * 2, model_channels * 4),
            ConvUpBlock(model_channels * 3, model_channels, model_channels * 4),
            ConvUpBlock(model_channels * 2, model_channels, model_channels * 4)
        ])
        
        # Output convolution
        self.output_conv = nn.Sequential(
            nn.GroupNorm(32, model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass through U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            t (torch.Tensor): Time embedding tensor of shape [B].
        
        Returns:
            torch.Tensor: Output tensor of shape [B, C, H, W].
        """
        # Time embedding
        t_emb = self.positional_encoding(t)
        
        # Initial features
        h = self.initial_conv(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Downsampling path
        for block in self.down_blocks:
            h = block(h, t_emb)
            skip_connections.append(h)
        
        # Bottleneck
        if isinstance(self.bottleneck, nn.Sequential):
            h = self.bottleneck[0](h, t_emb)  # First ResBlock
            h = self.bottleneck[1](h)         # Attention
            h = self.bottleneck[2](h, t_emb)  # Second ResBlock
        else:
            h = self.bottleneck(h, t_emb)
        
        # Upsampling path with skip connections
        skip_connections = list(reversed(skip_connections))

        for block, skip in zip(self.up_blocks, skip_connections):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
        
        # Output
        return self.output_conv(h)

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
        self.beta_start = config.get('beta_start', 1e-4)
        self.beta_end = config.get('beta_end', 1e-2)
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
            loss_type=config.get('model_config', {}).get('loss_type', 'mse'),
            loss_config=config.get('model_config', {}).get('loss_config', {})
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
    
    def sample(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Generate samples through the reverse diffusion process.
        
        Args:
            x (torch.Tensor): Input noise tensor of shape [B, C, H, W].
            t (torch.Tensor): Current timestep tensor of shape [B].
        
        Returns:
            torch.Tensor: Denoised samples of shape [B, C, H, W].
        """
        return self._reverse_diffusion_step(x, t)
    
    def generate_samples(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples through the reverse diffusion process.
        
        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
        
        Returns:
            torch.Tensor: Generated samples of shape [B, C, H, W].
        """
        shape = (batch_size, self.config['image_channels'], 
                self.config['image_size'], self.config['image_size'])
        x = torch.randn(shape, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self._reverse_diffusion_step(x, t_batch)
            
        return x
    
    def generate_samples_with_intermediates(self, batch_size: int, device: torch.device, save_interval: int = 100) -> torch.Tensor:
        """Generate samples through the reverse diffusion process and return intermediate states.
        
        Args:
            batch_size (int): Number of samples to generate.
            device (torch.device): Device to generate samples on.
            save_interval (int): Interval at which to save intermediate states.
        
        Returns:
            List[torch.Tensor]: List of tensors containing the initial noise and intermediate states.
        """
        shape = (batch_size, self.config['image_channels'], 
                self.config['image_size'], self.config['image_size'])
        x = torch.randn(shape, device=device)
        
        # Store intermediate samples
        intermediate_samples = [x.clone()]  # Start with initial noise
        
        # Generate samples with intermediate steps
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self._reverse_diffusion_step(x, t_batch)
            
            # Save intermediate result if it's a save timestep
            if t % save_interval == 0 or t == 0:  # Also save the final result
                intermediate_samples.append(x.clone())
        
        return intermediate_samples
    
    def _add_noise(self, x: torch.Tensor, t: torch.Tensor, 
                  noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to the input according to the noise schedule.
        
        Implementation of forward diffusion process from the paper.
        """
        if noise is None:
            noise = torch.randn_like(x)
            
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
    
    def _reverse_diffusion_step(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Perform one step of the reverse diffusion process.
        
        Implementation of Algorithm 2 from the DDPM paper.
        """
        # Get noise prediction from model
        noise_pred = self.forward(x, t)
        
        # Get alpha values for current timestep
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        
        # Get alpha_cumprod for previous timestep
        alpha_cumprod_prev = self.alphas_cumprod[t-1][:, None, None, None] if t[0] > 0 else torch.ones_like(alpha_cumprod_t)
        
        # Calculate variance (beta_tilde in the paper)
        beta_t = self.betas[t][:, None, None, None]
        beta_tilde = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * beta_t
        
        # Calculate mean
        mean = torch.pow(alpha_t, -0.5) * (
            x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * noise_pred
        )
        
        # Add variance if not the last step
        if t[0] > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_tilde) * noise
        else:
            x = mean
            
        return x 