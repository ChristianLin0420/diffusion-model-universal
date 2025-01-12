"""Residual blocks and related layers for diffusion models.

This module contains residual blocks and related layers that can be used across
different diffusion model architectures.
"""

import torch
import torch.nn as nn
from .attention import SelfAttentionBlock

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
        # Ensure num_groups is valid for both in_channels and out_channels
        num_groups_in = min(num_groups, in_channels)
        while in_channels % num_groups_in != 0 and num_groups_in > 1:
            num_groups_in -= 1
            
        num_groups_out = min(num_groups, out_channels)
        while out_channels % num_groups_out != 0 and num_groups_out > 1:
            num_groups_out -= 1
            
        self.norm1 = nn.GroupNorm(num_groups_in, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_channels, out_channels)
        
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Skip connection handling
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
            
        # Initialize weights
        nn.init.zeros_(self.time_mlp.weight)
        nn.init.zeros_(self.time_mlp.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
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