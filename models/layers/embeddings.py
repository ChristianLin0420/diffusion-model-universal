"""Embedding layers for diffusion models.

This module contains embedding layers that can be used across different
diffusion model architectures.
"""

import torch
import torch.nn as nn
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
        """Forward pass to compute positional embeddings.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B] containing timesteps.
            
        Returns:
            torch.Tensor: Positional embeddings of shape [B, dimension].
        """
        device = x.device
        half_dim = self.dimension // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):
    """Time embedding module that projects timesteps to a higher dimension.
    
    This module takes timesteps and projects them to a higher dimension using
    sinusoidal embeddings followed by an MLP.
    
    Args:
        base_dim (int): Base dimension for the embeddings.
        output_dim (int): Output dimension after MLP projection.
    """
    def __init__(self, base_dim: int, output_dim: int):
        super().__init__()
        self.positional_encoding = nn.Sequential(
            TransformerPositionalEmbedding(base_dim),
            nn.Linear(base_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )
        
        # Initialize weights
        for layer in self.positional_encoding:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute time embeddings.
        
        Args:
            t (torch.Tensor): Input tensor of shape [B] containing timesteps.
            
        Returns:
            torch.Tensor: Time embeddings of shape [B, output_dim].
        """
        return self.positional_encoding(t) 