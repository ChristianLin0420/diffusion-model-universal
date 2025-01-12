"""Attention layers for diffusion models.

This module contains attention-related layers that can be used across different
diffusion model architectures.
"""

import torch
import torch.nn as nn

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