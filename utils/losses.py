import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable

class DiffusionLoss:
    """Flexible loss module for diffusion models."""
    
    LOSS_TYPES = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'hybrid': None  # Will be handled separately
    }
    
    def __init__(self, loss_type: str = 'mse', loss_config: Optional[Dict] = None):
        """
        Initialize loss function.
        
        Args:
            loss_type: Type of loss ('mse', 'l1', 'huber', 'hybrid')
            loss_config: Additional configuration for loss function
        """
        self.loss_type = loss_type.lower()
        self.loss_config = loss_config or {}
        
        if self.loss_type not in self.LOSS_TYPES:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Special handling for hybrid loss
        if self.loss_type == 'hybrid':
            self.loss_weights = self.loss_config.get('weights', {'mse': 0.5, 'l1': 0.5})
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, 
                 timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate loss between prediction and target.
        
        Args:
            pred: Predicted values
            target: Target values
            timesteps: Optional timestep information for time-dependent weighting
            
        Returns:
            Calculated loss value
        """
        if self.loss_type == 'hybrid':
            return self._compute_hybrid_loss(pred, target, timesteps)
        
        loss_fn = self.LOSS_TYPES[self.loss_type]
        reduction = self.loss_config.get('reduction', 'mean')
        
        # Apply time-dependent weighting if provided
        if timesteps is not None and 'time_weights' in self.loss_config:
            weights = self._get_time_weights(timesteps)
            loss = loss_fn(pred, target, reduction='none')
            return (loss * weights).mean()
        
        return loss_fn(pred, target, reduction=reduction)
    
    def _compute_hybrid_loss(self, pred: torch.Tensor, target: torch.Tensor,
                           timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute hybrid loss combining multiple loss functions."""
        total_loss = 0
        for loss_type, weight in self.loss_weights.items():
            if loss_type not in self.LOSS_TYPES or loss_type == 'hybrid':
                continue
            loss = self.LOSS_TYPES[loss_type](pred, target, reduction='mean')
            total_loss += weight * loss
        return total_loss
    
    def _get_time_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get time-dependent weights for loss calculation."""
        time_weight_type = self.loss_config['time_weights'].get('type', 'linear')
        max_timesteps = self.loss_config['time_weights'].get('max_timesteps', 1000)
        
        if time_weight_type == 'linear':
            weights = 1 - (timesteps.float() / max_timesteps)
        elif time_weight_type == 'exponential':
            beta = self.loss_config['time_weights'].get('beta', 0.1)
            weights = torch.exp(-beta * timesteps.float())
        else:
            weights = torch.ones_like(timesteps, dtype=torch.float)
        
        return weights.view(-1, 1, 1, 1)

class ScoreMatchingLoss(nn.Module):
    """Score matching loss for score-based diffusion models."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def forward(self, score: torch.Tensor, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Compute denoising score matching loss.
        
        Args:
            score: Predicted score
            x: Input data
            sigma: Noise levels
            
        Returns:
            Loss value
        """
        noise = torch.randn_like(x)
        perturbed_x = x + sigma.view(-1, 1, 1, 1) * noise
        target = -noise / sigma.view(-1, 1, 1, 1)
        
        return F.mse_loss(score, target)

class EnergyBasedLoss(nn.Module):
    """Energy-based loss for energy-based diffusion models."""
    
    def __init__(self, energy_scale: float = 1.0, regularization_weight: float = 0.1):
        super().__init__()
        self.energy_scale = energy_scale
        self.regularization_weight = regularization_weight
    
    def forward(self, energy: torch.Tensor, x_real: torch.Tensor, x_fake: torch.Tensor) -> torch.Tensor:
        """
        Compute energy-based loss with regularization.
        
        Args:
            energy: Energy function output
            x_real: Real data samples
            x_fake: Generated samples
            
        Returns:
            Loss value
        """
        # Energy difference between real and fake samples
        energy_real = energy(x_real)
        energy_fake = energy(x_fake)
        
        # Contrastive divergence loss
        cd_loss = torch.mean(energy_real) - torch.mean(energy_fake)
        
        # Gradient penalty regularization
        alpha = torch.rand(x_real.size(0), 1, 1, 1, device=x_real.device)
        interpolated = alpha * x_real + (1 - alpha) * x_fake
        interpolated.requires_grad_(True)
        
        energy_interpolated = energy(interpolated)
        gradients = torch.autograd.grad(
            outputs=energy_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(energy_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return cd_loss + self.regularization_weight * gradient_penalty 