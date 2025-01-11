import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
from torchvision.models import vgg16
from torchvision.transforms import Normalize

class DiffusionLoss:
    """Flexible loss module for diffusion models."""
    
    LOSS_TYPES = {
        'mse': F.mse_loss,
        'l1': F.l1_loss,
        'huber': F.smooth_l1_loss,
        'hybrid': None  # Will be handled separately
    }
    
    def __init__(self, loss_type: str = 'mse', loss_config: Optional[Dict] = None):
        """Initialize loss function with comprehensive configuration.
        
        Args:
            loss_type: Type of loss ('mse', 'l1', 'huber', 'hybrid')
            loss_config: Additional configuration for loss function including:
                - mse_weight: Weight for MSE loss
                - l1_weight: Weight for L1 loss
                - huber_weight: Weight for Huber loss
                - huber_delta: Delta parameter for Huber loss
                - use_hybrid: Whether to use hybrid loss
                - hybrid_weights: Weights for different loss components
                - use_time_weighting: Whether to use time-dependent weighting
                - time_weight_type: Type of time weighting (snr, linear, inverse)
                - time_weight_params: Parameters for time weighting
                - perceptual_weight: Weight for perceptual loss
                - adversarial_weight: Weight for adversarial loss
        """
        self.loss_type = loss_type.lower()
        self.loss_config = loss_config or {}
        
        if self.loss_type not in self.LOSS_TYPES:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        # Initialize loss weights
        self.mse_weight = self.loss_config.get('mse_weight', 1.0)
        self.l1_weight = self.loss_config.get('l1_weight', 0.0)
        self.huber_weight = self.loss_config.get('huber_weight', 0.0)
        self.huber_delta = self.loss_config.get('huber_delta', 1.0)
        
        # Setup hybrid loss if enabled
        self.use_hybrid = self.loss_config.get('use_hybrid', False)
        if self.use_hybrid:
            weights = self.loss_config.get('hybrid_weights', {})
            self.hybrid_weights = {
                'mse': weights.get('mse', 1.0),
                'l1': weights.get('l1', 0.0),
                'huber': weights.get('huber', 0.0)
            }
        
        # Setup time-dependent weighting
        self.use_time_weighting = self.loss_config.get('use_time_weighting', True)
        self.time_weight_type = self.loss_config.get('time_weight_type', 'snr')
        self.time_weight_params = self.loss_config.get('time_weight_params', {
            'min_weight': 0.1,
            'max_weight': 1.0
        })
        
        # Setup additional loss components
        self.perceptual_weight = self.loss_config.get('perceptual_weight', 0.0)
        self.adversarial_weight = self.loss_config.get('adversarial_weight', 0.0)
        
        # Initialize perceptual loss if needed
        if self.perceptual_weight > 0:
            self.perceptual_loss = PerceptualLoss()
    
    def __call__(self, pred: torch.Tensor, target: torch.Tensor, 
                 timesteps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Calculate loss between prediction and target.
        
        Args:
            pred: Predicted values
            target: Target values
            timesteps: Optional timestep information for time-dependent weighting
            
        Returns:
            Calculated loss value
        """
        # Calculate base loss
        if self.use_hybrid:
            base_loss = self._compute_hybrid_loss(pred, target)
        else:
            base_loss = self._compute_single_loss(pred, target)
        
        # Apply time-dependent weighting if enabled
        if self.use_time_weighting and timesteps is not None:
            weights = self._get_time_weights(timesteps)
            base_loss = base_loss * weights
        
        # Add perceptual loss if enabled
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(pred, target)
            base_loss = base_loss + self.perceptual_weight * perceptual_loss
        
        # Take mean of all losses
        return base_loss.mean()
    
    def _compute_single_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss using a single loss function."""
        if self.loss_type == 'mse':
            return self.mse_weight * F.mse_loss(pred, target, reduction='none')
        elif self.loss_type == 'l1':
            return self.l1_weight * F.l1_loss(pred, target, reduction='none')
        elif self.loss_type == 'huber':
            return self.huber_weight * F.smooth_l1_loss(pred, target, reduction='none', beta=self.huber_delta)
        else:
            raise ValueError(f"Unsupported single loss type: {self.loss_type}")
    
    def _compute_hybrid_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute hybrid loss combining multiple loss functions."""
        total_loss = torch.zeros_like(pred)
        
        if self.hybrid_weights['mse'] > 0:
            total_loss += self.hybrid_weights['mse'] * F.mse_loss(pred, target, reduction='none')
        
        if self.hybrid_weights['l1'] > 0:
            total_loss += self.hybrid_weights['l1'] * F.l1_loss(pred, target, reduction='none')
        
        if self.hybrid_weights['huber'] > 0:
            total_loss += self.hybrid_weights['huber'] * F.smooth_l1_loss(
                pred, target, reduction='none', beta=self.huber_delta
            )
        
        return total_loss
    
    def _get_time_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get time-dependent weights for loss calculation.
        
        Args:
            timesteps: Current timesteps in the diffusion process [B]
        Returns:
            weights: Time-dependent weights [B, 1, 1, 1]
        """
        min_weight = self.time_weight_params['min_weight']
        max_weight = self.time_weight_params['max_weight']
        
        if self.time_weight_type == 'snr':
            # Proper SNR-based weighting using beta schedule
            # Convert timesteps to float and scale to [0, 1]
            t = timesteps.float() / timesteps.max()
            
            # Use linear beta schedule (you can adjust these values)
            beta_start = 1e-4
            beta_end = 2e-2
            betas = torch.linspace(beta_start, beta_end, timesteps.max().item() + 1, device=timesteps.device)
            
            # Calculate alpha products
            alphas = 1 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod = alphas_cumprod.index_select(0, timesteps)
            
            # Calculate SNR weights
            snr = alphas_cumprod / (1 - alphas_cumprod)
            weights = snr / snr.max()  # Normalize to [0, 1]
            
            # Ensure no division by zero or negative values
            weights = weights.clamp(min=1e-5)
            
        elif self.time_weight_type == 'linear':
            # Linear interpolation
            weights = 1 - (timesteps.float() / timesteps.max())
        elif self.time_weight_type == 'inverse':
            # Inverse time weighting with offset to prevent division by zero
            weights = 1 / (timesteps.float() + 1)
        else:
            weights = torch.ones_like(timesteps, dtype=torch.float)
        
        # Scale weights to [min_weight, max_weight]
        weights = min_weight + (max_weight - min_weight) * (
            (weights - weights.min()) / (weights.max() - weights.min() + 1e-5)
        )
        
        # Add dimensions for broadcasting
        return weights.view(-1, 1, 1, 1)

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""
    
    def __init__(self, layer_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.vgg = vgg16(pretrained=True).features.eval()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Default layer weights if none provided
        self.layer_weights = layer_weights or {
            '3': 1.0,   # relu1_2
            '8': 1.0,   # relu2_2
            '15': 1.0,  # relu3_3
        }
        
        for param in self.vgg.parameters():
            param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss between prediction and target."""
        # Normalize inputs
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Get features and calculate loss
        loss = 0
        for name, module in self.vgg.named_children():
            pred = module(pred)
            target = module(target)
            
            if name in self.layer_weights:
                loss += self.layer_weights[name] * F.mse_loss(pred, target)
        
        return loss

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