import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from .base_model import BaseDiffusion
from .ddpm import UNet
from utils.losses import DiffusionLoss

class DDIM(BaseDiffusion):
    """Denoising Diffusion Implicit Model implementation."""
    
    def __init__(self, config: Dict):
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
        """Forward pass predicting noise."""
        return self.model(x, t)
    
    def loss_function(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the diffusion loss."""
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
        """Generate samples using DDIM sampling."""
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
        """Get timesteps for DDIM sampling."""
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
        """Perform one DDIM sampling step."""
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
    
    def _add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Add noise to the input according to the noise schedule."""
        if noise is None:
            noise = torch.randn_like(x)
            
        alphas_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        return torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise 