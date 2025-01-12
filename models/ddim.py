"""Denoising Diffusion Implicit Model Implementation.

This module implements DDIM as described in 'Denoising Diffusion Implicit Models'
by Song et al. It extends the DDPM framework with deterministic sampling and
fewer sampling steps.

Key Features:
    - Implicit sampling process
    - Deterministic generation
    - Accelerated sampling
    - Compatible with DDPM training
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from .ddpm import DDPM, UNet
from utils.losses import DiffusionLoss

class DDIM(DDPM):
    """Denoising Diffusion Implicit Model implementation.
    
    Extends DDPM with deterministic sampling and fewer sampling steps.
    Uses the same training process as DDPM but implements a different
    sampling procedure.
    
    Args:
        config (Dict): Model configuration containing:
            - All DDPM configuration options
            - ddim_sampling_steps (int): Number of steps for DDIM sampling
            - ddim_discretize_method (str): Method to select timesteps
            - eta (float): Coefficient for stochastic noise
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # DDIM specific parameters
        self.ddim_sampling_steps = config.get('ddim_sampling_steps', 50)
        self.ddim_discretize = config.get('ddim_discretize_method', 'uniform')
        self.eta = config.get('eta', 0.0)  # 0 for deterministic sampling
        
        # Calculate timesteps for DDIM sampling
        self.ddim_timesteps = self._get_ddim_timesteps()
        
        # Calculate alpha, sigma values for DDIM sampling
        self._precompute_ddim_sampling_parameters()
    
    def _get_ddim_timesteps(self) -> torch.Tensor:
        """Get timesteps for DDIM sampling.
        
        Returns:
            torch.Tensor: Selected timesteps for DDIM sampling.
        """
        if self.ddim_discretize == 'uniform':
            c = self.num_timesteps // self.ddim_sampling_steps
            ddim_timesteps = torch.arange(0, self.num_timesteps, c)
        elif self.ddim_discretize == 'quad':
            ddim_timesteps = torch.linspace(0, torch.sqrt(torch.tensor(self.num_timesteps * .8)),
                                          self.ddim_sampling_steps) ** 2
            ddim_timesteps = ddim_timesteps.long()
        else:
            raise NotImplementedError(f"Unknown discretization method: {self.ddim_discretize}")
        
        return ddim_timesteps
    
    def _precompute_ddim_sampling_parameters(self):
        """Precompute alpha, sigma values for DDIM sampling."""
        # Register buffers for alpha, sigma values
        alphas = self.alphas_cumprod[self.ddim_timesteps]
        alphas_prev = torch.cat([self.alphas_cumprod[0:1], self.alphas_cumprod[self.ddim_timesteps[:-1]]])
        
        # Calculate sigma according to DDIM paper
        sigmas = self.eta * torch.sqrt(
            (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
        )
        
        self.register_buffer('ddim_alphas', alphas)
        self.register_buffer('ddim_alphas_prev', alphas_prev)
        self.register_buffer('ddim_sigmas', sigmas)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - alphas))
    
    def _ddim_sample(self, x: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                    pred_noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform one step of DDIM sampling.
        
        Args:
            x (torch.Tensor): Current noisy samples
            t (torch.Tensor): Current timestep
            t_prev (torch.Tensor): Previous timestep
            pred_noise (Optional[torch.Tensor]): Predicted noise if pre-computed
        
        Returns:
            torch.Tensor: Samples at timestep t_prev
        """
        # Get alphas for current and previous timestep
        alpha_t = self.ddim_alphas[t]
        alpha_t_prev = self.ddim_alphas_prev[t]
        sigma_t = self.ddim_sigmas[t]
        sqrt_one_minus_alpha_t = self.ddim_sqrt_one_minus_alphas[t]
        
        # Predict noise if not provided
        if pred_noise is None:
            pred_noise = self.forward(x, t)
        
        # Extract predicted x0
        pred_x0 = (x - sqrt_one_minus_alpha_t[:, None, None, None] * pred_noise) / \
                 torch.sqrt(alpha_t)[:, None, None, None]
        
        # Clip predicted x0 to prevent extreme values
        pred_x0 = pred_x0.clamp(-1, 1)
        
        # Direction pointing to x_t
        dir_xt = torch.sqrt(1. - alpha_t_prev - sigma_t**2)[:, None, None, None] * pred_noise
        
        # Random noise for stochastic sampling
        if self.eta > 0:
            noise = torch.randn_like(x)
            noise = noise.clamp(-3, 3)  # Clip noise for stability
        else:
            noise = 0
        
        x_prev = torch.sqrt(alpha_t_prev)[:, None, None, None] * pred_x0 + \
                dir_xt + sigma_t[:, None, None, None] * noise
        
        return x_prev
    
    def generate_samples(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate samples using DDIM sampling.
        
        Args:
            batch_size (int): Number of samples to generate
            device (torch.device): Device to generate samples on
        
        Returns:
            torch.Tensor: Generated samples
        """
        shape = (batch_size, self.config['image_channels'], 
                self.config['image_size'], self.config['image_size'])
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        # Iteratively denoise
        for i in range(len(self.ddim_timesteps) - 1, -1, -1):
            t = torch.full((batch_size,), self.ddim_timesteps[i], device=device)
            t_prev = torch.full((batch_size,), 
                              self.ddim_timesteps[i-1] if i > 0 else 0,
                              device=device)
            
            x = self._ddim_sample(x, t, t_prev)
        
        return x
    
    def generate_samples_with_intermediates(
        self, batch_size: int, device: torch.device, save_interval: int = 2
    ) -> List[torch.Tensor]:
        """Generate samples with intermediate steps using DDIM sampling.
        
        Args:
            batch_size (int): Number of samples to generate
            device (torch.device): Device to generate samples on
            save_interval (int): Interval at which to save intermediate samples
        
        Returns:
            List[torch.Tensor]: List of intermediate samples
        """
        shape = (batch_size, self.config['image_channels'], 
                self.config['image_size'], self.config['image_size'])
        x = torch.randn(shape, device=device)
        
        # Store intermediate samples
        intermediate_samples = [x.clone()]
        
        # Iteratively denoise
        for i in range(len(self.ddim_timesteps) - 1, -1, -1):
            t = torch.full((batch_size,), self.ddim_timesteps[i], device=device)
            t_prev = torch.full((batch_size,), 
                              self.ddim_timesteps[i-1] if i > 0 else 0,
                              device=device)
            
            x = self._ddim_sample(x, t, t_prev)
            
            # Save intermediate sample
            if i % save_interval == 0 or i == 0:
                intermediate_samples.append(x.clone())
        
        return intermediate_samples 