"""DDIM Model Trainer Implementation.

This module extends the DDPM trainer for DDIM models. It inherits most functionality
from the DDPM trainer but implements DDIM-specific sampling and logging.
"""

from .ddpm_trainer import DDPMTrainer
import torch
import wandb
from torchvision.utils import make_grid, save_image
import os
from typing import Dict

class DDIMTrainer(DDPMTrainer):
    """Trainer class for DDIM models.
    
    Extends DDPMTrainer with DDIM-specific functionality while maintaining
    compatibility with the base trainer.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Log DDIM-specific parameters
        if self.rank == 0 and self.config.get('logging', {}).get('use_wandb', False):
            wandb.config.update({
                'ddim_sampling_steps': self.config.get('model_config', {}).get('ddim_sampling_steps', 50),
                'ddim_discretize': self.config.get('model_config', {}).get('ddim_discretize_method', 'uniform'),
                'ddim_eta': self.config.get('model_config', {}).get('eta', 0.0)
            }, allow_val_change=True)
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save samples using DDIM sampling.
        
        Overrides the base class method to use DDIM sampling instead of DDPM.
        Shows the progression from noise to final image with fewer steps.
        
        Args:
            epoch (int): Current epoch number
            num_samples (int, optional): Number of samples to generate
        """
        if self.rank != 0:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Use EMA model for sampling if available
            model = self.ema_model if self.ema_model is not None else (
                self.model.module if self.is_distributed else self.model
            )
            
            # Generate samples with intermediate steps
            intermediate_samples = model.generate_samples_with_intermediates(
                batch_size=num_samples,
                device=self.device
            )
            
            # Create grid
            rows = []
            for i in range(num_samples):
                row = [sample[i:i+1] for sample in intermediate_samples]
                rows.append(torch.cat(row, dim=0))
            
            # Combine all rows
            all_samples = torch.cat(rows, dim=0)
            
            # Create and save grid
            grid = make_grid(all_samples, nrow=len(intermediate_samples), padding=2)
            save_path = os.path.join(self.sample_dir, f'ddim_samples_epoch_{epoch}.png')
            save_image(grid, save_path)
            
            # Log samples
            if self.config.get('logging', {}).get('use_wandb', False):
                wandb.log({
                    f'{self.model_name}/ddim_denoising_process': wandb.Image(grid),
                    'epoch': epoch
                })
            
            if self.writer is not None:
                self.writer.add_image(
                    f'{self.model_name}/ddim_denoising_process',
                    grid,
                    global_step=epoch
                ) 