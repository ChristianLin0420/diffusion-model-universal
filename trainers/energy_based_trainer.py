import torch
from .ddpm_trainer import DDPMTrainer
from typing import Dict, Optional

class EnergyBasedTrainer(DDPMTrainer):
    """Trainer class for Energy-based diffusion models."""
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """
        Generate and save samples using annealed Langevin dynamics.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
        """
        self.model.eval()
        with torch.no_grad():
            # Use energy-based sampling with annealed Langevin dynamics
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples using parent class method
            self._save_samples(samples, epoch)
    
    def _log_additional_metrics(self, loss: torch.Tensor, epoch: int):
        """Log additional metrics specific to energy-based models."""
        super()._log_additional_metrics(loss, epoch)
        
        if self.config.get('use_wandb', False):
            import wandb
            # Log energy model parameters
            wandb.log({
                'energy_scale': self.model.loss_fn.energy_scale,
                'regularization_weight': self.model.loss_fn.regularization_weight,
                'langevin_step_size': self.model.langevin_step_size
            }, step=epoch) 