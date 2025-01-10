import torch
from .ddpm_trainer import DDPMTrainer
from typing import Dict, Optional

class ScoreBasedTrainer(DDPMTrainer):
    """Trainer class for Score-based diffusion models."""
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """
        Generate and save samples using Langevin dynamics.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
        """
        self.model.eval()
        with torch.no_grad():
            # Use score-based sampling with Langevin dynamics
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples using parent class method
            self._save_samples(samples, epoch)
    
    def _log_additional_metrics(self, loss: torch.Tensor, epoch: int):
        """Log additional metrics specific to score-based models."""
        super()._log_additional_metrics(loss, epoch)
        
        if self.config.get('use_wandb', False):
            import wandb
            # Log noise schedule parameters
            wandb.log({
                'sigma_min': self.model.sigma_min,
                'sigma_max': self.model.sigma_max,
                'beta': self.model.beta
            }, step=epoch) 