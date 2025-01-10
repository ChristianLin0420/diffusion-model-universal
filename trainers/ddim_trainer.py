import torch
from .ddpm_trainer import DDPMTrainer
from typing import Dict, Optional

class DDIMTrainer(DDPMTrainer):
    """Trainer class for DDIM models with specialized sampling."""
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """
        Generate and save samples using DDIM sampling.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
        """
        self.model.eval()
        with torch.no_grad():
            # Use DDIM-specific sampling
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples using parent class method
            self._save_samples(samples, epoch) 