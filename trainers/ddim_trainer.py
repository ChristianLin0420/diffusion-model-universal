"""DDIM Model Trainer Implementation.

This module provides a trainer class for Denoising Diffusion Implicit Models
(DDIM). It extends the DDPM trainer with DDIM-specific sampling functionality
while maintaining the same training process.

Key Features:
    - DDIM-specific sample generation
    - Inherits training functionality from DDPM trainer
    - Configurable sampling parameters
    - Faster sampling than DDPM
    - Compatible with DDPM checkpoints
    - Train/val/test split support
"""

import torch
from .ddpm_trainer import DDPMTrainer
from typing import Dict, Optional

class DDIMTrainer(DDPMTrainer):
    """Trainer class for DDIM models with specialized sampling.
    
    This class extends the DDPM trainer to handle DDIM models. While the training
    process remains the same as DDPM (since DDIM uses the same training objective),
    the sampling process is modified to use DDIM's deterministic sampling procedure.
    
    Args:
        model (nn.Module): The DDIM model to train.
            Must implement loss_function() and sample() methods.
        train_loader (DataLoader): Training data loader.
            Should yield batches of images.
        val_loader (DataLoader): Validation data loader.
            Used for validation during training.
        test_loader (DataLoader): Test data loader.
            Used for final evaluation.
        config (Dict): Training configuration containing:
            - learning_rate (float): Learning rate for optimizer
            - beta1 (float): Adam beta1 parameter
            - beta2 (float): Adam beta2 parameter
            - output_dir (str): Directory for saving outputs
            - use_wandb (bool): Whether to use Weights & Biases
            - wandb_project (str): W&B project name
            - sample_interval (int): Epochs between sample generation
            - checkpoint_interval (int): Epochs between checkpoints
            - val_interval (int): Steps between validation
            - ddim_sampling_eta (float): DDIM sampling parameter
            - ddim_steps (int): Number of DDIM sampling steps
        device (torch.device): Device to train on (CPU/GPU).
    
    Attributes:
        Inherits all attributes from DDPMTrainer.
    """
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """Generate and save samples using DDIM sampling.
        
        Uses DDIM's deterministic sampling process to generate samples,
        which is typically faster than DDPM sampling due to requiring
        fewer steps.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the output file.
            num_samples (int, optional): Number of samples to generate.
                Should be a perfect square. Defaults to 16.
        """
        self.model.eval()
        with torch.no_grad():
            # Use DDIM-specific sampling
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples using parent class method
            self._save_samples(samples, epoch) 