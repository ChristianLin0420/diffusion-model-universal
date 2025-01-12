"""Energy-based Diffusion Model Trainer Implementation.

This module provides a trainer class for Energy-based Diffusion Models. It extends
the DDPM trainer with energy-based specific sampling functionality and additional
metric logging.

Key Features:
    - Annealed Langevin dynamics sampling
    - Energy-specific metric logging
    - Configurable energy scale
    - Regularization parameter tracking
    - Langevin dynamics parameter monitoring
    - Train/val/test split support
    - Multi-GPU training support
"""

import torch
from .ddpm_trainer import DDPMTrainer
from typing import Dict, Optional

class EnergyBasedTrainer(DDPMTrainer):
    """Trainer class for Energy-based diffusion models.
    
    This class extends the DDPM trainer to handle Energy-based models. While the
    basic training loop remains similar, it uses annealed Langevin dynamics for
    sampling and tracks additional metrics specific to energy-based models.
    
    Args:
        model (nn.Module): The Energy-based model to train.
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
            - energy_scale (float): Scale factor for energy values
            - regularization_weight (float): Weight for regularization
            - langevin_steps (int): Number of Langevin steps
            - langevin_step_size (float): Step size for Langevin dynamics
        device (torch.device): Device to train on (CPU/GPU).
        rank (int, optional): Process rank for distributed training.
            Defaults to 0 (single GPU training).
        world_size (int, optional): Total number of processes.
            Defaults to 1 (single GPU training).
    
    Attributes:
        Inherits all attributes from DDPMTrainer, plus:
        model.loss_fn.energy_scale (float): Scale factor for energy values
        model.loss_fn.regularization_weight (float): Regularization weight
        model.langevin_step_size (float): Langevin dynamics step size
    """
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """Generate and save samples using annealed Langevin dynamics.
        
        Uses annealed Langevin dynamics to generate samples, which involves
        running multiple steps of Langevin MCMC with an energy-based model
        at decreasing noise levels. Only runs on rank 0 in distributed mode.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the output file.
            num_samples (int, optional): Number of samples to generate.
                Should be a perfect square. Defaults to 16.
        """
        if self.rank != 0:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Use energy-based sampling with annealed Langevin dynamics
            if self.is_distributed:
                samples = self.model.module.sample(num_samples, self.device)
            else:
                samples = self.model.generate_sample(num_samples, self.device)
            
            # Save samples using parent class method
            self._save_samples(samples, epoch)
    
    def _log_additional_metrics(self, loss: torch.Tensor, epoch: int):
        """Log additional metrics specific to energy-based models.
        
        In addition to the base metrics, logs the energy scale, regularization
        weight, and Langevin dynamics parameters to track their values during
        training. Only logs on rank 0 in distributed mode.
        
        Args:
            loss (torch.Tensor): Current training loss.
                Passed to parent method for basic metric logging.
            epoch (int): Current epoch number.
                Used for synchronizing logging steps.
        """
        super()._log_additional_metrics(loss, epoch)
        
        if self.rank == 0 and self.config.get('use_wandb', False):
            import wandb
            # Log energy model parameters
            model = self.model.module if self.is_distributed else self.model
            wandb.log({
                'energy_scale': model.loss_fn.energy_scale,
                'regularization_weight': model.loss_fn.regularization_weight,
                'langevin_step_size': model.langevin_step_size
            }, step=epoch) 