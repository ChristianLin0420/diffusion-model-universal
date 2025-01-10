"""DDPM Model Trainer Implementation.

This module provides a trainer class for Denoising Diffusion Probabilistic Models
(DDPM). It handles the training loop, checkpoint management, sample generation,
and logging functionality.

Key Features:
    - Training loop with progress tracking
    - Validation during training
    - Checkpoint saving and loading
    - Sample generation during training
    - Weights & Biases integration
    - Automatic output directory management
    - Configurable training parameters
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import os
from tqdm import tqdm
import wandb
from typing import Dict, Optional

class DDPMTrainer:
    """Trainer class for DDPM models.
    
    This class handles the training process for Denoising Diffusion Probabilistic
    Models. It provides functionality for training, validation, sample generation,
    and checkpoint management.
    
    Args:
        model (nn.Module): The DDPM model to train.
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
        device (torch.device): Device to train on (CPU/GPU).
    
    Attributes:
        model (nn.Module): The DDPM model
        optimizer (Adam): Adam optimizer
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        test_loader (DataLoader): Test data loader
        config (Dict): Training configuration
        device (torch.device): Training device
        output_dir (str): Base output directory
        checkpoint_dir (str): Directory for checkpoints
        sample_dir (str): Directory for generated samples
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        """Initialize the DDPM trainer.
        
        Sets up the model, optimizer, output directories, and logging.
        Creates necessary directories and initializes Weights & Biases
        if specified in the configuration.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.get('learning_rate', 2e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999))
        )
        
        # Setup directories
        self.output_dir = config.get('output_dir', 'outputs')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.sample_dir = os.path.join(self.output_dir, 'samples')
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Initialize wandb
        if config.get('use_wandb', False):
            wandb.init(
                project=config.get('wandb_project', 'diffusion-models'),
                config=config
            )
        
        # Track best validation loss
        self.best_val_loss = float('inf')
    
    def train(self, num_epochs: int):
        """Train the DDPM model.
        
        Runs the training loop for the specified number of epochs. For each epoch:
        1. Trains on the entire training dataset
        2. Performs validation at specified intervals
        3. Computes and logs the average losses
        4. Generates samples at specified intervals
        5. Saves checkpoints at specified intervals
        
        Args:
            num_epochs (int): Number of epochs to train for.
                Training will continue from the current epoch if resuming.
        """
        global_step = 0
        val_interval = self.config.get('val_interval', len(self.train_loader) // 10)
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            with tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}') as pbar:
                for batch in pbar:
                    # Move batch to device
                    images = batch[0].to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    loss = self.model.loss_function(images)
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update progress bar
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                    
                    # Validation
                    if global_step % val_interval == 0:
                        val_loss = self.validate()
                        self.model.train()  # Switch back to train mode
                        
                        # Log validation metrics
                        if self.config.get('use_wandb', False):
                            wandb.log({
                                'val_loss': val_loss,
                                'step': global_step
                            })
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(epoch + 1, is_best=True)
                    
                    global_step += 1
            
            # Calculate average loss
            avg_loss = total_loss / len(self.train_loader)
            
            # Log metrics
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': avg_loss
                })
            
            # Generate and save samples
            if (epoch + 1) % self.config.get('sample_interval', 5) == 0:
                self.generate_samples(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch + 1)
    
    def validate(self) -> float:
        """Perform validation.
        
        Computes the average loss on the validation dataset.
        
        Returns:
            float: Average validation loss.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch[0].to(self.device)
                loss = self.model.loss_function(images)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def test(self) -> float:
        """Perform testing.
        
        Computes the average loss on the test dataset.
        
        Returns:
            float: Average test loss.
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch[0].to(self.device)
                loss = self.model.loss_function(images)
                total_loss += loss.item()
        
        return total_loss / len(self.test_loader)
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """Generate and save samples from the model.
        
        Generates a grid of samples from the current model state and saves
        them to disk. Also logs the samples to Weights & Biases if enabled.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the output file.
            num_samples (int, optional): Number of samples to generate.
                Should be a perfect square. Defaults to 16.
        """
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(num_samples, self.device)
            
            # Save samples
            grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(self.sample_dir, f'epoch_{epoch}.png'))
            
            if self.config.get('use_wandb', False):
                wandb.log({
                    'samples': wandb.Image(grid)
                })
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save a checkpoint of the model.
        
        Saves the current state of the model, optimizer, and training
        configuration to disk. Also logs the checkpoint to Weights & Biases
        if enabled.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the checkpoint file.
            is_best (bool, optional): Whether this is the best model so far.
                If True, saves an additional copy as 'best_model.pt'.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        if self.config.get('use_wandb', False):
            wandb.save(path)
            if is_best:
                wandb.save(best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a model checkpoint.
        
        Restores the model and optimizer state from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
                Should be a file created by save_checkpoint().
        
        Returns:
            int: The epoch number when the checkpoint was created.
        
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            RuntimeError: If the checkpoint is incompatible with the model.
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch'] 