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
    """Trainer class for DDPM models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The DDPM model to train
            train_loader: Training data loader
            test_loader: Test data loader
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
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
    
    def train(self, num_epochs: int):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
        """
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
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """
        Generate and save samples from the model.
        
        Args:
            epoch: Current epoch number
            num_samples: Number of samples to generate
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
    
    def save_checkpoint(self, epoch: int):
        """
        Save a checkpoint of the model.
        
        Args:
            epoch: Current epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        
        if self.config.get('use_wandb', False):
            wandb.save(path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'] 