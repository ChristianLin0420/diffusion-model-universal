from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseDiffusion(nn.Module, ABC):
    """Base class for all diffusion models."""
    
    def __init__(self, config):
        """
        Initialize the base diffusion model.
        
        Args:
            config (dict): Model configuration parameters
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, x, t):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            t (torch.Tensor): Timestep tensor
            
        Returns:
            torch.Tensor: Model output
        """
        pass
    
    @abstractmethod
    def loss_function(self, x):
        """
        Calculate the loss for training.
        
        Args:
            x (torch.Tensor): Input data
            
        Returns:
            torch.Tensor: Calculated loss
        """
        pass
    
    @abstractmethod
    def sample(self, batch_size, device):
        """
        Generate samples from the model.
        
        Args:
            batch_size (int): Number of samples to generate
            device (torch.device): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        pass
    
    def save(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to the checkpoint
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config'] 