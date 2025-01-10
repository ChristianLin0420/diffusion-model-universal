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
    - Multi-GPU training support
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import os
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional

class DDPMTrainer:
    """Trainer class for DDPM models.
    
    This class handles the training process for Denoising Diffusion Probabilistic
    Models. It provides functionality for training, validation, sample generation,
    and checkpoint management, with support for distributed training.
    
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
        rank (int, optional): Process rank for distributed training.
            Defaults to 0 (single GPU training).
        world_size (int, optional): Total number of processes.
            Defaults to 1 (single GPU training).
    
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
        rank (int): Process rank for distributed training
        world_size (int): Total number of processes
        is_distributed (bool): Whether running in distributed mode
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict,
        device: torch.device,
        rank: int = 0,
        world_size: int = 1
    ):
        """Initialize the DDPM trainer.
        
        Sets up the model, optimizer, output directories, and logging.
        Creates necessary directories and initializes Weights & Biases
        and TensorBoard if specified in the configuration.
        
        Raises:
            AttributeError: If model doesn't implement required methods.
            RuntimeError: If directories cannot be created or initialized.
        """
        # Verify model has required methods
        if not hasattr(model, 'loss_function') or not hasattr(model, 'sample'):
            raise AttributeError(
                "Model must implement 'loss_function' and 'sample' methods"
            )
        
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = world_size > 1
        self.device = device
        self.config = config
        
        # Get model name for logging
        self.model_name = config.get('model_name', model.__class__.__name__)
        
        # Move model to device
        self.model = model.to(device)
        
        # Wrap model with DDP if using distributed training
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True
            )
        
        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-4),
            betas=(config.get('beta1', 0.9), config.get('beta2', 0.999))
        )
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup directories and logging (only on rank 0)
        if self.rank == 0:
            try:
                self.output_dir = config.get('output_dir', 'outputs')
                self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
                self.sample_dir = os.path.join(self.output_dir, 'samples')
                self.log_dir = os.path.join(self.output_dir, 'logs')
                
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                os.makedirs(self.sample_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
                
                # Initialize wandb if enabled
                if config.get('use_wandb', False):
                    wandb.init(
                        project=config.get('wandb_project', 'diffusion-models'),
                        name=f"{self.model_name}_{config.get('run_name', '')}",
                        group=config.get('group', 'default'),
                        config={
                            **config,
                            'model_name': self.model_name,
                            'model_parameters': sum(p.numel() for p in model.parameters()),
                            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                            'dataset': config.get('dataset', 'unknown'),
                            'batch_size': config.get('batch_size', None),
                            'image_size': config.get('image_size', None),
                            'device': str(device)
                        }
                    )
                
                # Initialize TensorBoard if enabled
                if config.get('use_tensorboard', False):
                    self.writer = SummaryWriter(
                        log_dir=os.path.join(
                            self.log_dir,
                            f"{self.model_name}_{config.get('run_name', '')}"
                        )
                    )
                else:
                    self.writer = None
                    
            except Exception as e:
                raise RuntimeError(f"Failed to initialize directories and logging: {str(e)}")
        
        # Track best validation loss
        self.best_val_loss = float('inf')
    
    def _log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None, prefix: str = ''):
        """Log metrics to both Wandb and TensorBoard.
        
        Helper method to log metrics consistently across both logging platforms.
        Only logs on rank 0 in distributed mode.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values.
            step (Optional[int]): Current step for logging. If None, uses global step.
            prefix (str, optional): Prefix for metric names. Defaults to ''.
        """
        if self.rank != 0:
            return
            
        # Add prefix to metric names if provided
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Log to wandb if enabled
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=step)
        
        # Log to tensorboard if enabled
        if self.writer is not None:
            for name, value in metrics.items():
                self.writer.add_scalar(name, value, step)
    
    def train(self, num_epochs: int):
        """Train the DDPM model.
        
        Runs the training loop for the specified number of epochs. For each epoch:
        1. Trains on the entire training dataset
        2. Performs validation at specified intervals
        3. Computes and logs the average losses
        4. Generates samples at specified intervals
        5. Saves checkpoints at specified intervals
        
        In distributed mode, metrics are synchronized across processes and
        logging/saving operations only occur on rank 0.
        
        Args:
            num_epochs (int): Number of epochs to train for.
                Training will continue from the current epoch if resuming.
        """
        try:
            global_step = 0
            # Calculate validation interval to be roughly 10 times per epoch
            steps_per_epoch = len(self.train_loader)
            val_interval = max(1, min(
                self.config.get('val_interval', steps_per_epoch // 10),
                steps_per_epoch
            ))
            
            # Log hyperparameters
            if self.rank == 0:
                hparams = {
                    'learning_rate': self.config.get('learning_rate', 2e-4),
                    'batch_size': self.config.get('batch_size', None),
                    'num_epochs': num_epochs,
                    'val_interval': val_interval,
                    'model_name': self.model_name
                }
                if self.writer is not None:
                    self.writer.add_hparams(hparams, {'status': 1})
            
            for epoch in range(num_epochs):
                # Set model to train mode
                self.model.train()
                
                # Reset metrics
                total_loss = 0.0
                num_batches = 0
                
                # Create progress bar on rank 0
                if self.rank == 0:
                    pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
                else:
                    pbar = self.train_loader
                
                # Training loop
                for batch in pbar:
                    # Move batch to device
                    images = batch[0].to(self.device)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    loss = (
                        self.model.module.loss_function(images)
                        if self.is_distributed
                        else self.model.loss_function(images)
                    )
                    
                    # Backward pass
                    loss.backward()
                    self.optimizer.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Update progress bar on rank 0
                    if self.rank == 0:
                        pbar.set_postfix({'loss': loss.item()})
                    
                    # Validation
                    if global_step % val_interval == 0:
                        val_loss = self.validate()
                        self.model.train()  # Switch back to train mode
                        
                        # Log validation metrics
                        if self.rank == 0:
                            metrics = {
                                'loss': val_loss,
                                'step': global_step,
                                'epoch': epoch + global_step / steps_per_epoch
                            }
                            self._log_metrics(metrics, step=global_step, prefix='validation')
                        
                        # Save best model on rank 0
                        if self.rank == 0 and val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(epoch + 1, is_best=True)
                    
                    global_step += 1
                
                # Synchronize metrics across processes
                if self.is_distributed:
                    # Convert to tensors for all_reduce
                    loss_tensor = torch.tensor([total_loss], device=self.device)
                    batch_tensor = torch.tensor([num_batches], device=self.device)
                    
                    # Synchronize across processes
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
                    
                    # Update values
                    total_loss = loss_tensor.item()
                    num_batches = batch_tensor.item()
                    
                    # Average across processes
                    total_loss = total_loss / self.world_size
                    num_batches = num_batches / self.world_size
                
                avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
                
                # Log metrics on rank 0
                if self.rank == 0:
                    metrics = {
                        'loss': avg_loss,
                        'epoch': epoch + 1,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    self._log_metrics(metrics, step=global_step, prefix='train')
                    
                    # Generate and save samples
                    if (epoch + 1) % self.config.get('sample_interval', 5) == 0:
                        self.generate_samples(epoch + 1)
                    
                    # Save checkpoint
                    if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                        self.save_checkpoint(epoch + 1)
        
        except Exception as e:
            print(f"Error during training: {str(e)}")
            # Save emergency checkpoint on rank 0
            if self.rank == 0:
                self.save_checkpoint(epoch + 1, is_emergency=True)
            raise  # Re-raise the exception after saving checkpoint
    
    def validate(self) -> float:
        """Perform validation on the validation dataset.
        
        Computes the average loss on the validation dataset. The validation
        is performed with torch.no_grad() for memory efficiency. In distributed
        mode, the loss is synchronized and averaged across all processes.
        
        Returns:
            float: Average validation loss across all validation batches.
                In distributed mode, this is the synchronized average across
                all processes.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch in self.val_loader:
                    # Move batch to device
                    images = batch[0].to(self.device)
                    
                    # Compute loss
                    loss = (
                        self.model.module.loss_function(images)
                        if self.is_distributed
                        else self.model.loss_function(images)
                    )
                    
                    # Accumulate loss
                    total_loss += loss.item()
                    num_batches += 1
            
            # Synchronize metrics across processes
            if self.is_distributed:
                # Convert to tensors for all_reduce
                loss_tensor = torch.tensor([total_loss], device=self.device)
                batch_tensor = torch.tensor([num_batches], device=self.device)
                
                # Synchronize across processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
                
                # Update values
                total_loss = loss_tensor.item()
                num_batches = batch_tensor.item()
                
                # Average across processes
                total_loss = total_loss / self.world_size
                num_batches = num_batches / self.world_size
            
            return total_loss / num_batches if num_batches > 0 else float('inf')
            
        except Exception as e:
            print(f"Error during validation: {str(e)}")
            return float('inf')
    
    def test(self) -> float:
        """Evaluate the model on the test dataset.
        
        Computes the average loss on the test dataset. Similar to validation
        but uses the test dataloader instead. Only called after training is
        complete. In distributed mode, the loss is synchronized across processes.
        
        Returns:
            float: Average test loss across all test batches.
                In distributed mode, this is the synchronized average across
                all processes.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        try:
            with torch.no_grad():
                for batch in self.test_loader:
                    # Move batch to device
                    images = batch[0].to(self.device)
                    
                    # Compute loss
                    loss = (
                        self.model.module.loss_function(images)
                        if self.is_distributed
                        else self.model.loss_function(images)
                    )
                    
                    # Accumulate loss
                    total_loss += loss.item()
                    num_batches += 1
            
            # Synchronize metrics across processes
            if self.is_distributed:
                # Convert to tensors for all_reduce
                loss_tensor = torch.tensor([total_loss], device=self.device)
                batch_tensor = torch.tensor([num_batches], device=self.device)
                
                # Synchronize across processes
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(batch_tensor, op=dist.ReduceOp.SUM)
                
                # Update values
                total_loss = loss_tensor.item()
                num_batches = batch_tensor.item()
                
                # Average across processes
                total_loss = total_loss / self.world_size
                num_batches = num_batches / self.world_size
            
            test_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            
            # Log test metrics on rank 0
            if self.rank == 0 and self.config.get('use_wandb', False):
                wandb.log({
                    'test_loss': test_loss
                })
            
            return test_loss
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            return float('inf')
    
    def generate_samples(self, epoch: int, num_samples: int = 16):
        """Generate and save samples using the DDPM model.
        
        Uses the model's sampling process to generate new images. The samples
        are saved to the samples directory with the epoch number in the filename.
        In distributed mode, this only runs on rank 0 to avoid duplicate samples.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the output file.
            num_samples (int, optional): Number of samples to generate.
                Should be a perfect square for grid visualization.
                Defaults to 16.
        """
        if self.rank != 0:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Generate samples
            if self.is_distributed:
                samples = self.model.module.sample(num_samples, self.device)
            else:
                samples = self.model.sample(num_samples, self.device)
            
            # Create grid and save
            grid = make_grid(samples, nrow=int(num_samples ** 0.5))
            save_path = os.path.join(self.sample_dir, f'samples_epoch_{epoch}.png')
            save_image(grid, save_path)
            
            # Log samples
            if self.config.get('use_wandb', False):
                wandb.log({
                    f'{self.model_name}/samples': wandb.Image(grid),
                    'epoch': epoch
                })
            
            if self.writer is not None:
                self.writer.add_image(
                    f'{self.model_name}/samples',
                    grid,
                    global_step=epoch
                )
    
    def save_checkpoint(self, epoch: int, is_best: bool = False, is_emergency: bool = False):
        """Save a checkpoint of the model.
        
        Saves the current state of the model, optimizer, and training metrics.
        In distributed mode, this only runs on rank 0 to avoid duplicate saves.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the checkpoint file.
            is_best (bool, optional): Whether this is the best model so far.
                If True, saves an additional copy as 'best_model.pt'.
                Defaults to False.
            is_emergency (bool, optional): Whether this is an emergency save.
                If True, saves with an emergency suffix.
                Defaults to False.
        """
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if self.is_distributed else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        filename = f'checkpoint_epoch_{epoch}.pt'
        if is_emergency:
            filename = f'emergency_checkpoint_epoch_{epoch}.pt'
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        
        # Save best model if needed
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
        
        # Log to wandb if enabled
        if self.config.get('use_wandb', False):
            wandb.save(path)
            if is_best:
                wandb.save(best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a model checkpoint.
        
        In distributed mode, loads the state dict into the module.
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint.get('epoch', 0)
    
    def cleanup(self):
        """Perform cleanup operations.
        
        Closes wandb connection and tensorboard writer if they were used,
        and performs any necessary distributed cleanup. Should be called
        when training is complete or if an error occurs.
        """
        if self.rank == 0:
            if self.config.get('use_wandb', False):
                wandb.finish()
            if self.writer is not None:
                self.writer.close()
        
        if self.is_distributed:
            dist.destroy_process_group()
    
    def __del__(self):
        """Destructor to ensure cleanup is performed."""
        self.cleanup() 