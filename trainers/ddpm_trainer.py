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

import datetime
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
import time

from utils.config_utils import print_config
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
        if not hasattr(model, 'loss_function') or not hasattr(model, 'generate_samples'):
            raise AttributeError(
                "Model must implement 'loss_function' and 'generate_samples' methods"
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
        
        # Initialize EMA model if enabled
        self.ema_decay = config.get('training', {}).get('ema_decay', 0.9999)
        if self.ema_decay > 0:
            self.ema_model = self._copy_model_to_ema()
            self.ema_model.eval()
        else:
            self.ema_model = None
        
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
            lr=float(config.get('training', {}).get('learning_rate', 2e-4)),
            betas=(float(config.get('training', {}).get('beta1', 0.9)), float(config.get('training', {}).get('beta2', 0.999)))
        )
        
        # Setup learning rate scheduler
        scheduler_config = config.get('training', {}).get('scheduler', {})
        
        # Print scheduler configuration if on rank 0
        if self.rank == 0:
            if scheduler_config:
                print_config("Scheduler Configuration", scheduler_config)
        
        if scheduler_config:
            scheduler_type = scheduler_config.get('type', 'cosine')
            total_steps = int(len(train_loader) * config.get('training', {}).get('num_epochs', 500))
            warmup_steps = int(scheduler_config.get('warmup_steps', 0))
            min_lr = float(scheduler_config.get('min_lr', 1e-6))
            
            if scheduler_type == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=int(total_steps - warmup_steps),
                    eta_min=min_lr
                )
            elif scheduler_type == 'linear':
                def lr_lambda(step):
                    if step < warmup_steps:
                        return float(step) / float(max(1, warmup_steps))
                    return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            elif scheduler_type == 'step':
                step_size = int(scheduler_config.get('step_size', total_steps // 4))  # Default to 4 steps
                gamma = float(scheduler_config.get('gamma', 0.1))
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=step_size,
                    gamma=gamma
                )
            elif scheduler_type == 'exponential':
                gamma = float(scheduler_config.get('gamma', 0.95))
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer,
                    gamma=gamma
                )
            elif scheduler_type == 'one_cycle':
                max_lr = float(config.get('training', {}).get('learning_rate', 2e-4))
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=max_lr,
                    total_steps=total_steps,
                    pct_start=float(scheduler_config.get('pct_start', 0.3)),
                    anneal_strategy='cos',
                    final_div_factor=float(scheduler_config.get('final_div_factor', 1e4))
                )
            else:
                self.scheduler = None
                if self.rank == 0:
                    print(f"Warning: Unknown scheduler type '{scheduler_type}'. No scheduler will be used.")
        else:
            self.scheduler = None
            if self.rank == 0:
                print("No scheduler configuration found. Training will proceed without a learning rate scheduler.")
        
        # Setup data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup directories and logging (only on rank 0)
        if self.rank == 0:
            try:
                self.output_dir = config.get('output', {}).get('output_dir', 'outputs')
                self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
                self.sample_dir = os.path.join(self.output_dir, 'samples')
                self.log_dir = os.path.join(self.output_dir, 'logs')
                
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                os.makedirs(self.sample_dir, exist_ok=True)
                os.makedirs(self.log_dir, exist_ok=True)
                
                # Initialize wandb if enabled
                if config.get('logging', {}).get('use_wandb', False):
                    wandb.init(
                        project=config.get('logging', {}).get('wandb_project', 'diffusion-models'),
                        name=f"{self.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        group=config.get('logging', {}).get('group', 'default'),
                        config={
                            **config,
                            'model_name': self.model_name,
                            'model_parameters': sum(p.numel() for p in model.parameters()),
                            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                            'dataset': config.get('data', {}).get('dataset', 'unknown'),
                            'batch_size': config.get('training', {}).get('batch_size', None),
                            'image_size': config.get('data', {}).get('image_size', None),
                            'device': str(device)
                        }
                    )
                    
                    # Update wandb config with scheduler settings if present
                    if scheduler_config:
                        wandb.config.update({
                            'scheduler': {
                                'type': scheduler_config.get('type', 'cosine'),
                                'warmup_steps': scheduler_config.get('warmup_steps', 0),
                                'min_lr': scheduler_config.get('min_lr', 1e-6),
                                'cycle_length': scheduler_config.get('cycle_length', 50),
                                'cycle_mult': scheduler_config.get('cycle_mult', 2),
                                'step_size': scheduler_config.get('step_size', 100),
                                'gamma': scheduler_config.get('gamma', 0.1)
                            }
                        }, allow_val_change=True)
                    else:
                        wandb.config.update({'scheduler': None}, allow_val_change=True)
                
                # Initialize TensorBoard if enabled
                if config.get('logging', {}).get('use_tensorboard', False):
                    self.writer = SummaryWriter(
                        log_dir=os.path.join(
                            self.log_dir,
                            f"{self.model_name}"
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
        
        # Add memory usage metrics if tracking is enabled
        if self.config.get('logging', {}).get('track_memory_usage', False):
            metrics.update({
                f"{prefix}/gpu_memory_allocated": torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
                f"{prefix}/gpu_memory_cached": torch.cuda.memory_reserved(self.device) / 1024**2  # MB
            })
        
        # Log to wandb if enabled
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.log(metrics, step=step)
        
        # Log to tensorboard if enabled
        if self.writer is not None:
            for name, value in metrics.items():
                # Skip histogram values since they're not compatible with add_scalar
                if not isinstance(value, (wandb.Histogram, dict)):
                    self.writer.add_scalar(name, value, step)
                else:
                    # Handle histogram data
                    if isinstance(value, wandb.Histogram):
                        # Convert histogram data to tensor before logging
                        hist_data = torch.tensor(value.histogram)
                        self.writer.add_histogram(name, hist_data, step)
                    elif isinstance(value, dict) and any('hist' in k for k in value.keys()):
                        # Assuming histogram data is stored in tensor form
                        for k, v in value.items():
                            if 'hist' in k and isinstance(v, torch.Tensor):
                                self.writer.add_histogram(k, v, step)
                    elif isinstance(value, list):
                        # Convert list data to tensor before logging
                        list_data = torch.tensor(value)
                        self.writer.add_histogram(name, list_data, step)
                    
    
    def _log_model_gradients(self, step: int):
        """Log model gradients and weights statistics.
        
        Computes and logs gradient norms, weight norms, and other statistics
        for model parameters. Only logs on rank 0 in distributed mode.
        
        Args:
            step (int): Current step for logging.
        """
        if self.rank != 0 or not self.config.get('logging', {}).get('track_grad_norm', False):
            return
        
        grad_norms = {}
        weight_norms = {}
        grad_hist = {}
        weight_hist = {}
        
        # Get the actual model (handle DDP case)
        model = self.model.module if self.is_distributed else self.model
        
        # Compute statistics for each parameter
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Gradient statistics
                grad_norm = param.grad.norm().item()
                grad_norms[f"grad_norm/{name}"] = grad_norm
                grad_hist[f"grad_hist/{name}"] = param.grad.cpu()
                
                # Weight statistics
                weight_norm = param.norm().item()
                weight_norms[f"weight_norm/{name}"] = weight_norm
                weight_hist[f"weight_hist/{name}"] = param.cpu()
        
        # Compute global statistics
        total_grad_norm = sum(norm * norm for norm in grad_norms.values()) ** 0.5
        total_weight_norm = sum(norm * norm for norm in weight_norms.values()) ** 0.5
        
        # Log to wandb
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.log({
                "model/total_grad_norm": total_grad_norm,
                "model/total_weight_norm": total_weight_norm,
                "model/grad_norms": grad_norms,
                "model/weight_norms": weight_norms,
                "model/param_histograms": {
                    **{k: wandb.Histogram(v) for k, v in grad_hist.items()},
                    **{k: wandb.Histogram(v.detach()) for k, v in weight_hist.items()}
                }
            }, step=step)
    
    def _log_performance_metrics(self, batch_start_time: float, batch_size: int, step: int):
        """Log performance-related metrics.
        
        Args:
            batch_start_time (float): Start time of the batch
            batch_size (int): Size of the batch
            step (int): Current step for logging
        """
        if self.rank != 0 or not self.config.get('logging', {}).get('track_time_metrics', False):
            return
            
        batch_time = time.time() - batch_start_time
        samples_per_sec = batch_size / batch_time
        
        metrics = {
            'performance/batch_time': batch_time,
            'performance/samples_per_second': samples_per_sec,
            'performance/steps_per_second': 1.0 / batch_time
        }
        
        # Track GPU metrics if enabled
        if self.config.get('logging', {}).get('track_gpu_stats', False):
            metrics.update({
                'performance/gpu_utilization': torch.cuda.utilization(self.device),
                'performance/gpu_memory_used': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'performance/gpu_memory_cached': torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            })
        
        self._log_metrics(metrics, step=step)
    
    def _log_optimizer_stats(self, step: int):
        """Log optimizer-related statistics.
        
        Args:
            step (int): Current step for logging
        """
        if self.rank != 0 or not self.config.get('logging', {}).get('track_optimizer_stats', False):
            return
            
        # Get optimizer state
        state = self.optimizer.state_dict()['state']
        
        # Compute average statistics across all parameters
        avg_exp_avg = 0.0
        avg_exp_avg_sq = 0.0
        num_params = 0
        
        for param_state in state.values():
            if 'exp_avg' in param_state:
                avg_exp_avg += param_state['exp_avg'].abs().mean().item()
                avg_exp_avg_sq += param_state['exp_avg_sq'].abs().mean().item()
                num_params += 1
        
        if num_params > 0:
            metrics = {
                'optimizer/average_first_moment': avg_exp_avg / num_params,
                'optimizer/average_second_moment': avg_exp_avg_sq / num_params,
                'optimizer/learning_rate': self.optimizer.param_groups[0]['lr'],
                'optimizer/beta1': self.optimizer.param_groups[0]['betas'][0],
                'optimizer/beta2': self.optimizer.param_groups[0]['betas'][1]
            }
            self._log_metrics(metrics, step=step)
    
    def _log_diffusion_metrics(self, model, step: int):
        """Log diffusion process specific metrics.
        
        Args:
            model: The DDPM model (unwrapped from DDP if necessary)
            step (int): Current step for logging
        """
        if self.rank != 0:
            return
            
        config = self.config.get('logging', {})
        if config.get('track_noise_schedule', False):
            metrics = {
                'diffusion/beta_schedule': wandb.Histogram(model.betas.cpu()),
                'diffusion/alpha_schedule': wandb.Histogram(model.alphas.cpu()),
                'diffusion/alpha_cumprod': wandb.Histogram(model.alphas_cumprod.cpu())
            }
            self._log_metrics(metrics, step=step)
    
    def _copy_model_to_ema(self) -> nn.Module:
        """Create a copy of the model for EMA."""
        ema_model = type(self.model)(self.config['model_config'])
        ema_model.load_state_dict(self.model.state_dict())
        ema_model = ema_model.to(self.device)
        return ema_model
    
    def _update_ema_model(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return
            
        with torch.no_grad():
            model_params = dict(self.model.named_parameters())
            ema_params = dict(self.ema_model.named_parameters())
            
            for name, param in model_params.items():
                if self.is_distributed:
                    name = name.replace('module.', '')

                if name in ema_params:  
                    # Update EMA parameter
                    ema_params[name].data.mul_(self.ema_decay).add_(
                        param.data, alpha=(1 - self.ema_decay)
                    )
    
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
                self.config.get('training', {}).get('val_interval', steps_per_epoch // 10),
                steps_per_epoch
            ))
            
            # Log hyperparameters
            if self.rank == 0:
                hparams = {
                    'learning_rate': self.config.get('training', {}).get('learning_rate', 2e-4),
                    'batch_size': self.config.get('training', {}).get('batch_size', None),
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
                for batch_idx, batch in enumerate(pbar):
                    batch_start_time = time.time()
                    
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
                    
                    # Update EMA model
                    if self.ema_model is not None:
                        self._update_ema_model()
                    
                    # Update learning rate
                    if self.scheduler is not None:
                        # Step the scheduler based on its type
                        if isinstance(self.scheduler, (
                            torch.optim.lr_scheduler.OneCycleLR,
                            torch.optim.lr_scheduler.LambdaLR
                        )):
                            self.scheduler.step()
                        else:
                            # These schedulers should be stepped once per epoch
                            if batch_idx == len(self.train_loader) - 1:
                                self.scheduler.step()
                    
                    # Update metrics
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Log detailed training metrics
                    if self.rank == 0 and self.config.get('logging', {}).get('use_wandb', False):
                        # Get current learning rate
                        current_lr = self.optimizer.param_groups[0]['lr']
                        
                        # Log training dynamics
                        train_metrics = {
                            'train/loss': loss.item(),
                            'train/learning_rate': current_lr,
                            'train/epoch': epoch + batch_idx / len(self.train_loader),
                            'train/global_step': global_step,
                        }
                        
                        # Log metrics
                        self._log_metrics(train_metrics, step=global_step)
    
                        # Log additional metrics periodically
                        if (global_step + 1) % self.config.get('logging', {}).get('gradient_logging_freq', 100) == 0:
                            self._log_model_gradients(global_step)
                            self._log_optimizer_stats(global_step)
                            model = self.model.module if self.is_distributed else self.model
                            self._log_diffusion_metrics(model, global_step)
                        
                        # Log performance metrics
                        self._log_performance_metrics(batch_start_time, images.size(0), global_step)
                    
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
                        if self.rank == 0 and val_loss < self.best_val_loss and (global_step % (self.config.get('training', {}).get('checkpoint_interval', 10) * steps_per_epoch) == 0):
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
                    if (epoch + 1) % self.config.get('training', {}).get('sample_interval', 5) == 0:
                        self.generate_samples(epoch + 1)
                    
                    # Save checkpoint
                    if (epoch + 1) % self.config.get('training', {}).get('checkpoint_interval', 10) == 0:
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
            if self.rank == 0 and self.config.get('logging', {}).get('use_wandb', False):
                wandb.log({
                    'test_loss': test_loss
                })
            
            return test_loss
            
        except Exception as e:
            print(f"Error during testing: {str(e)}")
            return float('inf')
    
    def generate_samples(self, epoch: int, num_samples: int = 8):
        """Generate and save samples using the DDPM model.
        
        Uses the model's sampling process to generate new images. For each sample,
        shows the progression from noise to final image at different timesteps.
        The samples are arranged in a grid where each row shows the denoising
        process for one sample.
        
        Args:
            epoch (int): Current epoch number.
                Used for naming the output file.
            num_samples (int, optional): Number of samples to generate.
                This will be the number of rows in the grid. Defaults to 8.
        """
        if self.rank != 0:
            return
            
        self.model.eval()
        with torch.no_grad():
            # Use trained model for sampling
            model = self.model.module if self.is_distributed else self.model
            
            # Generate samples with intermediate steps
            intermediate_samples = model.generate_samples_with_intermediates(
                batch_size=num_samples,
                device=self.device,
                save_interval=100
            )
            
            # Create grid
            # Reshape samples to create rows of denoising process
            rows = []
            for i in range(num_samples):
                row = [sample[i:i+1] for sample in intermediate_samples]
                rows.append(torch.cat(row, dim=0))
            
            # Combine all rows
            all_samples = torch.cat(rows, dim=0)
            
            # Create and save grid
            grid = make_grid(all_samples, nrow=11, padding=2)  # 11 images per row (noise + 10 intermediate steps)
            save_path = os.path.join(self.sample_dir, f'samples_epoch_{epoch}.png')
            save_image(grid, save_path)
            
            # Log samples
            if self.config.get('logging', {}).get('use_wandb', False):
                wandb.log({
                    f'{self.model_name}/denoising_process': wandb.Image(grid),
                    'epoch': epoch
                })
            
            if self.writer is not None:
                self.writer.add_image(
                    f'{self.model_name}/denoising_process',
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
            'ema_model_state_dict': self.ema_model.state_dict() if self.ema_model is not None else None,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None
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
        if self.config.get('logging', {}).get('use_wandb', False):
            wandb.save(path)
            if is_best:
                wandb.save(best_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a checkpoint and return the epoch number.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        
        Returns:
            int: The epoch number of the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if self.is_distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load EMA model if available
        if self.ema_model is not None and checkpoint.get('ema_model_state_dict') is not None:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        return checkpoint['epoch']
    
    def cleanup(self):
        """Perform cleanup operations.
        
        Closes wandb connection and tensorboard writer if they were used,
        and performs any necessary distributed cleanup. Should be called
        when training is complete or if an error occurs.
        """
        if self.rank == 0:
            if self.config.get('logging', {}).get('use_wandb', False):
                wandb.finish()
            if self.writer is not None:
                self.writer.close()
        
        # if self.is_distributed:
        #     dist.destroy_process_group()
    
    def __del__(self):
        """Destructor to ensure cleanup is performed."""
        self.cleanup() 