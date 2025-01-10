"""Diffusion Model Benchmarking Utilities.

This module provides benchmarking tools for evaluating diffusion models,
including standard metrics like FID, Inception Score, SSIM, and PSNR.

Key Features:
    - Fréchet Inception Distance (FID) calculation
    - Inception Score (IS) calculation
    - Structural Similarity (SSIM) measurement
    - Peak Signal-to-Noise Ratio (PSNR) calculation
    - Batch processing support
    - GPU acceleration
    - Progress tracking
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import inception_v3
import numpy as np
from scipy import linalg
from tqdm import tqdm
import torch.nn.functional as F
from typing import Tuple, List, Optional
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

class InceptionStatistics:
    """Helper class for calculating Inception-based metrics.
    
    This class handles the computation of Inception network features
    for both FID and Inception Score calculations.
    
    Args:
        device (torch.device): Device to run calculations on.
        use_fid (bool): Whether to compute features for FID.
            Defaults to True.
    """
    
    def __init__(self, device: torch.device, use_fid: bool = True):
        """Initialize the Inception model and preprocessing."""
        self.device = device
        self.model = inception_v3(pretrained=True, transform_input=True).to(device)
        self.model.eval()
        
        if use_fid:
            # Remove final classification layer for FID
            self.model.fc = nn.Identity()
        
        # Register hook to get features
        self.features = None
        def hook(module, input, output):
            self.features = output.detach()
        
        if use_fid:
            self.model.avgpool.register_forward_hook(hook)
    
    @torch.no_grad()
    def get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images using Inception network.
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W].
                Should be in range [-1, 1].
        
        Returns:
            torch.Tensor: Extracted features.
        """
        # Resize images to inception input size
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Move to device and get features
        images = images.to(self.device)
        _ = self.model(images)
        
        return self.features

def calculate_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """Calculate Fréchet Inception Distance between real and generated images.
    
    Args:
        real_features (torch.Tensor): Features from real images [N, D].
        fake_features (torch.Tensor): Features from generated images [N, D].
    
    Returns:
        float: The calculated FID score (lower is better).
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(dim=0), torch_cov(real_features)
    mu2, sigma2 = fake_features.mean(dim=0), torch_cov(fake_features)
    
    # Calculate FID
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy())
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(torch.from_numpy(covmean).to(sigma1.device))
    
    return float(fid)

def calculate_inception_score(features: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
    """Calculate Inception Score for generated images.
    
    Args:
        features (torch.Tensor): Inception logits for generated images [N, num_classes].
        splits (int): Number of splits to calculate mean/std. Defaults to 10.
    
    Returns:
        Tuple[float, float]: Mean and standard deviation of the Inception Score.
    """
    scores = []
    split_size = features.size(0) // splits
    
    for i in range(splits):
        split_features = features[i * split_size:(i + 1) * split_size]
        
        # Calculate p(y|x)
        probs = F.softmax(split_features, dim=1)
        
        # Calculate p(y)
        p_y = probs.mean(dim=0, keepdim=True)
        
        # Calculate KL divergence
        kl = probs * (torch.log(probs) - torch.log(p_y))
        
        # Calculate IS for this split
        split_score = torch.exp(kl.sum(dim=1).mean())
        scores.append(split_score.item())
    
    return float(np.mean(scores)), float(np.std(scores))

def torch_cov(m: torch.Tensor) -> torch.Tensor:
    """Calculate covariance matrix using PyTorch.
    
    Args:
        m (torch.Tensor): Matrix to calculate covariance for [N, D].
    
    Returns:
        torch.Tensor: Covariance matrix [D, D].
    """
    fact = 1.0 / (m.size(0) - 1)
    m = m - torch.mean(m, dim=0)
    mt = m.t()
    return fact * m.matmul(mt)

class DiffusionBenchmark:
    """Comprehensive benchmarking suite for diffusion models.
    
    This class provides methods to evaluate diffusion models using
    multiple metrics including FID, Inception Score, SSIM, and PSNR.
    
    Args:
        device (torch.device): Device to run calculations on.
        n_samples (int): Number of samples to generate for evaluation.
            Defaults to 50000 for FID calculation.
        batch_size (int): Batch size for generation and evaluation.
            Defaults to 32.
    
    Attributes:
        device (torch.device): Computation device
        n_samples (int): Number of samples for evaluation
        batch_size (int): Batch size for processing
        inception_stats (InceptionStatistics): Inception feature extractor
        ssim (StructuralSimilarityIndexMeasure): SSIM calculator
        psnr (PeakSignalNoiseRatio): PSNR calculator
    """
    
    def __init__(
        self,
        device: torch.device,
        n_samples: int = 50000,
        batch_size: int = 32
    ):
        """Initialize benchmark suite with specified parameters."""
        self.device = device
        self.n_samples = n_samples
        self.batch_size = batch_size
        
        # Initialize metrics
        self.inception_stats = InceptionStatistics(device)
        self.ssim = StructuralSimilarityIndexMeasure().to(device)
        self.psnr = PeakSignalNoiseRatio().to(device)
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        test_loader: torch.utils.data.DataLoader,
        noise_scheduler: Optional[object] = None
    ) -> dict:
        """Evaluate a diffusion model using multiple metrics.
        
        Args:
            model (nn.Module): The diffusion model to evaluate.
                Must implement sample() method.
            test_loader (DataLoader): DataLoader for test set.
                Used for real image statistics.
            noise_scheduler (Optional[object]): Noise scheduler for sampling.
                Only needed if model requires it.
        
        Returns:
            dict: Dictionary containing computed metrics:
                - fid: Fréchet Inception Distance
                - is_mean: Mean Inception Score
                - is_std: Standard deviation of Inception Score
                - ssim: Mean Structural Similarity
                - psnr: Mean Peak Signal-to-Noise Ratio
        """
        print("Collecting real image features...")
        real_features = []
        for batch in tqdm(test_loader):
            images = batch[0].to(self.device)
            features = self.inception_stats.get_features(images)
            real_features.append(features.cpu())
        real_features = torch.cat(real_features, dim=0)
        
        print("Generating samples and collecting features...")
        fake_features = []
        generated_images = []
        n_batches = self.n_samples // self.batch_size
        
        for _ in tqdm(range(n_batches)):
            # Generate samples
            samples = model.sample(self.batch_size, self.device)
            features = self.inception_stats.get_features(samples)
            fake_features.append(features.cpu())
            generated_images.append(samples.cpu())
        
        fake_features = torch.cat(fake_features, dim=0)
        generated_images = torch.cat(generated_images, dim=0)
        
        # Calculate FID
        fid = calculate_fid(real_features, fake_features)
        
        # Calculate Inception Score
        is_mean, is_std = calculate_inception_score(fake_features)
        
        # Calculate SSIM and PSNR
        ssim_scores = []
        psnr_scores = []
        
        for i, batch in enumerate(test_loader):
            if i >= n_batches:
                break
            real_batch = batch[0].to(self.device)
            fake_batch = generated_images[i * self.batch_size:(i + 1) * self.batch_size].to(self.device)
            
            ssim_scores.append(self.ssim(fake_batch, real_batch).item())
            psnr_scores.append(self.psnr(fake_batch, real_batch).item())
        
        return {
            'fid': fid,
            'is_mean': is_mean,
            'is_std': is_std,
            'ssim': np.mean(ssim_scores),
            'psnr': np.mean(psnr_scores)
        } 