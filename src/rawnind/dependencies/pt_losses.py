# -*- coding: utf-8 -*-
"""
Loss functions and metrics for image quality assessment.

This module provides PyTorch implementations of various image quality metrics
and corresponding loss functions, primarily focused on the Multi-Scale Structural
Similarity Index (MS-SSIM) for image quality assessment.

The module includes:
1. Loss functions suitable for neural network training (inversions of metrics)
2. Metrics for model evaluation
3. Dictionaries mapping string names to loss/metric functions
"""

import torch
import pytorch_msssim

class MS_SSIM_loss(pytorch_msssim.MS_SSIM):
    """Multi-Scale Structural Similarity loss function.
    
    This class wraps pytorch_msssim's MS_SSIM implementation and inverts the score
    (1 - score) to convert it from a similarity metric (higher is better) to a loss
    function (lower is better) suitable for optimization.
    
    MS-SSIM is perceptually motivated and often correlates better with human judgment
    of image quality than pixel-wise losses like MSE.
    """

    def __init__(self, data_range=1.0, **kwargs):
        """Initialize the MS-SSIM loss.
        
        Args:
            data_range: Value range of input images (usually 1.0 for normalized images)
            **kwargs: Additional arguments passed to pytorch_msssim.MS_SSIM
        """
        super().__init__(data_range=data_range, **kwargs)

    def forward(self, input, target):
        """Calculate the MS-SSIM loss between input and target images.
        
        Args:
            input: Predicted images, shape [N, C, H, W]
            target: Ground truth images, shape [N, C, H, W]
            
        Returns:
            A loss value (1 - MS-SSIM) where lower values indicate better image quality
        """
        return 1 - super().forward(input, target)


def ms_ssim_metric(input, target, data_range=1.0, **kwargs):
    """
    Calculates the MS-SSIM score. Higher is better.
    This is derived from the loss function to ensure a single implementation.
    """
    return 1 - MS_SSIM_loss(data_range=data_range, **kwargs)(input, target)


class PSNR_metric(torch.nn.Module):
    """Peak Signal-to-Noise Ratio metric for evaluation.
    
    PSNR is a commonly used metric in image processing that measures the ratio
    between the maximum possible power of a signal and the power of corrupting noise.
    Higher values indicate better image quality.
    
    PSNR = 20 * log10(MAX_PIXEL_VALUE / sqrt(MSE))
    """
    
    def __init__(self, data_range=1.0, **kwargs):
        """Initialize the PSNR metric.
        
        Args:
            data_range: Value range of input images (usually 1.0 for normalized images)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.data_range = data_range
        
    def forward(self, input, target):
        """Calculate PSNR between input and target images.
        
        Args:
            input: Predicted images, shape [N, C, H, W]
            target: Ground truth images, shape [N, C, H, W]
            
        Returns:
            PSNR value in dB (higher is better)
        """
        mse = torch.nn.functional.mse_loss(input, target)
        if mse == 0:
            return torch.tensor(float('inf'), device=input.device)
        
        psnr = 20 * torch.log10(self.data_range / torch.sqrt(mse))
        return psnr

class L1_loss(torch.nn.L1Loss):
    pass

class MSE_loss(torch.nn.MSELoss):
    pass

# Dictionary mapping loss function names to their implementation classes
losses = {
    "l1"         : L1_loss,
    "mse"        : MSE_loss,  # Standard Mean Squared Error loss
    "ms_ssim": MS_SSIM_loss,  # Perceptual MS-SSIM loss (1 - MS_SSIM)
    "ms_ssim_loss": MS_SSIM_loss  # Alias for consistency with legacy code
}

# Dictionary mapping metric names to their implementation classes
metrics = {
    "ms_ssim"    : ms_ssim_metric,  # Multi-Scale Structural Similarity metric (higher = better)
    "mse"        : MSE_loss,  # Mean Squared Error (lower = better)
    "psnr"       : PSNR_metric,  # Peak Signal-to-Noise Ratio (higher = better)
}