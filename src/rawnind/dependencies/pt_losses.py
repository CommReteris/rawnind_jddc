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
4. Utility functions for determining valid dimensions for MS-SSIM calculation

The implementation uses pytorch_msssim as the backend for MS-SSIM calculation,
with previous implementations using piqa commented out due to compatibility issues.
"""

import torch

# import piqa  # disabled due to https://github.com/francois-rozet/piqa/issues/25
import pytorch_msssim
import sys


# from common.extlibs import DISTS_pt

# class MS_SSIM_loss(piqa.MS_SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)


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


class MS_SSIM_metric(pytorch_msssim.MS_SSIM):
    """Multi-Scale Structural Similarity metric for evaluation.
    
    Unlike MS_SSIM_loss, this class directly uses the MS-SSIM score without inversion,
    making it suitable for evaluation where higher values indicate better quality.
    
    Values range from 0 to 1, where 1 indicates perfect similarity between images.
    """

    def __init__(self, data_range=1.0, **kwargs):
        """Initialize the MS-SSIM metric.
        
        Args:
            data_range: Value range of input images (usually 1.0 for normalized images)
            **kwargs: Additional arguments passed to pytorch_msssim.MS_SSIM
        """
        super().__init__(data_range=data_range, **kwargs)


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


# class SSIM_loss(piqa.SSIM):
#     def __init__(self, **kwargs):
#         r""""""
#         super().__init__(**kwargs)
#     def forward(self, input, target):
#         return 1-super().forward(input, target)
#     # Note: SSIM implementation using piqa is commented out due to compatibility issues


# class DISTS_loss(DISTS_pt.DISTS):
#     def __init__(self, **kwargs):
#         super().__init__()
#
#     def forward(self, x, y):
#         return super().forward(x, y, require_grad=True, batch_average=True)
#     # Note: DISTS implementation is commented out and would require the DISTS_pt module


# Dictionary mapping loss function names to their implementation classes
losses = {
    "mse"        : torch.nn.MSELoss,  # Standard Mean Squared Error loss
    "ms_ssim_loss": MS_SSIM_loss  # Perceptual MS-SSIM loss (1 - MS_SSIM)
}  # "dists": DISTS_loss is commented out due to dependency issues

# Dictionary mapping metric names to their implementation classes
# Note: Python 3.8/3.10 compatibility workaround (can't use | operator for dict merging)
metrics = {
    "ms_ssim"    : MS_SSIM_metric,  # Multi-Scale Structural Similarity metric (higher = better)
    "ms_ssim_loss": MS_SSIM_loss,  # MS-SSIM loss function (lower = better) 
    "mse"        : torch.nn.MSELoss,  # Mean Squared Error (lower = better)
    "psnr"       : PSNR_metric,  # Peak Signal-to-Noise Ratio (higher = better)
    # "dists": DISTS_loss,  # DISTS metric commented out due to dependency issues
}

if __name__ == "__main__":
    """
    Utility script to find the minimum valid dimension for MS-SSIM calculation.
    
    MS-SSIM has minimum dimension requirements due to its multi-scale nature.
    This script attempts to find the smallest valid image dimension by testing
    increasingly larger square images until the calculation succeeds.
    
    Note: This test uses pytorch_msssim, not the commented-out piqa implementation.
    The result from previous tests was 162 as the minimum valid dimension.
    """


    def findvaliddim(start):
        """
        Recursively find the minimum valid dimension for MS-SSIM calculation.
        
        Args:
            start: Starting dimension to test
            
        Returns:
            The first valid dimension that works with MS-SSIM
        """
        try:
            # Using pytorch_msssim instead of piqa to match actual implementation
            pytorch_msssim.MS_SSIM()(
                torch.rand(1, 3, start, start), torch.rand(1, 3, start, start)
            )
            print(start)
            return start
        except RuntimeError:
            print(start)
            # Recursively try the next dimension
            return findvaliddim(start + 1)


    # Start testing from dimension 1
    findvaliddim(1)  # result is 162
