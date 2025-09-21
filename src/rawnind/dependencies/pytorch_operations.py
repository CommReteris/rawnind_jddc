# -*- coding: utf-8 -*-
"""
PyTorch tensor operations for image processing and deep learning.

This module provides utilities for manipulating PyTorch tensors, with a focus on
operations needed for image processing and neural network training. The primary
functions handle conversions between full images and patches/batches, which is
crucial for efficient processing in convolutional neural networks.

Key features:
- Image-to-batch and batch-to-image conversions for patch-based processing
- Tensor reshaping operations like pixel unshuffle for resolution changes
- Custom autograd functions for maintaining gradients with non-differentiable ops
- Utility functions for tensor cropping and validation
- Testing utilities for tensor checksums and placeholder losses

These operations are particularly useful for:
1. Training neural networks on image patches
2. Implementing custom layer architectures
3. Pre-processing tensors to conform to model requirements
4. Implementing spatial transformations on feature maps

Most functions support both CPU and CUDA tensors, and strive to maintain
differentiability where possible for end-to-end training.
"""

import unittest
import math
import torch
from torch import nn

# def img_to_batch(imgtensor, patch_size: int):
#     _, ch, height, width = imgtensor.shape
#     assert height%patch_size == 0 and width%patch_size == 0, 'img_to_batch: dims must be dividable by patch_size. {}%{}!=0'.format(imgtensor.shape, patch_size)
#     bs = math.ceil(height/patch_size) * math.ceil(width/patch_size)
#     btensor = torch.zeros([bs,ch,patch_size, patch_size], device=imgtensor.device, dtype=imgtensor.dtype)
#     xstart = ystart = 0
#     for i in range(bs):
#         btensor[i] = imgtensor[:, :, ystart:ystart+patch_size, xstart:xstart+patch_size]
#         xstart += patch_size
#         if xstart+patch_size > width:
#             xstart = 0
#             ystart += patch_size
#     return btensor


def pt_crop_batch(batch, cs: int):
    """
    center crop an image batch to cs
    also compatible with numpy tensors
    """
    x0 = (batch.shape[3] - cs) // 2
    y0 = (batch.shape[2] - cs) // 2
    return batch[:, :, y0 : y0 + cs, x0 : x0 + cs]


def img_to_batch(img, patch_size: int, nchans_per_prior: int = None):
    """
    Convert a full image tensor into a batch of image patches.
    
    This function takes a batch of images and divides each image into non-overlapping
    patches of size (patch_size × patch_size), rearranging them into a batch of
    individual patch tensors. This is useful for processing images in patches,
    which is common in neural network architectures like autoencoders.
    
    The function handles cases where image dimensions are not divisible by patch_size
    by truncating the image to the largest size that is divisible by patch_size.
    
    Args:
        img: Input tensor with shape [batch_size, channels, height, width]
        patch_size: Size of each square patch (both height and width)
        nchans_per_prior: Number of channels per prior distribution in entropy models.
                          If None, uses the same number as input channels.
    
    Returns:
        torch.Tensor: Batch of image patches with shape 
                      [num_patches, nchans_per_prior, patch_size, patch_size]
                      where num_patches = (batch_size * height * width) / (patch_size^2)
    
    Note:
        This implementation uses PyTorch's unfold operation followed by reshape and 
        transpose operations to efficiently convert images to patches while maintaining
        differentiability for backpropagation.
    """
    # Extract dimensions from input tensor
    _, ch, height, width = img.shape
    
    # If nchans_per_prior not specified, use input channel count
    if nchans_per_prior is None:
        nchans_per_prior = ch
        
    # Verify input is a 4D tensor (batch, channels, height, width)
    assert img.ndim == 4, "Input must be a 4D tensor with shape [batch, channels, height, width]"
    
    # Handle case where image dimensions aren't divisible by patch_size
    if not (height % patch_size == 0 and width % patch_size == 0):
        print(
            "img_to_batch: warning: "
            "img_to_batch: dims must be dividable by patch_size. {}%{}!=0".format(
                img.shape, patch_size
            )
        )
        # Truncate image to largest size divisible by patch_size
        _, _, imgy, imgx = img.shape
        img = img[
            :,
            :,
            : (imgy // patch_size) * patch_size,
            : (imgx // patch_size) * patch_size,
        ]

    # Convert image to patches using a series of unfold, transpose, and reshape operations
    # 1. unfold creates sliding windows over height and width dimensions
    # 2. transpose and reshape operations rearrange the tensor into the desired batch format
    return (
        img.unfold(2, patch_size, patch_size)  # Create patches along height (dim 2)
        .unfold(3, patch_size, patch_size)     # Create patches along width (dim 3)
        .transpose(1, 0)                       # Swap batch and channel dimensions
        .reshape(ch, -1, patch_size, patch_size)  # Reshape to [ch, num_patches, patch_size, patch_size]
        .transpose(1, 0)                       # Swap channels and num_patches
        .reshape(-1, nchans_per_prior, patch_size, patch_size)  # Final shape with nchans_per_prior
    )
    # Note: Previous incorrect implementation is commented out below
    # return img.unfold(2,patch_size,patch_size).unfold(
    #     3,patch_size,patch_size).contiguous().view(
    #         ch,-1,patch_size,patch_size).permute((1,0,2,3))


def batch_to_img(btensor, height: int, width: int, ch=3):
    """
    Reconstruct a full image tensor from a batch of image patches.
    
    This function is the inverse operation of img_to_batch(). It takes a batch of
    image patches and reconstructs a full image by placing each patch at its
    corresponding position in the output tensor. The patches are assumed to be
    non-overlapping and arranged in row-major order (left-to-right, top-to-bottom).
    
    Note:
        This implementation is not differentiable because it uses explicit indexing
        rather than tensor operations. If differentiability is required, consider
        implementing a version using fold/unfold operations.
    
    Args:
        btensor: Batch of image patches with shape [num_patches, channels, patch_size, patch_size]
        height: Height of the output image
        width: Width of the output image
        ch: Number of channels in the output image (default: 3)
        
    Returns:
        torch.Tensor: Reconstructed image tensor with shape [1, channels, height, width]
        
    Warning:
        The dimensions of the output image (height, width) must be compatible with
        the number and size of patches. Specifically, height and width should be
        divisible by patch_size, and the total number of patches should equal
        (height * width) / (patch_size^2).
    """
    # Create empty output tensor with specified dimensions
    imgtensor = torch.zeros(
        [1, ch, height, width], device=btensor.device, dtype=btensor.dtype
    )
    
    # Extract patch size from input tensor
    patch_size = btensor.shape[-1]
    
    # Initialize starting position for placing patches
    xstart = ystart = 0
    
    # Place each patch at its corresponding position in the output tensor
    for i in range(btensor.size(0)):
        # Copy patch to its position in the output tensor
        imgtensor[0, :, ystart : ystart + patch_size, xstart : xstart + patch_size] = (
            btensor[i]
        )
        
        # Move to the next position (left to right, then top to bottom)
        xstart += patch_size
        if xstart + patch_size > width:
            # When we reach the end of a row, move to the start of the next row
            xstart = 0
            ystart += patch_size
            
    return imgtensor


def pixel_unshuffle(input, downscale_factor):
    """
    Rearrange elements from spatial dimensions to channel dimension (inverse PixelShuffle).
    
    This function performs the inverse operation of torch.nn.PixelShuffle, reducing
    spatial dimensions (height and width) by a factor of downscale_factor while
    increasing the number of channels by downscale_factor^2. It's useful for:
    - Reducing spatial resolution while preserving information
    - Converting spatial features to channel features (space-to-depth)
    - Creating inputs for certain neural network architectures
    
    Args:
        input: Tensor with shape [batch_size, channels, height, width]
        downscale_factor: Factor by which to reduce spatial dimensions and increase channels
        
    Returns:
        torch.Tensor: Output tensor with shape [batch_size, channels*downscale_factor^2, 
                      height/downscale_factor, width/downscale_factor]
    
    Example:
        >>> input = torch.randn(1, 3, 8, 8)  # 1 batch, 3 channels, 8x8 spatial dims
        >>> output = pixel_unshuffle(input, 2)
        >>> output.shape
        torch.Size([1, 12, 4, 4])  # 1 batch, 12 channels, 4x4 spatial dims
    
    Note:
        For the inverse operation (pixel_shuffle), the input should have 
        channels = output_channels * scale_factor^2
    
    Source:
        Adapted from https://github.com/SsnL/pytorch/blob/c0a9167d2397f9064336bbb7ac73e0ed9ed44d78/torch/nn/functional.py
    """
    # Extract dimensions from input tensor
    batch_size, channels, in_height, in_width = input.size()
    
    # Calculate output dimensions
    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor
    
    # Reshape input to separate the spatial dimensions into blocks
    input_view = input.reshape(
        batch_size, channels, out_height, downscale_factor, out_width, downscale_factor
    )
    
    # Calculate new number of channels after unshuffle
    channels *= downscale_factor**2
    
    # Permute dimensions to move spatial blocks to channel dimension
    # From [batch, channels, out_height, downscale_factor, out_width, downscale_factor]
    # To [batch, channels, downscale_factor, downscale_factor, out_height, out_width]
    unshuffle_out = input_view.permute(0, 1, 3, 5, 2, 4)
    
    # Reshape to final output shape [batch, channels*downscale_factor^2, out_height, out_width]
    return unshuffle_out.reshape(batch_size, channels, out_height, out_width)


def oneloss(x, y):
    """
    Constant loss function that always returns 1.0 on the same device as the input.
    
    This function ignores its inputs and returns a scalar tensor containing 1.0.
    It's useful in several scenarios:
    - As a placeholder loss function during debugging
    - For testing model infrastructure without actual loss computation
    - As a baseline comparison for other loss functions
    - When you need to skip optimization for certain components
    
    Args:
        x: First input tensor (ignored except for device information)
        y: Second input tensor (completely ignored)
        
    Returns:
        torch.Tensor: A scalar tensor containing 1.0, placed on the same device as x
        
    Note:
        While this function takes two inputs like a typical loss function,
        it doesn't compute any actual difference between them.
    """
    return torch.ones(1).to(x.device)


def fragile_checksum(atensor):
    """
    Compute a simple statistical checksum for a tensor.
    
    This function calculates three statistical measures (mean, standard deviation, 
    and sum) to create a basic fingerprint of a tensor's content. It's called 
    "fragile" because even small changes to the tensor will alter the result, 
    which is useful for detecting any modifications during debugging.
    
    The checksum is primarily useful for:
    - Verifying tensor content during debugging
    - Checking whether tensors have changed between processing steps
    - Creating simple, human-readable tensor signatures
    - Unit testing to ensure operations maintain tensor values
    
    Args:
        atensor: Input tensor of any shape and dtype
        
    Returns:
        tuple: A tuple containing (mean, std, sum) of the tensor values,
               all converted to Python float values
               
    Note:
        This is not a cryptographic checksum and should not be used for security
        purposes. It's designed for debugging and validation where the goal is
        to detect any changes to tensor values.
    """
    return (
        torch.mean(atensor.float()).item(),  # Mean value (converted to Python float)
        atensor.float().std().item(),        # Standard deviation
        torch.sum(atensor).item(),           # Sum of all elements
    )


class RoundNoGradient(torch.autograd.Function):
    """
    Custom autograd function for rounding with straight-through gradient estimation.
    
    This class implements a differentiable version of the rounding operation by using
    a straight-through estimator (STE) for the gradient. During the forward pass,
    it performs standard rounding, but during the backward pass, it passes gradients
    through unchanged as if the operation were an identity function.
    
    This is particularly useful in neural networks that need rounding operations
    (such as quantization-aware training or certain compression algorithms) while
    still maintaining end-to-end differentiability for training.
    
    Usage:
        rounded_x = RoundNoGradient.apply(x)
        
    Note:
        The straight-through estimator is an approximation that ignores the true
        gradient of the rounding operation (which would be zero almost everywhere).
        This enables gradient-based optimization to work with non-differentiable
        operations, but the gradient is biased.
    """
    @staticmethod
    def forward(ctx, x):
        """
        Apply rounding to the input tensor.
        
        Args:
            ctx: Context object for storing information for backward pass
            x: Input tensor to round
            
        Returns:
            torch.Tensor: Tensor with rounded values
        """
        return x.round()

    @staticmethod
    def backward(ctx, g):
        """
        Pass gradients through unchanged (straight-through estimator).
        
        Args:
            ctx: Context object (not used in this implementation)
            g: Gradient tensor from subsequent operations
            
        Returns:
            torch.Tensor: Unmodified gradient tensor
        """
        return g  # Pass gradient through unchanged


def crop_to_multiple(tensor: torch.Tensor, multiple: int = 64) -> torch.Tensor:
    """
    Crop a tensor's spatial dimensions to ensure they are multiples of a given value.
    
    This function crops the height and width dimensions of a tensor (assumed to be
    the last two dimensions) to make them divisible by the specified multiple.
    This is often required for neural networks that:
    - Use pooling/upsampling layers that require specific dimension constraints
    - Implement architectures like U-Net with skip connections
    - Process image patches or employ convolutional operations with stride
    
    Args:
        tensor: Input tensor of any shape where the last two dimensions
               represent height and width
        multiple: Value that the spatial dimensions should be a multiple of
                 (default: 64)
                 
    Returns:
        torch.Tensor: Cropped tensor with spatial dimensions that are
                     multiples of the specified value
                     
    Example:
        >>> x = torch.randn(1, 3, 513, 257)  # Batch of images with non-multiple dimensions
        >>> cropped = crop_to_multiple(x, 32)
        >>> cropped.shape
        torch.Size([1, 3, 512, 256])  # Dimensions now multiples of 32
    """
    # Calculate new dimensions that are multiples of the specified value
    # by removing the remainder after division
    new_height = tensor.size(-2) - tensor.size(-2) % multiple
    new_width = tensor.size(-1) - tensor.size(-1) % multiple
    
    # Return cropped tensor using advanced indexing
    # The ellipsis (...) preserves all leading dimensions
    return tensor[
        ...,                  # Preserve batch, channels, and any other dimensions
        :new_height,          # Crop height dimension
        :new_width,           # Crop width dimension
    ]


class Test_PTOPS(unittest.TestCase):
    """
    Unit tests for PyTorch tensor operations in the pt_ops module.
    
    This test class verifies the functionality of the tensor manipulation operations
    defined in the module, ensuring they correctly transform tensors and maintain
    their content when applicable. The tests cover operations like:
    - PixelShuffle and pixel_unshuffle dimension transformations
    - Conversion between full images and image patches
    - Round-trip transformations that should preserve tensor content
    """
    
    def test_pixel_shuffle_size(self):
        """
        Test the shape transformations of pre-PixelShuffle convolution.
        
        This test verifies that:
        1. The convolution layer correctly expands the channel dimension as required
           for the PixelShuffle operation
        2. The shape of the output tensor matches the expected dimensions
        
        Note:
            This test focuses on the preparation for PixelShuffle but doesn't test
            the actual PixelShuffle or pixel_unshuffle operations completely.
            The commented code shows an example of pixel_unshuffle that would fail
            due to insufficient channels.
        """
        # Create a random image tensor (batch, channels, height, width)
        dim = 256
        bs = 4
        ch = 3
        img = torch.rand(bs, ch, dim, dim)
        
        # Configure upscaling parameters
        scale_factor = 4
        
        # Create a convolution layer that expands channels for PixelShuffle
        aconv = nn.Conv2d(ch, ch * scale_factor**2, 3, padding=3 // 2)
        
        # Apply convolution to prepare for PixelShuffle
        preshuffle_img = aconv(img)
        
        # Verify output shape has expanded channels but same spatial dimensions
        self.assertListEqual(
            [bs, ch * scale_factor**2, dim, dim], list(preshuffle_img.shape)
        )
        
        # Create PixelShuffle layer (not used in assertion but shows complete usage)
        upscaled_img = nn.PixelShuffle(4)
        
        # The following code is commented out as it would fail (for documentation purposes)
        # # This would fail because ch is too small for the intended downscaling
        # aconv = nn.Conv2d(ch, int(ch*(1/scale_factor)**2), 3, padding=3//2)
        # preshuffle_img = aconv(img)
        # downscaled_img = pixel_unshuffle(preshuffle_img, scale_factor)

    def test_img_to_batch_to_img(self):
        """
        Test the round-trip conversion between image and patches.
        
        This test verifies that:
        1. img_to_batch correctly converts an image to patches with expected dimensions
        2. batch_to_img correctly reconstructs the original image from patches
        3. The round-trip operation perfectly preserves all tensor values
        
        This is an important test for ensuring that these core operations maintain
        data integrity, which is essential for neural network training.
        """
        # Create a random image tensor
        imgtensor = torch.rand(1, 3, 768, 512)  # 1 batch, 3 channels, 768x512 resolution
        
        # Convert image to batch of patches
        btensor = img_to_batch(imgtensor, 64)  # Using 64x64 patch size
        
        # Verify the shape of the patch tensor is correct
        # Expected: [96, 3, 64, 64] = [num_patches, channels, patch_height, patch_width]
        # Where num_patches = (768/64) * (512/64) = 12 * 8 = 96
        self.assertListEqual(list(btensor.shape), [96, 3, 64, 64])
        
        # Convert patches back to full image
        imgtensorback = batch_to_img(btensor, 768, 512)
        
        # Verify that the reconstructed image is identical to the original
        # (imgtensor != imgtensorback).sum() counts differences - should be 0
        self.assertEqual((imgtensor != imgtensorback).sum(), 0)


if __name__ == "__main__":
    unittest.main()
