"""
Neural network architectures for image compression.

This module implements various neural network architectures for learned image compression, 
primarily based on the Ballé et al. approach to transform coding with autoencoders.
The models support both regular RGB (3-channel) and Bayer pattern (4-channel) images.

Key components:
- AbstractRawImageCompressor: Base class defining the compression model interface
- BalleEncoder: Encoder architecture from Ballé et al. with GDN activation
- BalleDecoder: Decoder architecture from Ballé et al. with inverse GDN
- BayerPSDecoder: Specialized decoder for Bayer images using PixelShuffle upsampling
- BayerTCDecoder: Specialized decoder for Bayer images using transposed convolutions

The implementation is modular, allowing different encoder and decoder architectures
to be combined flexibly based on the input type and desired output characteristics.

References:
    Ballé, J., Laparra, V., & Simoncelli, E. P. (2016). End-to-end optimized image compression.
    arXiv preprint arXiv:1611.01704.
"""

# -*- coding: utf-8 -*-

import math
import time
import logging
import sys
import statistics
from typing import Optional, Type, Literal
from typing_extensions import Self
import torch
from torch import nn
from torch.nn import functional as F

import ptflops
import numpy as np

sys.path.append("..")
from common.extlibs import gdn

# logger = logging.getLogger("ImageCompression")


class AbstractRawImageCompressor(nn.Module):
    """
    Abstract base class for neural image compression models.
    
    This class defines the interface for all compression models in the module,
    providing a framework for encoder-decoder architectures. It handles the 
    instantiation of encoder and decoder components and defines the expected
    input/output format.
    
    The compression pipeline typically consists of:
    1. Encoding an input image to a latent representation
    2. Quantizing the latent representation (simulated during training)
    3. Entropy coding the quantized representation (simulated during training)
    4. Decoding the latent representation back to an image
    
    Derived classes must implement the forward method to complete this pipeline.
    """
    def __init__(
        self,
        device: torch.device,
        in_channels: int,
        hidden_out_channels: Optional[int] = None,
        bitstream_out_channels: Optional[int] = None,
        encoder_cls: Optional[Type[nn.Module]] = None,
        decoder_cls: Optional[Type[nn.Module]] = None,
        preupsample=False,
    ):
        """
        Initialize the compression model with encoder and decoder components.
        
        Args:
            device: Device to place the model on (CPU or CUDA)
            in_channels: Number of input channels (3 for RGB, 4 for Bayer)
            hidden_out_channels: Number of hidden channels in encoder/decoder
            bitstream_out_channels: Number of output channels in the latent space
            encoder_cls: Class for the encoder component
            decoder_cls: Class for the decoder component
            preupsample: Whether to upsample Bayer pattern inputs before encoding
        """
        super().__init__()
        self.device: torch.device = device
        self.in_channels: int = in_channels
        if encoder_cls and decoder_cls:
            self.Encoder = encoder_cls(
                hidden_out_channels=hidden_out_channels,
                bitstream_out_channels=bitstream_out_channels,
                in_channels=in_channels,
                device=device,
                preupsample=preupsample,
            )
            self.Decoder = decoder_cls(
                hidden_out_channels=hidden_out_channels,
                bitstream_out_channels=bitstream_out_channels,
                device=device,
            )

    def forward(self, input_image: torch.Tensor) -> dict:
        """
        Process an image through the compression pipeline.
        
        Takes an input image batch, encodes it to a latent representation,
        applies quantization (simulated during training), and decodes it
        back to an image. Also calculates rate and distortion metrics.
        
        Args:
            input_image: Batch of images with shape (batch_size, channels, height, width)
            
        Returns:
            Dictionary containing:
            - "reconstructed_image": Tensor of reconstructed images, shape (b, c, h, w)
            - "visual_loss": Float tensor, distortion metric (e.g., MSE, MS-SSIM)
            - "bpp": Float tensor, total bits per pixel of the compressed representation
            - "bpp_feature": (optional) Float tensor, bpp of the main features
            - "bpp_sidestring": (optional) Float tensor, bpp of the side information
        """
        pass

    def cpu(self) -> Self:
        """
        Move the model to CPU device.
        
        Returns:
            Self reference for method chaining
        """
        self.device = torch.device("cpu")
        return self.to(self.device)

    def todev(self, device: torch.device) -> Self:
        """
        Move the model to the specified device.
        
        Args:
            device: Target device (CPU or CUDA)
            
        Returns:
            Self reference for method chaining
        """
        self.device = device
        return self.to(self.device)


class BalleEncoder(nn.Module):
    """
    Encoder network based on Ballé et al. architecture for image compression.
    
    This encoder takes an image (RGB or Bayer pattern) and transforms it into
    a latent representation suitable for entropy coding. The architecture consists
    of multiple strided convolutional layers with Generalized Divisive Normalization
    (GDN) activation, progressively reducing spatial dimensions while increasing
    the feature channel count.
    
    Features:
    - Support for both RGB (3-channel) and Bayer pattern (4-channel) inputs
    - Optional pre-upsampling for Bayer pattern inputs
    - 4 levels of downsampling (16x spatial reduction)
    - GDN activations for better statistical decorrelation
    - Carefully initialized weights for stable training
    
    The network reduces spatial dimensions by a factor of 16 (2^4) while 
    transforming input channels to bitstream_out_channels features.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
        in_channels: Literal[3, 4] = 3,
        preupsample: bool = False,
    ):
        """
        Initialize the encoder network.
        
        Args:
            device: Device to place the model on (CPU or CUDA)
            hidden_out_channels: Number of channels in the hidden layers (default: 192)
            bitstream_out_channels: Number of output channels in the latent space (default: 320)
            in_channels: Number of input channels (3 for RGB, 4 for Bayer)
            preupsample: Whether to upsample Bayer pattern inputs before encoding
                         (only applicable when in_channels=4)
        
        Raises:
            AssertionError: If trying to use preupsample with RGB input (in_channels=3)
        """
        super().__init__()
        assert (in_channels == 3 and not preupsample) or in_channels == 4
        self.gdn1 = gdn.GDN(hidden_out_channels, device=device)
        self.gdn2 = gdn.GDN(hidden_out_channels, device=device)
        self.gdn3 = gdn.GDN(hidden_out_channels, device=device)
        if preupsample:
            self.preprocess = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            self.preprocess = torch.nn.Identity()

        self.conv1 = nn.Conv2d(in_channels, hidden_out_channels, 5, stride=2, padding=2)
        torch.nn.init.xavier_normal_(
            self.conv1.weight.data,
            (math.sqrt(2 * (in_channels + hidden_out_channels) / (6))),
        )
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)

        self.conv2 = nn.Conv2d(
            hidden_out_channels, hidden_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)

        self.conv3 = nn.Conv2d(
            hidden_out_channels, hidden_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

        self.conv4 = nn.Conv2d(
            hidden_out_channels, bitstream_out_channels, 5, stride=2, padding=2
        )
        torch.nn.init.xavier_normal_(
            self.conv4.weight.data,
            (
                math.sqrt(
                    2
                    * (bitstream_out_channels + hidden_out_channels)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.conv4.bias.data, 0.01)

    def forward(self, x):
        """
        Encode input image to latent representation.
        
        Args:
            x: Input image tensor with shape [batch_size, channels, height, width]
               where channels is either 3 (RGB) or 4 (Bayer)
        
        Returns:
            Latent representation with shape 
            [batch_size, bitstream_out_channels, height/16, width/16]
        """
        x = self.preprocess(x)  # Optional upsampling for Bayer inputs
        x = self.gdn1(self.conv1(x))  # First downsampling + GDN
        x = self.gdn2(self.conv2(x))  # Second downsampling + GDN
        x = self.gdn3(self.conv3(x))  # Third downsampling + GDN
        return self.conv4(x)  # Final downsampling (no activation)


# class BayerPreUpEncoder(BalleEncoder):
#     def __init__(
#         self,
#         device: torch.device,
#         hidden_out_channels: int = 192,
#         bitstream_out_channels: int = 320,
#         in_channels: Literal[3, 4] = 4,
#     ):
#         assert in_channels == 4
#         super().__init__(
#             device, hidden_out_channels, bitstream_out_channels, in_channels
#         )
#         # bicubic upsampler
#         self.upsampler = nn.Upsample(
#             scale_factor=2, mode="bicubic", align_corners=False
#         )

#     def forward(self, x):
#         x = self.upsampler(x)
#         return super().forward(x)


class BalleDecoder(nn.Module):
    """
    Decoder network based on Ballé et al. architecture for image compression.
    
    This decoder transforms a latent representation back into an RGB image. 
    The architecture consists of multiple transposed convolutional layers with 
    inverse Generalized Divisive Normalization (GDN) activation, progressively 
    increasing spatial dimensions while decreasing the feature channel count.
    
    Features:
    - Reconstructs RGB (3-channel) images from latent representations
    - 4 levels of upsampling (16x spatial expansion)
    - Inverse GDN activations to match the encoder's normalization
    - Carefully initialized weights for stable training
    - Symmetric structure to the BalleEncoder
    
    The network increases spatial dimensions by a factor of 16 (2^4) while
    transforming bitstream_out_channels features to 3 RGB channels.
    """

    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        """
        Initialize the decoder network.
        
        Args:
            device: Device to place the model on (CPU or CUDA)
            hidden_out_channels: Number of channels in the hidden layers (default: 192)
            bitstream_out_channels: Number of input channels from the latent space (default: 320)
        """
        super().__init__()
        self.igdn1 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)
        self.igdn2 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)
        self.igdn3 = gdn.GDN(ch=hidden_out_channels, inverse=True, device=device)

        self.deconv1 = nn.ConvTranspose2d(
            bitstream_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            self.deconv1.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (bitstream_out_channels + hidden_out_channels)
                    / (bitstream_out_channels + bitstream_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        # self.igdn1 = GDN.GDN(hidden_out_channels, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        # self.igdn2 = GDN.GDN(hidden_out_channels, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        # self.igdn3 = GDN.GDN(hidden_out_channels, inverse=True)

        self.output_module = nn.ConvTranspose2d(
            hidden_out_channels,
            3,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            self.output_module.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(self.output_module.bias.data, 0.01)

    def forward(self, x):
        """
        Decode latent representation to RGB image.
        
        Args:
            x: Latent representation tensor with shape 
               [batch_size, bitstream_out_channels, height/16, width/16]
        
        Returns:
            Reconstructed RGB image with shape [batch_size, 3, height, width]
        """
        x = self.igdn1(self.deconv1(x))  # First upsampling + inverse GDN
        x = self.igdn2(self.deconv2(x))  # Second upsampling + inverse GDN
        x = self.igdn3(self.deconv3(x))  # Third upsampling + inverse GDN
        return self.output_module(x)      # Final upsampling to RGB


class BayerPSDecoder(BalleDecoder):
    """
    Specialized decoder for Bayer pattern images using PixelShuffle upsampling.
    
    This decoder extends BalleDecoder to handle Bayer pattern outputs, converting
    latent representations to RGB images with 2x the spatial resolution of the
    standard decoder output. This is designed for use with Bayer pattern inputs
    that need to maintain their higher effective resolution.
    
    The key difference from BalleDecoder is the output stage, which:
    1. Adds an additional transposed convolution (deconv4)
    2. Uses a 1x1 convolution to generate 12 channels (4 Bayer x 3 RGB)
    3. Applies PixelShuffle to rearrange these channels into a 2x upsampled RGB image
    
    This approach effectively doubles the spatial resolution compared to the
    standard BalleDecoder, making it suitable for Bayer pattern processing.
    """
    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        """
        Initialize the Bayer pattern decoder with PixelShuffle upsampling.
        
        Args:
            device: Device to place the model on (CPU or CUDA)
            hidden_out_channels: Number of channels in the hidden layers (default: 192)
            bitstream_out_channels: Number of input channels from the latent space (default: 320)
        """
        super().__init__(device, hidden_out_channels, bitstream_out_channels)
        # Additional transposed convolution for extra upsampling
        deconv4 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(deconv4.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(deconv4.bias.data, 0.01)
        
        # 1x1 convolution to generate 12 channels (4 Bayer pixels x 3 RGB channels)
        finalconv = torch.nn.Conv2d(hidden_out_channels, 4 * 3, 1)
        torch.nn.init.xavier_normal_(
            finalconv.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 4 * 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(finalconv.bias.data, 0.01)
        
        # Replace output_module with sequence ending in PixelShuffle for 2x upsampling
        self.output_module = nn.Sequential(deconv4, finalconv, nn.PixelShuffle(2))
        
    # Uses the parent class forward method, which processes through the custom output_module


class BayerTCDecoder(BalleDecoder):
    """
    Specialized decoder for Bayer pattern images using transposed convolutions.
    
    This decoder extends BalleDecoder to handle Bayer pattern outputs, converting
    latent representations to RGB images with 2x the spatial resolution of the
    standard decoder output. Unlike BayerPSDecoder which uses PixelShuffle, this
    decoder uses an additional transposed convolution for the final upsampling.
    
    The key difference from BalleDecoder is the output stage, which:
    1. Adds an additional transposed convolution (deconv4)
    2. Includes a LeakyReLU activation
    3. Uses a final transposed convolution to generate RGB output at 2x resolution
    
    This approach provides an alternative to PixelShuffle while still achieving
    the 2x spatial resolution needed for Bayer pattern processing.
    """
    def __init__(
        self,
        device: torch.device,
        hidden_out_channels: int = 192,
        bitstream_out_channels: int = 320,
    ):
        """
        Initialize the Bayer pattern decoder with transposed convolution upsampling.
        
        Args:
            device: Device to place the model on (CPU or CUDA)
            hidden_out_channels: Number of channels in the hidden layers (default: 192)
            bitstream_out_channels: Number of input channels from the latent space (default: 320)
        """
        super().__init__(device, hidden_out_channels, bitstream_out_channels)
        # Additional transposed convolution for extra upsampling
        deconv4 = nn.ConvTranspose2d(
            hidden_out_channels,
            hidden_out_channels,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(deconv4.weight.data, (math.sqrt(2 * 1)))
        torch.nn.init.constant_(deconv4.bias.data, 0.01)
        
        # Additional activation after extra convolution
        final_act = nn.LeakyReLU()
        
        # Final transposed convolution to generate RGB at 2x resolution
        finalconv = nn.ConvTranspose2d(
            hidden_out_channels,
            3,
            5,
            stride=2,
            padding=2,
            output_padding=1,
        )
        torch.nn.init.xavier_normal_(
            finalconv.weight.data,
            (
                math.sqrt(
                    2
                    * 1
                    * (hidden_out_channels + 3)
                    / (hidden_out_channels + hidden_out_channels)
                )
            ),
        )
        torch.nn.init.constant_(finalconv.bias.data, 0.01)

        # Replace output_module with sequence of additional layers
        self.output_module = nn.Sequential(deconv4, final_act, finalconv)
        
    # Uses the parent class forward method, which processes through the custom output_module


# class RawImageEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         raise NotImplementedError
#         # self.compressor = compressor

#     # def forward(self, input_image):
#     # return self.compressor.encode(input_image, entropy_coding=False)


# class RawImageDecoder(nn.Module):
#     def __init__(self, compressor):
#         super().__init__()
#         raise NotImplementedError
#         # self.compressor = compressor

#     # def forward(self, input_image):
#     # return self.compressor.decode(input_image)
