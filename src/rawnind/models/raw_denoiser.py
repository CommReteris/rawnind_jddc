# -*- coding: utf-8 -*-
"""
Raw image denoising neural network implementations.

This module contains various neural network architectures for denoising RAW
and RGB images. The primary implementation is based on U-Net with transposed
convolutions (ensuring consistent shape) and concatenations rather than additions
for skip connections.

The module supports both Bayer pattern (4-channel) and RGB (3-channel) inputs,
with appropriate output mechanisms for each case.
"""

import torch
from torch import nn

#
from rawnind.dependencies import raw_processing as rawproc


class Denoiser(nn.Module):
    """Base class for all image denoising models.

    This abstract base class defines the common interface for all denoising models
    in the module. It ensures that models can work with either RGB (3-channel) or
    Bayer pattern (4-channel) inputs.

    All denoiser implementations should inherit from this class and implement
    the forward method.
    """

    def __init__(self, in_channels: int):
        """Initialize a denoiser model.

        Args:
            in_channels: Number of input channels (must be 3 for RGB or 4 for Bayer)

        Raises:
            AssertionError: If in_channels is not 3 or 4
        """
        super().__init__()
        assert in_channels == 3 or in_channels == 4, f"{in_channels=} should be 3 or 4"


class Passthrough(Denoiser):
    """Identity model that optionally converts Bayer to RGB.

    This model either passes through RGB images unchanged or performs
    demosaicing on Bayer pattern inputs. It's useful as a baseline or for
    testing the debayering pipeline.

    For 3-channel inputs, it acts as a simple identity function.
    For 4-channel inputs, it applies demosaicing to convert to RGB.
    """

    def __init__(self, in_channels: int, **kwargs):
        """Initialize a Passthrough model.

        Args:
            in_channels: Number of input channels (3 for RGB, 4 for Bayer)
            **kwargs: Additional arguments (ignored, for compatibility with other models)
        """
        super().__init__(in_channels=in_channels)
        self.in_channels = in_channels
        # Dummy parameter to ensure the model has at least one parameter
        # (needed for compatibility with PyTorch optimizers)
        self.dummy_parameter = torch.nn.Parameter(torch.randn(3))
        if kwargs:
            print(f"Passthrough: ignoring unexpected kwargs: {kwargs}")

    def forward(self, batch: torch.Tensor):
        """Process input images without denoising.

        For RGB inputs, returns the input unchanged.
        For Bayer inputs, applies demosaicing to convert to RGB.

        Args:
            batch: Input tensor with shape [batch_size, channels, height, width]
                  where channels is either 3 (RGB) or 4 (Bayer)

        Returns:
            RGB tensor with shape [batch_size, 3, height, width]
        """
        if self.in_channels == 3:
            return batch
        debayered_batch: torch.Tensor = rawproc.demosaic(batch)
        debayered_batch.requires_grad_()
        return debayered_batch


def get_activation_class_params(activation: str) -> tuple:
    """Get the PyTorch activation class and parameters for a given activation name.

    This utility function maps activation function names to their corresponding
    PyTorch implementation classes and parameter dictionaries. It's used to
    configure activation functions throughout the network architecture.

    Args:
        activation: Name of the activation function to use.
                   Supported values: "PReLU", "ELU", "Hardswish", "LeakyReLU"

    Returns:
        Tuple of (activation_class, parameter_dict) where:
        - activation_class is a PyTorch nn.Module class for the activation
        - parameter_dict contains configuration parameters for the activation

    Raises:
        SystemExit: If an unsupported activation name is provided
    """
    if activation == "PReLU":
        return nn.PReLU, {}
    elif activation == "ELU":
        return nn.ELU, {"inplace": True}
    elif activation == "Hardswish":
        return nn.Hardswish, {"inplace": True}
    elif activation == "LeakyReLU":
        return nn.LeakyReLU, {"inplace": True, "negative_slope": 0.2}
        # negative_slope from https://github.com/lavi135246/pytorch-Learning-to-See-in-the-Dark/blob/master/model.py
    else:
        exit(f"get_activation_class: unknown activation function: {activation}")


class UtNet2(Denoiser):
    """U-Net architecture for image denoising with transposed convolutions.

    This implements a U-Net model optimized for image denoising, featuring:
    - Encoder path with 4 levels of downsampling (via max pooling)
    - Bottleneck layer at the lowest resolution
    - Decoder path with 4 levels of upsampling (via transposed convolutions)
    - Skip connections that concatenate encoder features with decoder features
    - Consistent spatial dimensions through proper padding
    - Configurable capacity through the funit parameter
    - Support for both RGB (3-channel) and Bayer pattern (4-channel) inputs

    The model can also optionally pre-upsample Bayer inputs before processing.
    When using 4-channel Bayer input, the output uses PixelShuffle to convert
    to 3-channel RGB while doubling the spatial resolution.

    Architecture:
                  Skip Connections
                 ↙     ↙     ↙     ↙
    Input → Enc1 → Enc2 → Enc3 → Enc4 → Bottom
              ↓     ↓     ↓     ↓
    Output ← Dec4 ← Dec3 ← Dec2 ← Dec1

    Where Enc = Convolution blocks with downsampling
          Dec = Convolution blocks with upsampling
    """

    def __init__(
            self,
            in_channels: int,
            funit: int = 32,
            activation: str = "LeakyReLU",
            preupsample: bool = False,
            **kwargs,
    ):
        """Initialize a U-Net model for image denoising.

        Args:
            in_channels: Number of input channels (3 for RGB, 4 for Bayer)
            funit: Base feature unit multiplier that determines network capacity
                   (higher values = more parameters and capacity)
            activation: Activation function to use throughout the network
            preupsample: If True, upsample the input before processing
                        (only valid with 4-channel input)

        Raises:
            AssertionError: If trying to use preupsample with 3-channel input
            NotImplementedError: If in_channels is not 3 or 4
        """
        super().__init__(in_channels=in_channels)
        assert (in_channels == 3 and not preupsample) or in_channels == 4
        activation_fun, activation_params = get_activation_class_params(activation)

        # Optional upsampling of input (for 4-channel Bayer only)
        if preupsample:
            self.preprocess = torch.nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=False
            )
        else:
            self.preprocess = torch.nn.Identity()

        # Encoder path - level 1 (highest resolution)
        self.convs1 = nn.Sequential(
            nn.Conv2d(in_channels, funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(funit, funit, 3, padding=1),
            activation_fun(**activation_params),
        )
        self.maxpool = nn.MaxPool2d(2)

        # Encoder path - level 2
        self.convs2 = nn.Sequential(
            nn.Conv2d(funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Encoder path - level 3
        self.convs3 = nn.Sequential(
            nn.Conv2d(2 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Encoder path - level 4 (lowest resolution before bottleneck)
        self.convs4 = nn.Sequential(
            nn.Conv2d(4 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Bottleneck at lowest resolution
        self.bottom = nn.Sequential(
            nn.Conv2d(8 * funit, 16 * funit, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(16 * funit, 16 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Decoder path - level 1 (lowest resolution after bottleneck)
        self.up1 = nn.ConvTranspose2d(16 * funit, 8 * funit, 2, stride=2)
        self.tconvs1 = nn.Sequential(
            nn.Conv2d(16 * funit, 8 * funit, 3, padding=1),  # 16 = 8 (from up1) + 8 (from skip)
            activation_fun(**activation_params),
            nn.Conv2d(8 * funit, 8 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Decoder path - level 2
        self.up2 = nn.ConvTranspose2d(8 * funit, 4 * funit, 2, stride=2)
        self.tconvs2 = nn.Sequential(
            nn.Conv2d(8 * funit, 4 * funit, 3, padding=1),  # 8 = 4 (from up2) + 4 (from skip)
            activation_fun(**activation_params),
            nn.Conv2d(4 * funit, 4 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Decoder path - level 3
        self.up3 = nn.ConvTranspose2d(4 * funit, 2 * funit, 2, stride=2)
        self.tconvs3 = nn.Sequential(
            nn.Conv2d(4 * funit, 2 * funit, 3, padding=1),  # 4 = 2 (from up3) + 2 (from skip)
            activation_fun(**activation_params),
            nn.Conv2d(2 * funit, 2 * funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Decoder path - level 4 (highest resolution)
        self.up4 = nn.ConvTranspose2d(2 * funit, funit, 2, stride=2)
        self.tconvs4 = nn.Sequential(
            nn.Conv2d(2 * funit, funit, 3, padding=1),  # 2 = 1 (from up4) + 1 (from skip)
            activation_fun(**activation_params),
            nn.Conv2d(funit, funit, 3, padding=1),
            activation_fun(**activation_params),
        )

        # Output layer - depends on input type
        if in_channels == 3 or preupsample:
            # For RGB input, direct mapping to RGB output (same resolution)
            self.output_module = nn.Sequential(nn.Conv2d(funit, 3, 1))
        elif in_channels == 4:
            # For Bayer input, map to RGB while doubling resolution with PixelShuffle
            self.output_module = nn.Sequential(
                nn.Conv2d(funit, 4 * 3, 1), nn.PixelShuffle(2)
            )
        else:
            raise NotImplementedError(f"{in_channels=}")

        # Initialize weights for better convergence
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, l):
        """Process input through U-Net for denoising.

        Args:
            l: Input tensor with shape [batch_size, channels, height, width]
               where channels is either 3 (RGB) or 4 (Bayer)

        Returns:
            Denoised RGB image tensor with shape [batch_size, 3, height, width]
            or [batch_size, 3, height*2, width*2] if using Bayer input with PixelShuffle
        """
        # Preprocessing (identity or upsampling)
        l1 = self.preprocess(l)

        # Encoder path with skip connection storage
        l1 = self.convs1(l1)  # Level 1 features (stored for skip connection)
        l2 = self.convs2(self.maxpool(l1))  # Level 2 features
        l3 = self.convs3(self.maxpool(l2))  # Level 3 features
        l4 = self.convs4(self.maxpool(l3))  # Level 4 features

        # Bottleneck and decoder path with skip connections
        l = torch.cat([self.up1(self.bottom(self.maxpool(l4))), l4], dim=1)  # Skip connection 1
        l = torch.cat([self.up2(self.tconvs1(l)), l3], dim=1)  # Skip connection 2
        l = torch.cat([self.up3(self.tconvs2(l)), l2], dim=1)  # Skip connection 3
        l = torch.cat([self.up4(self.tconvs3(l)), l1], dim=1)  # Skip connection 4

        # Final convolutions and output
        l = self.tconvs4(l)
        return self.output_module(l)


class ResBlock(torch.nn.Module):
    """Residual block with two convolutional layers and skip connection.

    This implements a standard residual block that adds the input to the output
    of a series of convolutions, allowing better gradient flow and feature reuse.
    The block maintains the same number of channels and spatial dimensions.

    Structure:
        Input → Conv2d → Activation → Conv2d → Activation → + → Output
                                                            ↑
                                                          Input
    """

    def __init__(
            self,
            num_channels: int,
            activation="LeakyReLU",
    ):
        """Initialize a residual block.

        Args:
            num_channels: Number of input and output channels
            activation: Activation function name to use (default: "LeakyReLU")
        """
        super().__init__()
        activation_fun, activation_params = get_activation_class_params(activation)
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            activation_fun(**activation_params),
            nn.Conv2d(num_channels, num_channels, 3, padding=1),
            activation_fun(**activation_params),
        )

    def forward(self, x):
        """Apply residual block to input tensor.

        Args:
            x: Input tensor of shape [batch_size, num_channels, height, width]

        Returns:
            Output tensor with same shape as input, with residual connection applied
        """
        return self.conv(x) + x


class UtNet3(UtNet2):
    """Enhanced U-Net model with additional ResBlocks in the output stage.

    This model extends UtNet2 by adding a more complex output module with:
    1. A channel expansion layer (1x1 convolution)
    2. Two ResBlocks for additional feature refinement
    3. A channel reduction layer (1x1 convolution)

    This additional processing can help with difficult denoising cases by
    allowing more complex feature transformations at the output stage.
    Currently only supports Bayer pattern inputs (4 channels).
    """

    def __init__(self, in_channels: int = 4, funit: int = 32, activation="LeakyReLU", preupsample: bool = False, **kwargs):
        """Initialize an enhanced U-Net model with ResBlocks in output stage.

        Args:
            in_channels: Number of input channels (must be 4 for Bayer pattern)
            funit: Base feature unit multiplier that determines network capacity
            activation: Activation function to use throughout the network

        Raises:
            AssertionError: If in_channels is not 4
        """
        super().__init__(in_channels=in_channels, funit=funit, activation=activation, preupsample=preupsample, **kwargs)
        assert in_channels == 4
        # Replace the output module with an enhanced version that includes ResBlocks
        self.output_module = nn.Sequential(
            torch.nn.Conv2d(funit, funit * 8, 1),  # Channel expansion
            ResBlock(
                funit * 8,
            ),
            ResBlock(funit * 8),  # Additional feature refinement
            torch.nn.Conv2d(funit * 8, funit, 1),  # Channel reduction
            self.output_module,  # Original output processing
        )


architectures = {
    "UtNet2": UtNet2,
    "UtNet3": UtNet3,
    "Passthrough": Passthrough,
    "unet": UtNet2,
    "utnet3": UtNet3,
    "identity": Passthrough,
    "autoencoder": UtNet2,  # Alias for testing compatibility
}

if __name__ == "__main__":
    utnet3 = UtNet3(in_channels=4)
    rawtensor = torch.rand(1, 4, 16, 16)
    output = utnet3(rawtensor)
    print(f"{rawtensor.shape=}, {output.shape=}")

    rawnet = UtNet2(in_channels=4)
    rgbnet = UtNet2(in_channels=3)
    rgbtensor = torch.rand(1, 3, 16, 16)

    print(f"{rawtensor.shape=}, {rawnet(rawtensor).shape=}")
    print(f"{rgbtensor.shape=}, {rgbnet(rgbtensor).shape=}")
