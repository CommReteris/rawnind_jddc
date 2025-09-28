"""Sequential denoising and compression pipeline model.

This module implements a two-stage neural network model that first denoises an input image
using a pre-trained denoising model (UtNet2), and then compresses the denoised result
using a separate neural compression model (ManyPriors_RawImageCompressor).

This architecture serves multiple purposes:
1. It allows comparison with end-to-end joint denoising+compression models
2. It represents a traditional pipeline approach to image processing
3. It enables ablation studies by isolating the effect of each stage

The model supports both Bayer pattern (4-channel) and profiled RGB (3-channel) inputs
by loading the appropriate pre-trained denoising model for each input type.

Example usage:
    model = DenoiseThenCompress(
        in_channels=4,                      # For Bayer pattern input
        encoder_cls=BalleEncoder,           # Compression encoder
        decoder_cls=BalleDecoder,           # Compression decoder
        device=torch.device("cuda:0")
    )

    # Process an input image
    result = model(input_tensor)

    # Access the compressed/reconstructed image
    reconstructed = result["reconstructed_image"]

    # Access the bitrate (compression efficiency)
    bpp = result["bpp"]
"""

from typing import Literal, Optional

import torch

from rawnind.dependencies import raw_processing as rawproc
#
from .raw_denoiser import UtNet2
from .manynets_compression import ManyPriors_RawImageCompressor


# DBG
# from rawnind.libs import raw
# import os
# ENDDBG


class DenoiseThenCompress(torch.nn.Module):
    """Sequential denoising followed by compression pipeline.

    This class implements a two-stage model where an image is first denoised
    using a pre-trained denoising network and then compressed using a neural
    compression network. The pipeline represents a traditional approach where
    denoising and compression are treated as separate, sequential operations.

    The model supports both Bayer pattern (4-channel) and profiled RGB (3-channel)
    inputs by loading the appropriate pre-trained denoising model.

    Attributes:
        DENOISING_ARCH: The denoising architecture class (UtNet2)
        BAYER_DENOISING_MODEL_FPATH: Path to pre-trained Bayer pattern denoising model
        PRGB_DENOISING_MODEL_FPATH: Path to pre-trained profiled RGB denoising model
    """
    # Denoising architecture to use (U-Net based model)
    DENOISING_ARCH = UtNet2

    # Path to pre-trained Bayer pattern (4-channel) denoising model
    BAYER_DENOISING_MODEL_FPATH = "models/rawnind_denoise/DenoiserTrainingBayerToProfiledRGB_4ch_2024-02-21-bayer_ms-ssim_mgout_notrans_valeither_-4/saved_models/iter_4350000.pt"

    # Path to pre-trained profiled RGB (3-channel) denoising model
    PRGB_DENOISING_MODEL_FPATH = "models/rawnind_denoise/DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1/saved_models/iter_3900000.pt"

    def __init__(
            self,
            in_channels: Literal[3, 4],
            encoder_cls: Optional[torch.nn.Module],
            decoder_cls: Optional[torch.nn.Module],
            device: torch.device,
            hidden_out_channels: int = 192,
            bitstream_out_channels: int = 320,
            num_distributions: int = 64,
            preupsample: bool = False,
            *args,
            **kwargs,
    ):
        """Initialize the sequential denoising and compression pipeline.

        Args:
            in_channels: Number of input channels (3 for RGB, 4 for Bayer)
            encoder_cls: Encoder class for the compression model
            decoder_cls: Decoder class for the compression model
            device: Device to place the model on (CPU or CUDA)
            hidden_out_channels: Number of hidden channels in compression model
            bitstream_out_channels: Number of bitstream channels in compression model
            num_distributions: Number of entropy distributions for compression model
            preupsample: Whether to upsample before compression
            *args, **kwargs: Additional arguments (ignored)

        Raises:
            ValueError: If in_channels is not 3 or 4
        """
        super().__init__()

        # Initialize compression model (always uses 3-channel input since denoiser outputs RGB)
        self.compressor = manynets_compression.ManyPriors_RawImageCompressor(
            hidden_out_channels=hidden_out_channels,
            bitstream_out_channels=bitstream_out_channels,
            in_channels=3,  # Compressor always takes RGB input (denoiser output)
            device=device,
            # Commented parameters are not used but kept for reference
            # min_feat=min_feat,
            # max_feat=max_feat,
            # precision=precision,
            # entropy_coding=entropy_coding,
            encoder_cls=encoder_cls,
            decoder_cls=decoder_cls,
            preupsample=preupsample,
            num_distributions=num_distributions,
        )

        # Initialize denoising model with appropriate architecture
        self.denoiser = self.DENOISING_ARCH(in_channels=in_channels, funit=32)

        # Load the appropriate pre-trained denoising model based on input channels
        if in_channels == 3:
            denoiser_model_fpath = self.PRGB_DENOISING_MODEL_FPATH
        elif in_channels == 4:
            denoiser_model_fpath = self.BAYER_DENOISING_MODEL_FPATH
        else:
            raise ValueError(f"Unknown in_channels: {in_channels}")

        # Move denoiser to the specified device and load pre-trained weights
        self.denoiser = self.denoiser.to(device)
        self.denoiser.load_state_dict(
            torch.load(denoiser_model_fpath, map_location=device)
        )
        # Set denoiser to evaluation mode (no training)
        self.denoiser.eval()

    def forward(self, x: torch.Tensor):
        """Process an image through the denoising and compression pipeline.

        This method implements the forward pass through the sequential pipeline:
        1. First, denoise the input image using the pre-trained denoiser
        2. Match the gain of the denoised image to the original input
        3. Compress the denoised image using the neural compression model

        Args:
            x: Input image tensor with shape [batch_size, channels, height, width]
               where channels is either 3 (RGB) or 4 (Bayer)

        Returns:
            dict: Dictionary containing compression results with keys:
                - "reconstructed_image": Decompressed image tensor
                - "bpp": Bits per pixel (compression rate)
                - "visual_loss": Visual distortion metric
                - "bpp_feature": Feature bitrate component
                - "bpp_z": Latent bitrate component
        """
        # Commented debugging code for saving input image
        # raw.hdr_nparray_to_file(
        #     x[0].detach().cpu().numpy(),
        #     os.path.join(
        #         "dbg",
        #         "x.tif",
        #     ),
        #     color_profile="lin_rec2020",
        # )

        # Step 1: Apply denoising using the pre-trained denoiser
        x_denoised = self.denoiser(x)

        # Commented alternative normalization method
        # # scale to the original values
        # xmin = x.min()
        # xmax = x.max()
        # x_denoised = xmin + (x_denoised - x_denoised.min()) * (xmax - xmin) / (x_denoised.max() - x_denoised.min())
        # # check that xmin and x.min() are the same, and xmax and x.max() are the same
        # assert torch.allclose(xmin, x_denoised.min())
        # assert torch.allclose(xmax, x_denoised.max())

        # Step 2: Match the gain of the denoised image to the original input
        # This ensures consistent brightness/contrast between input and output
        x_denoised = rawproc.match_gain(x, x_denoised)

        # Commented debugging code for saving denoised image
        # raw.hdr_nparray_to_file(
        #     x_denoised[0].detach().cpu().numpy(),
        #     os.path.join(
        #         "dbg",
        #         "denoised_x.tif",
        #     ),
        #     color_profile="lin_rec2020",
        # )

        # Step 3: Compress the denoised image
        x_compressed = self.compressor(x_denoised)

        return x_compressed

        # Commented alternative debug return (bypasses denoising)
        # compressed = self.compressor(x)
        # compressed["reconstructed_image"] = x  # dbg
        # return compressed

    def parameters(self, *args, **kwargs):
        """Return the trainable parameters of the model.

        This method delegates to the compressor's parameters method, as only
        the compression model is trained (the denoiser is pre-trained and frozen).

        Args:
            *args, **kwargs: Arguments to pass to the compressor's parameters method

        Returns:
            Iterator over the trainable parameters
        """
        return self.compressor.parameters(*args, **kwargs)

    def load_state_dict(self, state_dict: dict):
        """Load state dictionary into the model.

        This method loads only the compressor's state dictionary, as the
        denoiser is already loaded with pre-trained weights.

        Args:
            state_dict: State dictionary containing model parameters
        """
        self.compressor.load_state_dict(state_dict)

    def get_parameters(self, *args, **kwargs):
        """Get specific parameters from the model.

        This is a wrapper around the compressor's get_parameters method,
        which may return a filtered subset of parameters based on criteria.

        Args:
            *args, **kwargs: Arguments to pass to the compressor's get_parameters method

        Returns:
            Parameters returned by the compressor's get_parameters method
        """
        return self.compressor.get_parameters(*args, **kwargs)

    # Commented unfinished method
    # def
