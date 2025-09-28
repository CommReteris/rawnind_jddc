"""Model factory for creating and loading model instances.

This module provides factory functions for creating model instances
and loading pre-trained models for inference. It handles model
architecture selection and initialization.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import os
import torch

from .base_inference import ImageToImageNN, BayerImageToImageNN
from .configs import InferenceConfig
from ..models import bm3d_denoiser, compression_autoencoders, denoise_then_compress, manynets_compression, \
    raw_denoiser, standard_compressor


# Import from dependencies package (will be moved later)


class DenoiseCompress(ImageToImageNN):
    """Combined denoising and compression model for inference.

    This class handles models that perform both denoising and compression
    operations on input images.
    """
    MODELS_BASE_DPATH = os.path.join(os.path.dirname(__file__), "../..", "models", "rawnind_dc")
    ARCHS = {
        "ManyPriors"         : manynets_compression.ManyPriors_RawImageCompressor,
        "DenoiseThenCompress": denoise_then_compress.DenoiseThenCompress,
        "JPEGXL"             : standard_compressor.JPEGXL_ImageCompressor,
        "JPEG"               : standard_compressor.JPEGXL_ImageCompressor,
        "Passthrough"        : standard_compressor.Passthrough_ImageCompressor,
        "standard"           : standard_compressor.Std_ImageCompressor,
        "autoencoder"        : compression_autoencoders.AbstractRawImageCompressor,
    }
    ARCHS_ENC = {
        "Balle": compression_autoencoders.BalleEncoder,
        # "BayerPreUp": compression_autoencoders.BayerPreUpEncoder,
    }
    ARCHS_DEC = {
        "Balle"  : compression_autoencoders.BalleDecoder,
        "BayerPS": compression_autoencoders.BayerPSDecoder,
        "BayerTC": compression_autoencoders.BayerTCDecoder,
    }

    def __init__(self, config):
        # Accept config dataclass or dict
        if isinstance(config, dict):
            config_obj = InferenceConfig(**config)
        else:
            config_obj = config
        # Set attributes before calling super().__init__ because instantiate_model uses them
        self.arch = getattr(config_obj, 'architecture', 'ManyPriors')
        self.in_channels = getattr(config_obj, 'input_channels', 3)
        self.funit = getattr(config_obj, 'filter_units', 48)
        self.hidden_out_channels = getattr(config_obj, 'hidden_out_channels', 192)
        self.bitstream_out_channels = getattr(config_obj, 'bitstream_out_channels', 64)
        self.arch_enc = getattr(config_obj, 'encoder_arch', 'Balle')
        self.arch_dec = getattr(config_obj, 'decoder_arch', 'Balle')
        super().__init__(config_obj)

    def instantiate_model(self) -> None:
        self.model: torch.nn.Module = self.ARCHS[self.arch](
            in_channels=self.in_channels,
            funit=self.funit,
            device=self.device,
            hidden_out_channels=self.hidden_out_channels,
            bitstream_out_channels=self.bitstream_out_channels,
            encoder_cls=self.ARCHS_ENC[self.arch_enc],
            decoder_cls=self.ARCHS_DEC[self.arch_dec],
            preupsample=vars(self).get("preupsample", False),
        ).to(self.device)

    # CLI interface removed - use clean factory functions instead

    def _get_resume_suffix(self) -> str:
        return "combined"


class BayerDenoiseCompress(DenoiseCompress, BayerImageToImageNN):
    """Bayer-specific combined denoising and compression model for inference."""

    def __init__(self, config):
        # Accept config dataclass or dict
        if isinstance(config, dict):
            config_obj = InferenceConfig(**config)
        else:
            config_obj = config
        super().__init__(config_obj)
        self.arch = getattr(config_obj, 'architecture', 'ManyPriors')
        self.in_channels = getattr(config_obj, 'input_channels', 3)
        self.funit = getattr(config_obj, 'filter_units', 48)
        self.hidden_out_channels = getattr(config_obj, 'hidden_out_channels', 192)
        self.bitstream_out_channels = getattr(config_obj, 'bitstream_out_channels', 64)
        self.arch_enc = getattr(config_obj, 'encoder_arch', 'Balle')
        self.arch_dec = getattr(config_obj, 'decoder_arch', 'Balle')


class Denoiser(ImageToImageNN):
    """Pure denoising model for inference.

    This class handles models that perform only denoising operations
    on input images.
    """
    MODELS_BASE_DPATH = os.path.join(os.path.dirname(__file__), "../..", "models", "rawnind_denoise")
    ARCHS = {
        "unet"    : raw_denoiser.UtNet2,
        "utnet3"  : raw_denoiser.UtNet3,
        "autoencoder": raw_denoiser.UtNet2,  # Alias for testing
        # "runet": runet.Runet,
        "identity": raw_denoiser.Passthrough,
        # "edsr": edsr.EDSR,
        "bm3d"    : bm3d_denoiser.BM3D_Denoiser,
    }

    def __init__(self, config):
        # Accept config dataclass or dict
        if isinstance(config, dict):
            config_obj = InferenceConfig(**config)
        else:
            config_obj = config
        # Set attributes before calling super().__init__ because instantiate_model uses them
        self.arch = getattr(config_obj, 'architecture', 'unet')
        self.in_channels = getattr(config_obj, 'input_channels', 3)
        self.funit = getattr(config_obj, 'filter_units', 48)
        self.loss = getattr(config_obj, 'loss_function', 'mse')
        super().__init__(config_obj)

    def _get_resume_suffix(self) -> str:
        return self.loss

    def instantiate_model(self):
        self.model: torch.nn.Module = self.ARCHS[self.arch](
            in_channels=self.in_channels,
            funit=self.funit,
            preupsample=vars(self).get("preupsample", False),
        ).to(self.device)

    # CLI interface removed - use clean factory functions instead


class BayerDenoiser(Denoiser, BayerImageToImageNN):
    """Bayer-specific denoising model for inference."""

    def __init__(self, config):
        # Accept config dataclass or dict
        if isinstance(config, dict):
            config_obj = InferenceConfig(**config)
        else:
            config_obj = config
        # Set attributes before calling super().__init__ because instantiate_model uses them
        self.arch = getattr(config_obj, 'architecture', 'unet')
        self.in_channels = getattr(config_obj, 'input_channels', 3)
        self.funit = getattr(config_obj, 'filter_units', 48)
        self.loss = getattr(config_obj, 'loss_function', 'mse')
        super().__init__(config_obj)


# CLI interface removed - use clean factory functions instead:
# from rawnind.inference import create_rgb_denoiser, create_bayer_denoiser, load_model_from_checkpoint

def get_and_load_test_object(**kwargs) -> ImageToImageNN:
    """DEPRECATED: Use clean factory functions instead.

    This function is kept for backward compatibility but should not be used.
    Use the clean API factory functions instead:
    - create_rgb_denoiser() for RGB denoising
    - create_bayer_denoiser() for Bayer denoising
    - load_model_from_checkpoint() for loading trained models

    Raises:
        DeprecationWarning: This function is deprecated
    """
    import warnings
    warnings.warn(
        "get_and_load_test_object() is deprecated. Use clean factory functions: "
        "create_rgb_denoiser(), create_bayer_denoiser(), or load_model_from_checkpoint()",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError(
        "CLI-based model loading removed. Use clean factory functions: "
        "create_rgb_denoiser(), create_bayer_denoiser(), or load_model_from_checkpoint()"
    )


def get_and_load_model(**kwargs):
    """Load a model for inference (deprecated - use get_and_load_test_object instead).

    This function is deprecated and will be removed in future versions.
    Use get_and_load_test_object() instead.

    Args:
        **kwargs: Arguments passed to get_and_load_test_object

    Returns:
        torch.nn.Module: The loaded model in evaluation mode
    """
    return get_and_load_test_object(**kwargs).model
