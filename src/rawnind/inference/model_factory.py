"""Model factory for creating and loading model instances.

This module provides factory functions for creating model instances
and loading pre-trained models for inference. It handles model
architecture selection and initialization.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import os
import torch

from .base_inference import ImageToImageNN, BayerImageToImageNN
from ..models import bm3d_denoiser, compression_autoencoders, denoise_then_compress, manynets_compression, \
    raw_denoiser, standard_compressor


# Import from dependencies package (will be moved later)


class DenoiseCompress(ImageToImageNN):
    """Combined denoising and compression model for inference.

    This class handles models that perform both denoising and compression
    operations on input images.
    """
    MODELS_BASE_DPATH = os.path.join("..", "..", "models", "rawnind_dc")
    ARCHS = {
        "ManyPriors"         : manynets_compression.ManyPriors_RawImageCompressor,
        "DenoiseThenCompress": denoise_then_compress.DenoiseThenCompress,
        "JPEGXL"             : standard_compressor.JPEGXL_ImageCompressor,
        "JPEG"               : standard_compressor.JPEGXL_ImageCompressor,
        "Passthrough"        : standard_compressor.Passthrough_ImageCompressor,
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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--arch_enc",
            help="Encoder architecture",
            required=True,
            choices=self.ARCHS_ENC.keys(),
        )
        parser.add_argument(
            "--arch_dec",
            help="Decoder architecture",
            required=True,
            choices=self.ARCHS_DEC.keys(),
        )
        parser.add_argument("--hidden_out_channels", type=int)
        parser.add_argument("--bitstream_out_channels", type=int)

    def _get_resume_suffix(self) -> str:
        return "combined"


class BayerDenoiseCompress(DenoiseCompress, BayerImageToImageNN):
    """Bayer-specific combined denoising and compression model for inference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Denoiser(ImageToImageNN):
    """Pure denoising model for inference.

    This class handles models that perform only denoising operations
    on input images.
    """
    MODELS_BASE_DPATH = os.path.join("..", "..", "models", "rawnind_denoise")
    ARCHS = {
        "unet"    : raw_denoiser.UtNet2,
        "utnet3"  : raw_denoiser.UtNet3,
        # "runet": runet.Runet,
        "identity": raw_denoiser.Passthrough,
        # "edsr": edsr.EDSR,
        "bm3d"    : bm3d_denoiser.BM3D_Denoiser,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_resume_suffix(self) -> str:
        return self.loss

    def instantiate_model(self):
        self.model: torch.nn.Module = self.ARCHS[self.arch](
            in_channels=self.in_channels,
            funit=self.funit,
            preupsample=vars(self).get("preupsample", False),
        ).to(self.device)

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--loss",
            help="Distortion loss function",
            choices=["msssim", "psnr", "l1", "l2"],  # Simplified for inference
            required=True,
        )


class BayerDenoiser(Denoiser, BayerImageToImageNN):
    """Bayer-specific denoising model for inference."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def get_and_load_test_object(
        **kwargs,
) -> ImageToImageNN:  # only used in denoise_image.py
    """Parse config file or arch parameter to get the class name, ie Denoiser or DenoiseCompress.

    This function creates a test object based on the architecture specified in the
    configuration or command line arguments. It handles both denoiser and
    denoise+compress models for different input channel configurations.

    Args:
        **kwargs: Additional arguments passed to the model constructor

    Returns:
        ImageToImageNN: Configured model instance ready for testing/inference

    Raises:
        NotImplementedError: If the specified architecture is not supported
    """
    import configargparse

    # Parse arguments
    parser = configargparse.ArgumentParser(
        description=__doc__,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config",
        is_config_file=True,
        dest="config",
        required=False,
        help="config in yaml format",
    )
    parser.add_argument(
        "--arch",
        help="Model architecture",
        required=True,
        choices=Denoiser.ARCHS.keys() | DenoiseCompress.ARCHS.keys(),
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Number of input channels (3 for profiled RGB, 4 for Bayer)",
        choices=[3, 4],
        required=True,
    )
    args, _ = parser.parse_known_args()

    if args.arch in Denoiser.ARCHS.keys():
        if args.in_channels == 4:
            test_obj = BayerDenoiser(test_only=True, **kwargs)
        else:
            test_obj = Denoiser(test_only=True, **kwargs)
    elif args.arch in DenoiseCompress.ARCHS.keys():
        if args.in_channels == 4:
            test_obj = BayerDenoiseCompress(test_only=True, **kwargs)
        else:
            test_obj = DenoiseCompress(test_only=True, **kwargs)
    else:
        raise NotImplementedError(f"Unknown architecture {args.arch}")

    # Set model to evaluation mode
    test_obj.model = test_obj.model.eval()
    return test_obj


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
