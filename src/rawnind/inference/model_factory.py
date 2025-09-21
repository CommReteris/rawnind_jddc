"""Model factory for creating and loading model instances.

This module provides factory functions for creating model instances
and loading pre-trained models for inference. It handles model
architecture selection and initialization.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import torch


# Import from dependencies package (will be moved later)


def get_and_load_test_object(
        **kwargs,
) -> 'ImageToImageNN':  # Forward reference to avoid circular import
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
    # Import configargparse for argument parsing
    import configargparse

    # Import model classes (will be moved to models package later)
    from ..models.raw_denoiser import UtNet2, UtNet3, Passthrough
    from ..models.bm3d_denoiser import BM3D_Denoiser
    from ..models.denoise_then_compress import DenoiseThenCompress
    from ..models.manynets_compression import ManyPriors_RawImageCompressor
    from ..models.standard_compressor import JPEGXL_ImageCompressor, Passthrough_ImageCompressor

    # Define available architectures
    DENOISER_ARCHS = {
        "unet"    : UtNet2,
        "utnet3"  : UtNet3,
        "identity": Passthrough,
        "bm3d"    : BM3D_Denoiser,
    }

    COMPRESS_ARCHS = {
        "ManyPriors"         : ManyPriors_RawImageCompressor,
        "DenoiseThenCompress": DenoiseThenCompress,
        "JPEGXL"             : JPEGXL_ImageCompressor,
        "JPEG"               : JPEGXL_ImageCompressor,
        "Passthrough"        : Passthrough_ImageCompressor,
    }

    # Parse arguments
    parser = configargparse.ArgumentParser(
        description="Parse config file or arch parameter to get the class name",
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
        choices=list(DENOISER_ARCHS.keys()) + list(COMPRESS_ARCHS.keys()),
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Number of input channels (3 for profiled RGB, 4 for Bayer)",
        choices=[3, 4],
        required=True,
    )
    args, _ = parser.parse_known_args()

    # Create appropriate model instance
    if args.arch in DENOISER_ARCHS:
        if args.in_channels == 4:
            # Bayer-specific denoiser (will be implemented later)
            from .base_inference import BaseInference  # Forward reference
            test_obj = BaseInference(test_only=True, **kwargs)
        else:
            # RGB denoiser (will be implemented later)
            from .base_inference import BaseInference  # Forward reference
            test_obj = BaseInference(test_only=True, **kwargs)
    elif args.arch in COMPRESS_ARCHS:
        if args.in_channels == 4:
            # Bayer-specific compression (will be implemented later)
            from .base_inference import BaseInference  # Forward reference
            test_obj = BaseInference(test_only=True, **kwargs)
        else:
            # RGB compression (will be implemented later)
            from .base_inference import BaseInference  # Forward reference
            test_obj = BaseInference(test_only=True, **kwargs)
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
