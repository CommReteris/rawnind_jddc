'''Simple denoising script demonstrating clean API usage.

This script shows how to use the clean inference API for single image denoising,
replacing the legacy CLI-based approach with modern programmatic interfaces.

Example usage:
    python -m rawnind.inference.simple_denoiser --input_fpath image.raw --output_fpath denoised.tif
'''

import time
from pathlib import Path

import torch

# Use clean API instead of direct imports
from .clean_api import create_bayer_denoiser, create_rgb_denoiser, InferenceConfig
from ..dependencies import pytorch_helpers, raw_processing as rawproc

DEFAULT_MODEL_PATTERN = "*bayer*"  # Pattern to find default Bayer model


def denoise_single_image_clean_api(input_fpath: str, output_fpath: str = None,
                                   model_checkpoint: str = None, device: str = 'auto',
                                   architecture: str = 'unet', input_channels: int = 3):
    '''Denoise a single image using the clean API.
    
    Args:
        input_fpath: Input image file path
        output_fpath: Output image file path (auto-generated if None)
        model_checkpoint: Model checkpoint directory path (auto-detected if None)
        device: Device to use ('auto', 'cpu', 'cuda', or device number)
        architecture: Model architecture ('unet', 'utnet3', etc.)
        input_channels: Number of input channels (3 for RGB, 4 for Bayer)
    '''
    # Auto-detect model if not provided
    if model_checkpoint is None:
        weights_dir = Path("src/rawnind/models/weights")
        model_dirs = list(weights_dir.glob(DEFAULT_MODEL_PATTERN))
        if not model_dirs:
            raise FileNotFoundError(f"No models found matching pattern {DEFAULT_MODEL_PATTERN} in {weights_dir}")
        model_checkpoint = str(model_dirs[0])
        print(f"Auto-detected model: {model_checkpoint}")

    # Set device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create denoiser using clean API factory
    print(f"Creating {architecture} denoiser for {input_channels} channels...")
    if input_channels == 3:
        denoiser = create_rgb_denoiser(
            architecture=architecture,
            checkpoint_path=model_checkpoint,
            device=device
        )
    elif input_channels == 4:
        denoiser = create_bayer_denoiser(
            architecture=architecture,
            checkpoint_path=model_checkpoint,
            device=device
        )
    else:
        raise ValueError(f"Unsupported input channels: {input_channels}")

    # Prepare input file (handle X-Trans if needed)
    if pytorch_helpers.is_xtrans(input_fpath):
        infile = input_fpath.replace('RAF', 'exr')
        pytorch_helpers.xtrans_fpath_to_OpenEXR(input_fpath, infile)
    else:
        infile = input_fpath

    # Generate output path if not provided
    if output_fpath is None:
        model_name = Path(model_checkpoint).name
        input_name = Path(input_fpath).stem
        output_fpath = f"{input_name}_denoised_{model_name}.tif"

    # Load and process image
    print(f"Processing {infile}...")
    with torch.no_grad():
        input_image, rgb_xyz_matrix = rawproc.load_image(infile, device)

        # Time the inference
        start = time.time()
        if input_channels == 3:
            processed = denoiser.denoise(input_image)
        else:  # Bayer
            processed = denoiser.denoise_bayer(input_image, rgb_xyz_matrix)
        end = time.time()

        print(f"Saving to {output_fpath}. Processing time: {end - start:.2f} s")
        rawproc.save_image(processed.unsqueeze(0), output_fpath, src_fpath=input_fpath)


"""
Example programmatic usage:

denoise_single_image_clean_api(
    input_fpath="path/to/noisy_image.raw",
    output_fpath="path/to/denoised_image.tif",
    model_checkpoint="path/to/model/weights",
    device="cuda",
    architecture="unet",
    input_channels=4  # for Bayer
)
"""
