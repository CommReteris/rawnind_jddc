<<<<<<< HEAD
import argparse
import os
import sys
import time

import torch

from rawnind.inference import image_denoiser
from rawnind.dependencies import pytorch_helpers
from rawnind.libs import rawproc
from ..models import raw_denoiser

MODEL_FPATH = os.path.join(os.path.abspath(os.path.curdir),
                           "src/rawnind/models/rawnind_denoise/DenoiserTrainingBayerToProfiledRGB_4ch_2024-11-22"
                           "-bayer_ms-ssim_mgout_notrans_valeither_noowwnpics_mgdef_-1/saved_models/iter_1245000.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model_fpath", default=MODEL_FPATH)
    parser.add_argument("-i", "--input_fpath", required=True)
    parser.add_argument("-o", "--output_fpath")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    # TODO debayer if prgb model
    model_is_bayer = 'bayer' in args.model_fpath.lower()
    if pytorch_helpers.is_xtrans(args.input_fpath):
        infile = args.input_fpath.replace('RAF', 'exr')
        pytorch_helpers.xtrans_fpath_to_OpenEXR(args.input_fpath, infile)
    else:
        infile = args.input_fpath
    if args.output_fpath:
        output_fpath = args.output_fpath
    else:
        # {input_fpath}_{model grandparent directory}_{model_fn}.tif
        model_fpath = os.path.abspath(args.model_fpath)
        model_parent_dir = os.path.basename(os.path.dirname(model_fpath))
        model_grandparent_dir = os.path.basename(
            os.path.dirname(os.path.dirname(model_fpath))
        )
        model_fn = os.path.basename(model_fpath).replace(".", "-")
        output_fpath = f"{args.input_fpath}_{model_grandparent_dir}_{model_fn}.tif"

    device = pytorch_helpers.get_device(use_cpu=args.cpu)

    with torch.no_grad():
        input_image, rgb_xyz_matrix = image_denoiser.load_image(
            infile, device=device
        )
        input_image = input_image.unsqueeze(0)
        model = raw_denoiser.UtNet2(
            in_channels=4 if model_is_bayer and not infile.endswith(".exr") else 3, funit=32
        )
        model.load_state_dict(
            torch.load(
                args.model_fpath, map_location=torch.device("cpu") if args.cpu else None
            )
        )
        model.eval()
        model = model.to(device)
        input_image = input_image.to(device)
        # time it
        start = time.time()
        out_image = model(input_image)
        end = time.time()
        out_image = rawproc.match_gain(anchor_img=input_image, other_img=out_image)
        out_image = rawproc.camRGB_to_lin_rec2020_images(out_image, rgb_xyz_matrix)
        out_image = out_image.cpu()
        print(f"Saving to {output_fpath}. Processing time: {end - start:.2f} s")
        image_denoiser.save_image(out_image, output_fpath, src_fpath=args.input_fpath)
=======
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
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
