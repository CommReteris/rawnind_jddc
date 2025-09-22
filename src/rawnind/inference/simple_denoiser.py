"""Simple denoising script demonstrating clean API usage.

This script shows how to use the clean inference API for single image denoising,
replacing the legacy CLI-based approach with modern programmatic interfaces.

Example usage:
    python -m rawnind.inference.simple_denoiser --input_fpath image.raw --output_fpath denoised.tif
"""

import argparse
import os
import time
from pathlib import Path

import torch

# Use clean API instead of direct imports
from rawnind.inference import load_model_from_checkpoint, compute_image_metrics
from rawnind.inference import image_denoiser  # For utility functions
from rawnind.dependencies import pytorch_helpers
from rawnind.dependencies import raw_processing as rawproc

DEFAULT_MODEL_PATTERN = "*bayer*"  # Pattern to find default Bayer model

def denoise_single_image_clean_api(input_fpath: str, output_fpath: str = None,
                                 model_checkpoint: str = None, device: str = 'auto'):
    """Denoise a single image using the clean API.
    
    Args:
        input_fpath: Input image file path
        output_fpath: Output image file path (auto-generated if None)
        model_checkpoint: Model checkpoint directory path (auto-detected if None)
        device: Device to use ('auto', 'cpu', 'cuda', or device number)
    """
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
    
    # Load model using clean API
    print(f"Loading model from {model_checkpoint}...")
    denoiser = load_model_from_checkpoint(model_checkpoint, device=device)
    
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
        input_image, rgb_xyz_matrix = image_denoiser.load_image(infile, device)
        
        # Time the inference
        start = time.time()
        processed = denoiser.denoise(input_image)
        end = time.time()
        
        # Apply post-processing if Bayer
        if denoiser.input_channels == 4:  # Bayer input
            processed = rawproc.match_gain(anchor_img=input_image, other_img=processed)
            processed = rawproc.camRGB_to_lin_rec2020_images(processed.unsqueeze(0), rgb_xyz_matrix)
            processed = processed.squeeze(0)
        
        print(f"Saving to {output_fpath}. Processing time: {end - start:.2f} s")
        image_denoiser.save_image(processed.unsqueeze(0), output_fpath, src_fpath=input_fpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple denoising using clean API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input_fpath", required=True, help="Input image path")
    parser.add_argument("-o", "--output_fpath", help="Output image path (auto-generated if not provided)")
    parser.add_argument("--model_checkpoint", help="Model checkpoint directory (auto-detected if not provided)")
    parser.add_argument("--device", default="auto", help="Device: 'auto', 'cpu', 'cuda', or device number")
    args = parser.parse_args()
    
    denoise_single_image_clean_api(
        input_fpath=args.input_fpath,
        output_fpath=args.output_fpath,
        model_checkpoint=args.model_checkpoint,
        device=args.device
    )