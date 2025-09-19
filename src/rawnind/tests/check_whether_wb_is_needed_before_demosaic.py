"""
White balance order test for the raw image processing pipeline.

This script investigates whether applying white balance before or after demosaicing
affects the final image quality in the Natural Image Noise Dataset (NIND) processing pipeline.
The test compares two processing pipelines:

1. Without pre-demosaic white balance:
   raw (mono) → demosaic → color conversion to sRGB

2. With pre-demosaic white balance (and subsequent reversal):
   raw (mono) → white balance → demosaic → reverse white balance → color conversion to sRGB

The script calculates the mean absolute difference between the final output images
from both pipelines across various test images. Smaller differences suggest that
the order of operations has minimal impact, while larger differences indicate that
the order matters significantly.

Testing multiple demosaicing algorithms helps determine if the sensitivity to white balance
order is algorithm-dependent. The results can guide the implementation of the main
image processing pipeline to ensure optimal quality.

Usage:
    python check_whether_wb_is_needed_before_demosaic.py

Output:
    Mean absolute difference values for each image and algorithm combination,
    plus the overall average difference across all tested images.
"""

import os
import statistics
import sys
import numpy as np
import cv2

from ..libs import raw

# Path to the ground truth images in the Natural Image Noise Dataset
IMAGE_SETS_DPATH = os.path.join("..", "..", "datasets", "RawNIND", "Bayer")

if __name__ == "__main__":
    # Test multiple demosaicing algorithms to check if results are algorithm-dependent
    for demosaic_algorithm_name, demosaic_algorithm in {
        "COLOR_BayerRGGB2RGB_EA": cv2.COLOR_BayerRGGB2RGB_EA,  # Edge-aware demosaicing (higher quality)
        "COLOR_BayerRGGB2RGB"   : cv2.COLOR_BayerRGGB2RGB,  # Standard demosaicing
    }.items():
        print(f"{demosaic_algorithm_name=}")
        
        # Container for storing difference measurements for this algorithm
        losses = []
        
        # Iterate through all image sets in the dataset
        for aset in os.listdir(IMAGE_SETS_DPATH):
            # Focus on ground truth (clean) images only
            dpath = os.path.join(IMAGE_SETS_DPATH, aset, "gt")

            # Process each image in the current set
            for fn in os.listdir(dpath):
                fpath = os.path.join(dpath, fn)

                # Load raw image and its metadata
                mono_img, metadata = raw.raw_fpath_to_mono_img_and_metadata(fpath)

                # ===== PIPELINE 1: WITHOUT PRE-DEMOSAIC WHITE BALANCE =====
                # Step 1: Apply demosaicing directly to raw mono image
                camRGB_img_nowb = raw.demosaic(
                    mono_img, metadata, method=demosaic_algorithm
                )

                # Step 2: Convert to gamma-corrected sRGB color space
                gamma_sRGB_img_nowb = raw.camRGB_to_profiledRGB_img(
                    camRGB_img_nowb, metadata, "gamma_sRGB"
                )

                # ===== PIPELINE 2: WITH PRE-DEMOSAIC WHITE BALANCE =====
                # Step 1: Apply white balance to raw mono image
                mono_img_wb = raw.apply_whitebalance(
                    mono_img, metadata, wb_type="daylight", in_place=False
                )

                # Step 2: Apply demosaicing to white-balanced mono image
                camRGB_img_wb = raw.demosaic(mono_img_wb, metadata, method=demosaic_algorithm)

                # Step 3: Reverse the white balance on the demosaiced image
                # This ensures color accuracy while isolating the effect of pre-demosaic white balance
                camRGB_img_wb = raw.apply_whitebalance(
                    camRGB_img_wb,
                    metadata,
                    wb_type="daylight",
                    in_place=False,
                    reverse=True,
                )

                # Step 4: Convert to gamma-corrected sRGB color space (same as pipeline 1)
                gamma_sRGB_img_wb = raw.camRGB_to_profiledRGB_img(
                    camRGB_img_wb, metadata, "gamma_sRGB"
                )

                # ===== ANALYSIS: COMPARE RESULTS FROM BOTH PIPELINES =====
                # Calculate mean absolute difference between the two output images
                # Lower values indicate less sensitivity to white balance order
                loss = np.abs(gamma_sRGB_img_nowb - gamma_sRGB_img_wb).mean()
                print(f"{fn=}, {loss=}")
                losses.append(loss)

        # Report the overall average difference across all images for this algorithm
        # This indicates the overall sensitivity to white balance order
        print(f"{statistics.mean(losses)=}")