"""Image alignment testing script for the RawNIND image processing pipeline.

This script tests the image alignment functionality by finding the optimal alignment between
two test images and applying the alignment transformation. It demonstrates the use of:

1. find_best_alignment: Searches for the optimal integer shift (dy, dx) that minimizes
   the mean L1 difference between two images.
2. shift_images: Applies the calculated shift to align the images and crops them
   to ensure they cover the same spatial area.

The test uses two sample images from the Moor frog dataset ("Moor_frog_bl.jpg" and
"Moor_frog_tr.jpg") and evaluates how well they can be aligned. This alignment process
is critical for:
- Comparing clean and noisy versions of the same scene
- Preparing training data for denoising networks
- Evaluating image quality metrics between reference and processed images

The script outputs:
- The calculated optimal alignment shift as (dy, dx) coordinates
- Two aligned images saved to the tests_output directory

Usage:
    python test_alignment.py

Expected output:
    Prints the alignment shift vector and saves aligned images.
    Small shift values indicate minimal misalignment between source images.
"""

import os
import sys

from rawnind.libs import rawproc
from common.libs import np_imgops

if __name__ == "__main__":
    # Input file paths for the test images
    FP1: str = os.path.join("test_data", "Moor_frog_bl.jpg")
    FP2: str = os.path.join("test_data", "Moor_frog_tr.jpg")

    # Load images as floating-point numpy arrays
    img1 = np_imgops.img_fpath_to_np_flt(FP1)
    img2 = np_imgops.img_fpath_to_np_flt(FP2)

    # Find the optimal alignment between the two images
    # The verbose flag enables detailed logging of alignment search progress
    best_alignment = rawproc.find_best_alignment(
        anchor_img=img1, target_img=img2, verbose=True
    )
    print(f"Optimal alignment shift (dy, dx): {best_alignment}")

    # Apply the calculated alignment to both images
    # This ensures both images cover the same spatial area after alignment
    img1_aligned, img2_aligned = rawproc.shift_images(img1, img2, best_alignment)

    # Save the aligned images to the output directory
    output_dir = os.path.join("tests_output")
    os.makedirs(output_dir, exist_ok=True)
    np_imgops.np_to_img(
        img1_aligned, os.path.join(output_dir, "Moor_frog_bl_aligned.png")
    )
    np_imgops.np_to_img(
        img2_aligned, os.path.join(output_dir, "Moor_frog_tr_aligned.png")
    )
