"""
Test module for image alignment in the RawNIND pipeline.

Objective: To verify that the alignment process correctly computes an optimal integer pixel shift 
between two images and applies it such that the aligned images exhibit high structural similarity, 
ensuring they represent the same spatial content for downstream tasks like denoising or quality evaluation.

Test criteria: 
- The alignment shift is computed without errors using L1 minimization.
- Post-alignment, the Structural Similarity Index (SSIM) between the two images exceeds 0.95, 
  indicating minimal misalignment and perceptual equivalence.
- No exceptions occur during image loading, alignment computation, or SSIM calculation.

How testing for this criteria fulfills the purpose: The SSIM assertion directly quantifies the 
effectiveness of the alignment by measuring structural, luminance, and contrast similarity, 
replacing subjective manual inspection with an objective, automated metric that detects residual 
misalignments (e.g., shifts causing feature mismatches) while tolerating expected variations 
like noise in raw data pairs. This ensures the pipeline's alignment step reliably prepares 
comparable image pairs, fulfilling the intent of accurate data preprocessing.

No components are mocked, monkeypatched, or are fixtures: This test uses the real 'rawproc' 
module for alignment functions and 'np_imgops' for image I/O, along with scikit-image for SSIM.

Reasons for using real components without mocking/patching/fixturing: The test's objective is 
integration-level validation of the full alignment pipeline (loading -> shift computation -> 
application -> quality check) to ensure end-to-end correctness in the RawNIND environment. 
Mocking would abstract away potential issues in real image handling or computation, compromising 
the test's ability to detect unintended behaviors like numerical instability or I/O errors. 
Using real components guarantees the assertion reflects actual runtime performance without 
unnecessary complexity, while keeping the test performant (SSIM is efficient for typical image sizes).
"""

import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

import pytest

from rawnind.libs import rawproc, np_imgops


def test_image_alignment():
    """Test image alignment verification using SSIM."""
    # Input file paths for the test images (assume test_data/ exists in project root or adjust path)
    fp1 = os.path.join("test_data", "Moor_frog_bl.jpg")
    fp2 = os.path.join("test_data", "Moor_frog_tr.jpg")

    # Load images as floating-point numpy arrays
    img1 = np_imgops.img_fpath_to_np_flt(fp1)
    img2 = np_imgops.img_fpath_to_np_flt(fp2)

    # Find the optimal alignment between the two images
    best_alignment = rawproc.find_best_alignment(
        anchor_img=img1, target_img=img2, verbose=False  # Disable verbose for test quietness
    )

    # Apply the calculated alignment to both images
    img1_aligned, img2_aligned = rawproc.shift_images(img1, img2, best_alignment)

    # Normalize images to [0, 1] range if necessary
    if img1_aligned.max() > 1.0:
        img1_aligned = img1_aligned / img1_aligned.max()
    if img2_aligned.max() > 1.0:
        img2_aligned = img2_aligned / img2_aligned.max()

    # Compute SSIM for verification (multichannel for RGB)
    similarity_score = ssim(
        img1_aligned, img2_aligned, multichannel=True, data_range=1.0
    )

    # Assert alignment quality
    alignment_threshold = 0.95
    assert similarity_score > alignment_threshold, (
        f"Alignment verification failed! SSIM score: {similarity_score:.4f} "
        f"(threshold: {alignment_threshold}). Check shift {best_alignment} or image data."
    )

if __name__ == "__main__":
    pytest.main([__file__])