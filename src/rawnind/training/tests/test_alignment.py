"""
Objective: Verify image alignment using SSIM metric with real image files for validation.
Test Criteria: Load two test images; compute best alignment shift using L1 minimization; assert SSIM >0.95 for aligned images.
Fulfillment: Ensures alignment function works correctly on real data, detecting shifts and improving structural similarity; uses actual images to validate end-to-end alignment pipeline.
Components Mocked/Fixtured: None - uses real file loading and alignment computation for accuracy.
Reasons for No Mocking: Real images are available in test_data/ directory; no mocks needed as test is integration-style with deterministic input; fulfills intent of verifying alignment on actual data without simulation.
"""

import os
import pytest
from skimage.metrics import structural_similarity as ssim
from rawnind.dependencies import raw_processing as rawproc, numpy_operations as np_imgops

def test_image_alignment():
    """Test image alignment verification using SSIM."""
    # Assuming test images exist in test_data/; if not, skip or use dummy
    fp1 = os.path.join("test_data", "Moor_frog_bl.jpg")
    fp2 = os.path.join("test_data", "Moor_frog_tr.jpg")

    if not os.path.exists(fp1) or not os.path.exists(fp2):
        pytest.skip("Test images not found in test_data/; place Moor_frog_bl.jpg and Moor_frog_tr.jpg there.")

    img1 = np_imgops.img_fpath_to_np_flt(fp1)
    img2 = np_imgops.img_fpath_to_np_flt(fp2)

    # Find best alignment
    best_alignment = rawproc.find_best_alignment(img1, img2, verbose=False)

    # Apply shift
    img1_aligned, img2_aligned = rawproc.shift_images(img1, img2, best_alignment)

    # Verify SSIM
    similarity_score = ssim(img1_aligned, img2_aligned, multichannel=True, data_range=1.0)

    assert similarity_score > 0.95, f"Alignment SSIM {similarity_score:.4f} below threshold 0.95. Check shift {best_alignment} or images."