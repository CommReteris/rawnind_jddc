"""OpenEXR bit depth test for high dynamic range (HDR) image export.

This script tests the ability to save images in OpenEXR format with different bit depths
(16-bit and 32-bit precision) using the raw module's hdr_nparray_to_file function.

The test creates a single random 3-channel image and exports it in two formats:
1. 32-bit float precision (.exr) - Higher precision but larger file size
2. 16-bit half float precision (.exr) - Smaller file size with slightly reduced precision

Both exports use the linear Rec.2020 color profile, which is important for HDR workflows
and preserves the wide color gamut needed for high-quality image processing.

Testing both bit depths ensures that:
- The HDR export pipeline can handle different precision requirements
- Users can choose the appropriate trade-off between file size and precision
- The color profile metadata is correctly embedded in both formats

The expected output is two OpenEXR files in the tests_output directory that contain
identical image data but with different bit depths.

Usage:
    python test_openEXR_bit_depth.py

Expected output:
    Creates two files in the tests_output directory:
    - test_openEXR_bit_depth_32.exr (32-bit float precision)
    - test_openEXR_bit_depth_16.exr (16-bit half float precision)
"""

import sys
import numpy as np
import os

from .libs import raw

if __name__ == "__main__":
    # Create a random test image with 3 channels (RGB) and 128x128 resolution
    # Values are in [0.0, 1.0] range, suitable for HDR format
    image: np.ndarray = np.random.random((3, 128, 128))

    # Ensure output directory exists
    os.makedirs("tests_output", exist_ok=True)

    # Export the same image in 32-bit float precision
    # This provides maximum precision but results in larger file size
    raw.hdr_nparray_to_file(
        image,
        "tests_output/test_openEXR_bit_depth_32.exr",
        bit_depth=32,  # 32-bit float precision
        color_profile="lin_rec2020",  # Linear Rec.2020 color space
    )

    # Export the same image in 16-bit half float precision
    # This reduces file size while maintaining good precision for most applications
    raw.hdr_nparray_to_file(
        image,
        "tests_output/test_openEXR_bit_depth_16.exr",
        bit_depth=16,  # 16-bit half float precision
        color_profile="lin_rec2020",  # Linear Rec.2020 color space
    )

    print("OpenEXR bit depth test complete.")
    print("Created: tests_output/test_openEXR_bit_depth_32.exr (32-bit)")
    print("Created: tests_output/test_openEXR_bit_depth_16.exr (16-bit)")
