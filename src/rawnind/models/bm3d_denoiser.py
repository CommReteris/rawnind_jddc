"""BM3D denoiser wrapper for image denoising using an external binary.

This module provides a PyTorch-compatible wrapper around the BM3D (Block-Matching and 3D
filtering) denoising algorithm, which is a state-of-the-art method for image denoising.
Instead of reimplementing the algorithm in Python/PyTorch, this wrapper interfaces with
an external BM3D binary (https://github.com/gfacciol/bm3d) for better performance.

BM3D algorithm overview:
The BM3D algorithm, published by Dabov et al. in 2007 ("Image Denoising by Sparse 3-D 
Transform-Domain Collaborative Filtering"), operates in two major steps:

1. Basic estimate using hard thresholding:
   - Groups similar 2D image patches into 3D blocks based on similarity metrics
   - Applies 3D transformations (DCT, wavelet) to these grouped blocks
   - Performs hard thresholding in the transform domain to remove noise
   - Applies inverse transformation and returns patches to original positions
   - Aggregates results with weights based on patch similarity

2. Final estimate using Wiener filtering:
   - Uses the basic estimate as a reference for better block matching
   - Groups similar patches from both noisy image and basic estimate
   - Applies 3D transformations to these grouped blocks
   - Performs Wiener filtering using the basic estimate as a pilot signal
   - Applies inverse transformation and aggregates with weights

The algorithm is particularly effective for Gaussian noise and often outperforms
deep learning methods on specific noise patterns.

Key features of this implementation:
- Implements the Denoiser interface for consistent usage with other denoisers
- Handles temporary file creation and cleanup for binary interface
- Supports PyTorch tensor input/output for seamless integration with neural networks
- Configurable noise level (sigma) for different denoising strengths
- Command-line interface for standalone usage

Requirements:
- The 'bm3d' binary must be installed and available in the system PATH
- Temporary directory access for intermediate file storage

Note: This implementation focuses on RGB images with 3 channels. The sigma parameter
(passed as 'funit' for compatibility with other models) controls the denoising strength.
Higher sigma values result in stronger denoising but may cause loss of fine details.
Typical values range from 10 (light denoising) to 50 (strong denoising).

Example usage:
    ```python
    # Create a BM3D denoiser with RGB input and sigma=25
    denoiser = BM3D_Denoiser(in_channels=3, funit=25)
    
    # Denoise an image
    noisy_tensor = torch.tensor(noisy_image).unsqueeze(0)  # Add batch dimension
    denoised_tensor = denoiser(noisy_tensor)
    ```
"""

import os
import platform
import random
import shutil
import string
import subprocess
import sys

import torch

from typing import Union
from rawnind.models import raw_denoiser
from rawnind.libs import raw
from rawnind.libs import pt_helpers
from rawnind.libs import np_imgops

TMPDIR = f"tmp_{platform.uname().node}"


class BM3D_Denoiser(raw_denoiser.Denoiser):
    """PyTorch-compatible wrapper for the BM3D denoising algorithm.
    
    This class inherits from the abstract Denoiser class and implements
    a wrapper around an external BM3D binary for image denoising. The
    implementation uses a filesystem-based approach where:
    1. The noisy image is saved to a temporary file
    2. The BM3D binary is called via subprocess to process this file
    3. The denoised result is loaded back into a PyTorch tensor
    
    This approach allows leveraging optimized C++ implementations of BM3D
    without reimplementing the algorithm in PyTorch, while still providing
    a compatible interface with other PyTorch-based denoisers.
    
    Note that this implementation requires the BM3D binary to be installed
    and available in the system PATH. It also requires write access to a
    temporary directory for intermediate file storage.
    """

    def __init__(self, in_channels: int, funit: Union[int, str], *args, **kwargs):
        """Initialize the BM3D denoiser.
        
        Args:
            in_channels: Number of input image channels (must be 3 for RGB)
            funit: Noise standard deviation for the BM3D algorithm. 
                  This parameter controls the denoising strength.
                  (Named 'funit' for compatibility with other models)
            *args, **kwargs: Additional arguments (unused, for compatibility)
            
        Raises:
            AssertionError: If in_channels is not 3 (RGB)
            AssertionError: If the BM3D binary is not found in PATH
            
        Notes:
            - Creates a temporary directory for file-based operations
            - Adds a dummy parameter to make the model compatible with PyTorch's
              parameter management system
            - The 'funit' parameter is stored as 'sigma' for BM3D algorithm
        """
        super().__init__(in_channels=in_channels)
        assert in_channels == 3, f"{in_channels=} should be 3 for BM3D"
        self.sigma = (
            funit  # we use the funit parameter because it's common to other models
        )
        self.dummy_parameter = torch.nn.Parameter(torch.randn(3))
        # check that the bm3d binary exists
        assert shutil.which("bm3d"), "bm3d binary not found in PATH"
        os.makedirs(TMPDIR, exist_ok=True)

    def forward(self, noisy_image):
        """Denoise an image using the external BM3D binary.
        
        This method performs the actual denoising by:
        1. Converting the PyTorch tensor to a NumPy array
        2. Saving the image to a temporary PNG file
        3. Calling the external BM3D binary via subprocess
        4. Loading the denoised result back as a PyTorch tensor
        
        The method handles batch dimension removal/addition and performs
        validation on the input shape to ensure compatibility with BM3D.
        
        Args:
            noisy_image: PyTorch tensor containing the noisy image to denoise.
                        Expected shape is [1, 3, H, W] (batch of 1 RGB image)
                        or [3, H, W] (single RGB image).
                        
        Returns:
            PyTorch tensor containing the denoised image with shape [1, 3, H, W]
            (always includes batch dimension)
            
        Raises:
            AssertionError: If input doesn't have 3 channels (RGB)
            AssertionError: If input shape is invalid
            AssertionError: If BM3D subprocess fails
            
        Notes:
            - Uses 8-bit PNG for file exchange (BM3D binary limitation)
            - Generates random filenames to avoid collisions in parallel usage
            - Currently does not handle values outside [0,1] range specially
            - Temporary files are not automatically deleted (commented out)
            - Contains commented-out alternative approaches for reference
        """
        # The commented-out section below would handle values outside the [0,1] range
        # by storing them separately and adding them back after denoising.
        # This is because the BM3D implementation works with 8-bit PNGs that clamp values.
        # Currently disabled as it can introduce artifacts, but kept for reference.
        # out_of_range_values = noisy_image - torch.clamp(noisy_image, 0, 1)

        # Remove batch dimension and convert to NumPy array for file-based processing
        noisy_image = noisy_image.squeeze(0).numpy()
        print(f"{noisy_image.shape=}, {noisy_image.mean()=}")

        # Validate input shape and channel count
        assert noisy_image.shape[0] == 3, f"{noisy_image.shape=} should be (3, H, W)"
        assert len(noisy_image.shape) == 3, f"{noisy_image.shape=} should be (3, H, W)"

        # Generate unique random filenames to avoid collisions when multiple processes
        # use this denoiser simultaneously
        tmp_str = "".join(random.choices(string.ascii_letters + string.digits, k=23))
        tmp_input_img_fpath = os.path.join(TMPDIR, f"{tmp_str}_input.png")
        tmp_denoised_img_fpath = os.path.join(TMPDIR, f"{tmp_str}_denoised.png")

        # The commented-out section below would normalize the image to [0,1] range
        # This approach is useful when processing HDR images with values outside [0,1]
        # Currently disabled as it can change the relative intensity of image features
        # but kept for potential future use with HDR inputs
        # img_min = noisy_image.min()
        # img_max = noisy_image.max()
        # noisy_image = (noisy_image - img_min) / (img_max - img_min)

        # Save the noisy image as 8-bit PNG (required by BM3D binary)
        # Values outside [0,1] will be clamped due to PNG format limitations
        np_imgops.np_to_img(noisy_image, tmp_input_img_fpath, precision=8)

        # Execute the BM3D binary with the specified sigma (noise level)
        # The command passes three arguments to the BM3D binary:
        # 1. Input image path
        # 2. Sigma value (denoising strength)
        # 3. Output image path
        cmd = ("bm3d", tmp_input_img_fpath, str(self.sigma), tmp_denoised_img_fpath)
        cmd_res = subprocess.run(cmd)

        # Verify that the denoising process completed successfully
        assert cmd_res.returncode == 0 and os.path.isfile(tmp_denoised_img_fpath)

        # Load the denoised image back into a PyTorch tensor
        # The 'batch=True' ensures we add back the batch dimension that was removed earlier
        denoised_image = pt_helpers.fpath_to_tensor(
            tmp_denoised_img_fpath, device=noisy_image.device, batch=True
        )

        # If we had stored out-of-range values earlier, we would add them back here
        # denoised_image = denoised_image + out_of_range_values

        # If we had normalized the image earlier, we would restore the original scale here
        # denoised_image = denoised_image * (img_max - img_min) + img_min

        # Temporary file cleanup is currently disabled to aid debugging
        # In production, these lines should be uncommented to avoid filling the temp directory
        # os.remove(tmp_input_img_fpath)
        # os.remove(tmp_denoised_img_fpath)

        # The section below is an alternative implementation using OpenCV's BM3D
        # It's kept for reference but was found not to work correctly with RGB images
        # and is therefore disabled in favor of the binary-based approach above
        # 
        # orig_dtype = noisy_image.dtype
        # # convert image to opencv dimension order (HWC instead of CHW)
        # noisy_image = np.moveaxis(noisy_image, 0, -1)
        # # convert noisy_image to uint8 for OpenCV processing
        # noisy_image = (noisy_image * 255).astype(np.uint8)
        # 
        # denoised_image = cv2.xphoto.bm3dDenoising(src=noisy_image, h=float(self.sigma))
        # denoised_image = (
        #     torch.from_numpy(denoised_image).to(dtype=orig_dtype) / 255
        # ).unsqueeze(0)

        return denoised_image


# Dictionary mapping architecture names to their implementation classes
# This enables dynamic model selection based on configuration parameters
architectures = {"bm3d": BM3D_Denoiser}

# Command-line interface for direct usage of the BM3D denoiser
if __name__ == "__main__":
    """Command-line interface for the BM3D denoiser.
    
    This script provides a direct way to denoise images using the BM3D algorithm
    from the command line, without requiring the full training framework.
    
    Usage:
        python bm3d_denoiser.py <noisy_image_fpath> <sigma> <denoised_fpath>
        
    Arguments:
        noisy_image_fpath: Path to the input noisy image
        sigma: Noise standard deviation parameter for BM3D
        denoised_fpath: Path where the denoised output will be saved
        
    The script loads the noisy image, runs BM3D denoising with the specified
    sigma value, and saves the result to the specified output path using the
    lin_rec2020 color profile for HDR image saving.
    """
    assert len(sys.argv) == 4, (
        f"Usage: python {sys.argv[0]} <noisy_image_fpath> <sigma> <denoised_fpath>"
    )
    noisy_image = pt_helpers.fpath_to_tensor(sys.argv[1])
    sigma = sys.argv[2]
    denoiser = architectures["bm3d"](in_channels=3, funit=sigma)
    denoised_image = denoiser(noisy_image)
    raw.hdr_nparray_to_file(
        denoised_image.squeeze(0).numpy(), sys.argv[3], color_profile="lin_rec2020"
    )
