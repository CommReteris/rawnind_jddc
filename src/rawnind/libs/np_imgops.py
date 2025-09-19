# cv2.setNumThreads(0)
"""NumPy-based image operations for loading, manipulating, and saving images.

This module provides utilities for working with images in NumPy array format, with
support for various image formats including TIFF, JPEG, PNG, and RAW files. The module
prioritizes flexibility and precision in image handling while maintaining a simple
interface.

Key features:
- Dual backend support (OpenImageIO preferred, OpenCV as fallback)
- Loading images with metadata extraction
- Padding and cropping operations for image pairs
- Support for floating-point precision (ideal for HDR images)
- Channel-first tensor format (C,H,W) for compatibility with deep learning frameworks
- Normalization to [0,1] range regardless of input bit depth

The module is designed to work with the raw image processing pipeline and supports
both regular RGB images and specialized formats like raw camera files.

Backend handling:
- OpenImageIO: Preferred for broad format support, especially 16-bit float TIFFs
- OpenCV: Used as fallback with limitations for specialized formats
- Raw image handling: Delegated to the raw module for camera-specific formats

Usage examples:
    # Load an image as a normalized float array (channels first)
    img = img_fpath_to_np_flt('image.tif')  # Returns shape (C,H,W) in range [0,1]
    
    # Load with metadata
    img, metadata = img_fpath_to_np_flt('image.raw', incl_metadata=True)
    
    # Pad two images to the same size
    img1_padded, img2_padded = np_pad_img_pair(img1, img2, target_size=256)
    
    # Randomly crop two images to the same size
    crop1, crop2 = np_crop_img_pair(img1, img2, crop_size=128, crop_method=CropMethod.RAND)
    
    # Save a numpy array as an image
    np_to_img(processed_array, 'output.png', precision=16)
"""
import os
import random
import sys
import unittest
from enum import Enum, auto
from typing import Tuple, Union

# import multiprocessing
# multiprocessing.set_start_method('spawn')
import cv2
import numpy as np

try:
    import OpenImageIO as oiio

    TIFF_PROVIDER = "OpenImageIO"
except ImportError:
    TIFF_PROVIDER = "OpenCV"
    print(
        "np_imgops.py warning: missing OpenImageIO library; falling back to OpenCV which cannot open 16-bit float tiff images"
    )

from rawnind.libs import raw
from rawnind.libs import libimganalysis

# Directory for temporary files used in testing
TMP_DPATH = "tmp"


class CropMethod(Enum):
    """Cropping method options for image pair cropping.
    
    This enum defines the available strategies for cropping image pairs:
    
    Attributes:
        RAND: Random cropping - selects a random region of the specified size
              from the input images, useful for data augmentation during training
        CENTER: Center cropping - crops the center region of the specified size,
                useful for consistent evaluation and testing
    """
    RAND = auto()  # Random crop location
    CENTER = auto()  # Center crop location


def _oiio_img_fpath_to_np(fpath: str):
    """Load an image using OpenImageIO and convert to channel-first NumPy array.
    
    This private helper function uses OpenImageIO to load images, which provides
    better support for specialized formats like 16-bit float TIFFs compared to OpenCV.
    
    Args:
        fpath: Path to the image file
        
    Returns:
        np.ndarray: Image as a NumPy array with shape (C,H,W)
    
    Notes:
        - Moves channels from last dimension to first (HWC -> CHW)
        - Preserves original precision and range of the image data
        - Handles multi-channel images with arbitrary number of channels
    """
    inp = oiio.ImageInput.open(fpath)
    spec = inp.spec()
    # Read the entire image with original channel count and format
    pixels = inp.read_image(0, 0, 0, spec.nchannels, spec.format)
    # Move channels to first dimension for deep learning framework compatibility
    pixels = np.moveaxis(pixels, -1, 0)
    inp.close()
    return pixels


def _opencv_img_fpath_to_np(fpath: str):
    """Load an image using OpenCV and convert to channel-first RGB NumPy array.
    
    This private helper function serves as a fallback when OpenImageIO is not
    available. It loads the image with OpenCV and converts from BGR to RGB format,
    then transposes to channel-first order.
    
    Args:
        fpath: Path to the image file
        
    Returns:
        np.ndarray: Image as a NumPy array with shape (C,H,W) in RGB order
        
    Raises:
        ValueError: If OpenCV fails to read the image, suggesting to install
                   OpenImageIO as an alternative
    
    Notes:
        - OpenCV loads images in BGR order; this function converts to RGB
        - Uses IMREAD_ANYDEPTH flag to preserve bit depth (8/16-bit)
        - Limited support for specialized formats compared to OpenImageIO
    """
    try:
        return cv2.cvtColor(
            cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH),
            cv2.COLOR_BGR2RGB,
        ).transpose(2, 0, 1)  # HWC -> CHW format
    except cv2.error as e:
        raise ValueError(
            f"img_fpath_to_np_flt: error {e} with {fpath} (hint: consider installing OpenImageIO instead of OpenCV backend)"
        )


def img_fpath_to_np_flt(
        fpath: str,
        incl_metadata=False,  # , bit_depth: Optional[int] = None
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Load an image file into a normalized NumPy array with optional metadata.
    
    This function serves as the primary interface for loading images from various formats
    into a standardized NumPy representation. It supports multiple image formats including
    TIFF, JPEG, PNG, and RAW files, and handles different bit depths transparently.
    
    Args:
        fpath: Path to the image file to load
        incl_metadata: If True, returns a tuple of (image_array, metadata_dict);
                       if False, returns only the image array
        
    Returns:
        Union[np.ndarray, Tuple[np.ndarray, dict]]:
            - If incl_metadata=False: NumPy array with shape (C,H,W) in range [0,1]
            - If incl_metadata=True: Tuple of (NumPy array, metadata dictionary)
            
    Raises:
        ValueError: If the file does not exist or has an unknown format
        TypeError: If the image has an unsupported data type
        
    Notes:
        - All images are normalized to [0,1] range regardless of input bit depth
        - Channel order is maintained as RGB (or original color space)
        - The function automatically detects and uses the appropriate backend:
          * RAW files: Uses raw.raw_fpath_to_rggb_img_and_metadata
          * TIFF: Uses OpenImageIO if available, falls back to OpenCV
          * Other formats: Uses OpenCV
        - Output array is always in float32 precision
    """
    if not os.path.isfile(fpath):
        raise ValueError(f"File not found {fpath}")
    if fpath.endswith(".npy"):
        assert not incl_metadata
        return np.load(fpath)
    if libimganalysis.is_raw(fpath):
        rggb_img, metadata = raw.raw_fpath_to_rggb_img_and_metadata(fpath)
        if incl_metadata:
            return rggb_img, metadata
        print("img_fpath_to_np_flt warning: ignoring raw image metadata")
        return rggb_img
    if (
            fpath.lower().endswith(".tif") or fpath.lower().endswith(".tiff")
    ) and TIFF_PROVIDER == "OpenImageIO":
        rgb_img = _oiio_img_fpath_to_np(fpath)
    else:
        rgb_img = _opencv_img_fpath_to_np(fpath)

    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float16:
        res = rgb_img
        # if bit_depth is None:
        #     return rgb_img
        # elif bit_depth == 16:
        #     return rgb_img.astype(np.float16)
        # elif bit_depth == 32:
        #     return rgb_img.astype(np.float32)
    elif rgb_img.dtype == np.ubyte:
        res = rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        res = rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            f"img_fpath_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )
    if incl_metadata:
        return res, {}
    else:
        return res


def np_pad_img_pair(img1, img2, cs):
    """Pad two images to the same target size with center alignment.
    
    This function pads both input images to reach the specified target size, 
    maintaining their alignment by centering the original content. The padding
    is applied equally on all sides where possible.
    
    Args:
        img1: First image array with shape (C,H,W)
        img2: Second image array with shape (C,H,W)
        cs: Target size (both height and width) for the padded images
        
    Returns:
        tuple: (padded_img1, padded_img2) where both images have dimensions (C,cs,cs)
        
    Notes:
        - Padding is applied only to spatial dimensions (H,W), not to channels
        - Padding is distributed evenly on both sides of each dimension
        - If the original image is larger than the target size in any dimension, 
          no padding is applied to that dimension
        - Uses zeros for padding values
    """
    # Calculate padding for width (x-axis)
    xpad0 = max(0, (cs - img1.shape[2]) // 2)  # Left padding
    xpad1 = max(0, cs - img1.shape[2] - xpad0)  # Right padding

    # Calculate padding for height (y-axis)
    ypad0 = max(0, (cs - img1.shape[1]) // 2)  # Top padding
    ypad1 = max(0, cs - img1.shape[1] - ypad0)  # Bottom padding

    # Create padding tuple for numpy.pad - format ((before_axis1, after_axis1), ...)
    # No padding for channel dimension, only for height and width
    padding = ((0, 0), (ypad0, ypad1), (xpad0, xpad1))

    # Apply same padding to both images and return
    return np.pad(img1, padding), np.pad(img2, padding)


def np_crop_img_pair(img1, img2, cs: int, crop_method=CropMethod.RAND):
    """Crop a pair of images to the specified size using the same crop region.
    
    This function extracts matching regions from two images, ensuring both
    crops come from the same spatial location. The crop location can be either
    random (for data augmentation) or centered.
    
    Args:
        img1: First image array with shape (C,H,W)
        img2: Second image array with shape (C,H,W)
        cs: Crop size (both height and width)
        crop_method: Method to determine crop location (RAND or CENTER)
        
    Returns:
        tuple: (cropped_img1, cropped_img2) where both have dimensions (C,cs,cs)
        
    Notes:
        - Assumes both input images have the same spatial dimensions
        - Compatible with both NumPy arrays and PyTorch tensors
        - For random crops, provides different crops on each call
        - For centered crops, provides consistent results across calls
    """
    # Determine crop starting coordinates based on the specified method
    if crop_method is CropMethod.RAND:
        # Random crop: select random starting point that allows for full crop size
        x0 = random.randint(0, img1.shape[2] - cs)
        y0 = random.randint(0, img1.shape[1] - cs)
    elif crop_method is CropMethod.CENTER:
        # Center crop: calculate starting point for centered crop
        x0 = (img1.shape[2] - cs) // 2
        y0 = (img1.shape[1] - cs) // 2

    # Extract the same region from both images
    return img1[:, y0: y0 + cs, x0: x0 + cs], img2[:, y0: y0 + cs, x0: x0 + cs]


def np_to_img(img: np.ndarray, fpath: str, precision: int = 16):
    """Save a NumPy array as an image file with specified bit precision.
    
    This function converts a channel-first (C,H,W) NumPy array to an image file,
    handling color space conversion, bit depth adjustment, and format selection
    based on the output path.
    
    Args:
        img: NumPy array with shape (C,H,W) or (H,W) in range [0,1]
        fpath: Output file path (extension determines format)
        precision: Bit depth for the output image:
                   - 8 for 8-bit (0-255, standard formats)
                   - 16 for 16-bit (0-65535, suitable for TIFF/PNG)
                   
    Raises:
        NotImplementedError: If precision is not 8 or 16
        
    Notes:
        - Automatically handles single-channel images by adding a channel dimension
        - Converts from RGB to BGR color order for OpenCV compatibility
        - Scales values from [0,1] to [0,255] or [0,65535] based on precision
        - Uses OpenCV for saving, which supports various formats based on extension
    """
    # Handle single-channel (H,W) images by adding channel dimension
    if len(img.shape) == 2:
        img = np.expand_dims(img, 0)

    # Convert from CHW to HWC format for OpenCV
    hwc_img = img.transpose(1, 2, 0)

    # Convert from RGB to BGR color order for OpenCV
    hwc_img = cv2.cvtColor(hwc_img, cv2.COLOR_RGB2BGR)

    # Scale and convert to appropriate bit depth
    if precision == 16:
        # 16-bit: scale to [0, 65535] and convert to uint16
        hwc_img = (hwc_img * 65535).clip(0, 65535).astype(np.uint16)
    elif precision == 8:
        # 8-bit: scale to [0, 255] and convert to uint8
        hwc_img = (hwc_img * 255).clip(0, 255).astype(np.uint8)
    else:
        raise NotImplemented(precision)

    # Save the image using OpenCV
    cv2.imwrite(fpath, hwc_img)


class TestImgOps(unittest.TestCase):
    """Unit tests for NumPy-based image operations.
    
    This test suite verifies the functionality of the image operations in the module,
    focusing on padding, cropping, and image loading. It tests:
    1. Padding images to a target size
    2. Cropping images with different methods (random and center)
    3. Loading images with different backends (OpenCV and OpenImageIO)
    
    The tests use both even and odd-sized images to ensure proper handling of
    edge cases, and verify both the dimensions and content of the processed images.
    """

    def setUp(self):
        """Set up test images and fixtures.
        
        Creates:
        1. Even-sized test images (8x8)
        2. Odd-sized test images (5x5)
        3. A random test image saved as a TIFF file
        
        Note: Requires tifffile package for saving test images.
        """
        import tifffile

        # Create even-sized test images (8x8)
        self.imgeven1 = np.random.rand(3, 8, 8)
        self.imgeven2 = np.random.rand(3, 8, 8)

        # Create odd-sized test images (5x5)
        self.imgodd1 = np.random.rand(3, 5, 5)
        self.imgodd2 = np.random.rand(3, 5, 5)

        # Create a larger test image and save it as a TIFF file
        self.random_image = np.random.rand(3, 512, 768).astype(np.float32)
        self.random_image_fpath = os.path.join(TMP_DPATH, "rand.tiff")
        os.makedirs(TMP_DPATH, exist_ok=True)
        tifffile.imwrite(self.random_image_fpath, self.random_image.transpose(1, 2, 0))

    def tearDown(self):
        """Clean up test fixtures by removing temporary files."""
        os.remove(self.random_image_fpath)

    def test_pad(self):
        """Test image padding functionality.
        
        Verifies:
        1. Even-sized images are correctly padded to target size
        2. Odd-sized images are correctly padded to target size
        3. Original image content is preserved at the correct position in padded image
        """
        # Pad images to size 16x16
        imgeven1_padded, imgeven2_padded = np_pad_img_pair(
            self.imgeven1, self.imgeven2, 16
        )
        imgodd1_padded, imgodd2_padded = np_pad_img_pair(self.imgodd1, self.imgodd2, 16)

        # Verify the dimensions of padded images
        self.assertTupleEqual(imgeven1_padded.shape, (3, 16, 16), imgeven1_padded.shape)
        self.assertTupleEqual(imgodd2_padded.shape, (3, 16, 16), imgodd2_padded.shape)

        # Verify the original content is preserved at the correct position
        # For 8x8 image centered in 16x16 result, pixel (0,0) should be at (4,4)
        self.assertEqual(imgeven1_padded[0, 4, 4], self.imgeven1[0, 0, 0])

    def test_crop(self):
        """Test image cropping functionality.
        
        Verifies:
        1. Random crops produce correct output dimensions
        2. Center crops produce correct output dimensions and content
        3. Cropping to the original size returns the original image unchanged
        """
        # Test random crop: verify dimensions
        imgeven1_randcropped, imgeven2_randcropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 4, CropMethod.RAND
        )
        self.assertTupleEqual(
            imgeven1_randcropped.shape, (3, 4, 4), imgeven1_randcropped.shape
        )

        # Test center crop: verify dimensions and content
        imgeven1_centercropped, imgeven2_centercropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 4, CropMethod.CENTER
        )
        self.assertTupleEqual(
            imgeven1_centercropped.shape, (3, 4, 4), imgeven1_centercropped.shape
        )

        # For 8x8 image with center crop of 4x4, the crop starts at (2,2)
        # Visual representation:
        # orig:    0 1 2 3 4 5 6 7
        # cropped: x x 2 3 4 5 x x
        self.assertEqual(
            imgeven1_centercropped[0, 0, 0],
            self.imgeven1[0, 2, 2],
            f"imgeven1_centercropped[0]={imgeven1_centercropped[0]}, self.imgeven1[0]={self.imgeven1[0]}",
        )

        # Test cropping to original size: should return identical image
        imgeven1_fullcropped, imgeven2_fullcropped = np_crop_img_pair(
            self.imgeven1, self.imgeven2, 8, CropMethod.CENTER
        )
        self.assertTrue(
            (imgeven1_fullcropped == self.imgeven1).all(), "Crop to same size is broken"
        )

    def test_read_img_opencv_equals_oiio(self):
        """Test consistency between different image loading backends.
        
        Verifies:
        1. OpenCV and OpenImageIO produce identical results when loading the same image
        2. The loaded images match the original image used to create the test file
        3. The default image loading function returns the same result as the backends
        """
        # Load the same image with different backends
        cvimg = _opencv_img_fpath_to_np(self.random_image_fpath)
        oiioimg = _oiio_img_fpath_to_np(self.random_image_fpath)
        default_img = img_fpath_to_np_flt(self.random_image_fpath, incl_metadata=False)

        # Verify that all loaded versions match each other and the original
        self.assertTrue(
            (cvimg == oiioimg).all(), "OpenCV and OpenImageIO images do not match"
        )
        self.assertTrue(
            (cvimg == self.random_image).all(), "OpenCV and random image do not match"
        )
        self.assertTrue(
            (cvimg == default_img).all(), "OpenCV and default image do not match"
        )


if __name__ == "__main__":
    unittest.main()
