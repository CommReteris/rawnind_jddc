import torch
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

# Import the refactored raw_processing module
from . import raw_processing as raw
from . import image_analysis as libimganalysis
from .raw_processing import ProcessingConfig

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
        # Check if the file actually exists for OpenCV to read
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"File not found for OpenCV: {fpath}")

        img_bgr_hwc = cv2.imread(fpath, flags=cv2.IMREAD_COLOR + cv2.IMREAD_ANYDEPTH)
        if img_bgr_hwc is None: # imread returns None if file cannot be read
            raise ValueError(f"OpenCV failed to read image {fpath}. Might be an unsupported format or corrupted.")

        img_rgb_hwc = cv2.cvtColor(img_bgr_hwc, cv2.COLOR_BGR2RGB)
        return img_rgb_hwc.transpose(2, 0, 1)  # HWC -> CHW format
    except Exception as e:
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
          * RAW files: Uses raw.RawLoader().load_raw_data()
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
        # Use the refactored RawLoader class from raw_processing
        raw_loader = raw.RawLoader(config=ProcessingConfig())
        rggb_img, metadata_nt = raw_loader.load_raw_data(fpath, return_float=True)
        if incl_metadata:
            # Convert Metadata NamedTuple to dict for consistency with legacy metadata handling expected by some parts of codebase
            return rggb_img, metadata_nt._asdict() # Returns dict here
        return rggb_img
    if (
            fpath.lower().endswith(".tif") or
            fpath.lower().endswith(".tiff") or
            fpath.lower().endswith(".exr")
    ) and TIFF_PROVIDER == "OpenImageIO":
        rgb_img = _oiio_img_fpath_to_np(fpath)
    else:
        rgb_img = _opencv_img_fpath_to_np(fpath)

    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float16:
        res = rgb_img
    elif rgb_img.dtype == np.ubyte:
        res = rgb_img.astype(np.single) / 255
    elif rgb_img.dtype == np.ushort:
        res = rgb_img.astype(np.single) / 65535
    else:
        raise TypeError(
            f"img_fpath_to_np_flt: Error: fpath={fpath} has unknown format ({rgb_img.dtype})"
        )
    if incl_metadata:
        return res, {} # Return empty dict as metadata if not raw
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
    else:
        raise NotImplementedError(f"Unsupported crop method: {crop_method}")

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
    # Handle single-channel (H,W) or (1,H,W) images by converting to 3 channels for saving (e.g., as grayscale RGB)
    if len(img.shape) == 2: # (H,W)
        img = np.expand_dims(img, 0) # -> (1, H, W)

    if img.shape[0] == 1: # (1, H, W) -> (3, H, W) by replicating channel
        img = np.tile(img, (3, 1, 1))

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
        raise NotImplementedError(precision) # Corrected to raise NotImplementedError

    # Save the image using OpenCV
    cv2.imwrite(fpath, hwc_img)

# Utility Functions
def np_l1(img1: np.ndarray, img2: np.ndarray, avg: bool = True) -> Union[float, np.ndarray]:
    '''Compute per-element L1 distance between two images.

    Args:
        img1: First image (NumPy array) of identical shape as img2.
        img2: Second image (NumPy array) of identical shape as img1.
        avg: If True, return the mean L1 value over all elements; otherwise return the element-wise map.

    Returns:
        A scalar float if avg is True, otherwise a NumPy array of absolute differences with the same shape as inputs.
    '''
    if avg:
        return np.abs(img1 - img2).mean()
    return np.abs(img1 - img2)


def gamma(img: np.ndarray, gamma_val: float = 2.2, in_place: bool = False) -> np.ndarray:
    '''Apply gamma correction to a NumPy image.

    Only strictly positive values are gamma-encoded; non-positive values are preserved
    as-is to avoid creating NaNs when operating on linear-light data that may contain
    small negative values (e.g., after filtering).

    Args:
        img: Input NumPy array. Broadcastable operations are applied element-wise.
        gamma_val: Gamma exponent to apply (default 2.2). Effective transform is x**(1/gamma).
        in_place: If True, modify the input array in place; otherwise operate on a copy.

    Returns:
        NumPy array with gamma applied to positive entries.
    '''
    res = img if in_place else img.copy()
    res[res > 0] = res[res > 0] ** (1 / gamma_val)
    return res

def scenelin_to_pq(
        img: Union[np.ndarray, torch.Tensor], compat=True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Scene linear input signal to PQ opto-electronic transfer function (OETF).
    This function depends on the `colour-science` library.
    """
    try:
        import colour
    except ImportError:
        raise ImportError("The 'colour-science' library is required for scenelin_to_pq. Please install it.")

    if isinstance(img, np.ndarray):
        return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ(img)
    elif isinstance(img, torch.Tensor):
        # The torch implementation requires internals of colour, which are not directly exposed.
        # This part should ideally be contained within pytorch_operations.
        # For now, if called for a torch tensor, raise an error indicating it's not implemented here.
        raise NotImplementedError("PyTorch Tensor support for scenelin_to_pq for numpy_operations is not implemented here. Use pytorch_operations' equivalent.")
    else:
        raise NotImplementedError(f"Unsupported image type for scenelin_to_pq: {type(img)=}")


def pq_to_scenelin(
        img: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    PQ non-linear to scene linear signal, inverse opto-electronic transfer function (OETF^-1).
    This function depends on the `colour-science` library.
    """
    try:
        import colour
    except ImportError:
        raise ImportError("The 'colour-science' library is required for pq_to_scenelin. Please install it.")

    if isinstance(img, np.ndarray):
        return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_inverse_PQ_BT2100(img)
    elif isinstance(img, torch.Tensor):
        raise NotImplementedError("PyTorch Tensor support for pq_to_scenelin for numpy_operations is not implemented here. Use pytorch_operations' equivalent.")
    else:
        raise NotImplementedError(f"Unsupported image type for pq_to_scenelin: {type(img)=}")


def get_leaf(fpath: str) -> str:
    """Extract the filename (leaf) from a file path.

    Args:
        fpath: File path

    Returns:
        Filename component of the path
    """
    return os.path.basename(fpath)


def match_gain(
        anchor_img: Union[np.ndarray, torch.Tensor],
        other_img: Union[np.ndarray, torch.Tensor],
        return_val: bool = False,
) -> Union[np.ndarray, torch.Tensor, float]:
    '''Match average intensity (gain) between two images.

    Supports single images shaped [C,H,W] and batched images shaped [N,C,H,W].
    '''
    # Basic implementation for numpy arrays
    if isinstance(anchor_img, np.ndarray) and isinstance(other_img, np.ndarray):
        anchor_mean = np.mean(anchor_img)
        other_mean = np.mean(other_img)
        if other_mean == 0:
            gain = 1.0 # Avoid division by zero, no change
        else:
            gain = anchor_mean / other_mean

        if return_val:
            return float(gain)
        else:
            return other_img * gain
    # Placeholder for torch tensors, or more complex cases
    raise NotImplementedError("match_gain with Torch tensors or specific backend handling is not yet implemented.")
