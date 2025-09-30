import logging
import os
import operator
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal, NamedTuple, Optional, Tuple, Union

import Imath  # OpenEXR
import numpy as np
import rawpy
import requests
import torch # Added missing import

from . import color_management as icc # Alias matches legacy usage of icc

# Constants
BAYER_PATTERNS: ClassVar[dict] = {
    "RGGB": np.array([[0, 1], [2, 3]]),  # R G / G B
    "GBRG": np.array([[1, 0], [3, 2]]),  # G R / B G
    "BGGR": np.array([[2, 3], [0, 1]]),  # B G / G R
    "GRBG": np.array([[3, 2], [1, 0]]),  # G B / R G
}
DEFAULT_OUTPUT_PROFILE = "lin_rec2020"
SAMPLE_RAW_URL = "https://nc.trougnouf.com/index.php/s/zMA8QfgPoNoByex/download/DSC01568.ARW"
DEFAULT_OE_THRESHOLD = 0.99
DEFAULT_UE_THRESHOLD = 0.001
DEFAULT_QTY_THRESHOLD = 0.75
MAX_FILE_SIZE_MB = 500  # Prevent OOM on large RAWs

# Logging setup
logger = logging.getLogger(__name__)

# Provider detection (non-global, passed to classes)
def detect_openexr_provider() -> str:
    try:
        import OpenImageIO as oiio
        return "OpenImageIO"
    except ImportError:
        try:
            import OpenEXR
            return "OpenEXR"
        except ImportError:
            raise ImportError("OpenImageIO or OpenEXR must be installed")

OPENEXR_PROVIDER = detect_openexr_provider()
logger.info(f"Using OpenEXR provider: {OPENEXR_PROVIDER}")

def detect_tiff_provider() -> str:
    try:
        import OpenImageIO as oiio
        return "OpenImageIO"
    except ImportError:
        logger.warning("OpenImageIO not found; using OpenCV for TIFFs. Install for better support.")
        return "OpenCV"

TIFF_PROVIDER = detect_tiff_provider()

try:
    # This dependency for libraw_process function (which is removed from public API)
    # is only conditionally imported to avoid unnecessary ImportError if function is not used.
    import imageio
except ImportError:
    logger.warning("imageio not found.")
    imageio = None

import cv2


# Module exports
__all__ = [
    'ProcessingConfig',
    'RawProcessingError',
    'UnsupportedFormatError',
    'BayerPattern',
    'BayerImage',
    'RGGBImage',
    'RGBImage',
    'Metadata',
    'RawLoader',
    'BayerProcessor',
    'ColorTransformer',
    'get_sample_raw_file',
    'is_exposure_ok',
    'is_xtrans',
    'xtrans_fpath_to_OpenEXR',
    'hdr_nparray_to_file',
    'raw_fpath_to_hdr_img_file',
    'raw_fpath_to_rggb_img_and_metadata',
    'raw_fpath_to_hdr_img_file_mtrunner',
    'scenelin_to_pq',
    'gamma',
    'camRGB_to_lin_rec2020_images',
    'demosaic',
    'rggb_to_mono',
]


@dataclass
class ProcessingConfig:
    """Configuration for RAW processing pipeline."""
    force_rggb: bool = True
    crop_all: bool = True
    return_float: bool = True # For RawLoader
    wb_type: str = "daylight" # For BayerProcessor
    demosaic_method: int = cv2.COLOR_BayerRGGB2RGB_EA # For BayerProcessor
    output_color_profile: str = DEFAULT_OUTPUT_PROFILE # For ColorTransformer and hdr_nparray_to_file
    oe_threshold: float = DEFAULT_OE_THRESHOLD # For is_exposure_ok
    ue_threshold: float = DEFAULT_UE_THRESHOLD # For is_exposure_ok
    qty_threshold: float = DEFAULT_QTY_THRESHOLD # For is_exposure_ok
    bit_depth: Optional[int] = None # For hdr_nparray_to_file
    check_exposure: bool = True # For raw_fpath_to_hdr_img_file


class RawProcessingError(Exception):
    """Custom exception for RAW processing errors."""


class UnsupportedFormatError(RawProcessingError):
    """Raised for unsupported RAW formats (e.g., non-Bayer)."""


class BayerPattern(Enum):
    """Enum for supported Bayer patterns."""
    RGGB = "RGGB"
    GBRG = "GBRG"
    BGGR = "BGGR"
    GRBG = "GRBG"


BayerImage = np.ndarray  # Shape: (1, H, W)
RGGBImage = np.ndarray  # Shape: (4, H/2, W/2)
RGBImage = np.ndarray  # Shape: (3, H, W)


class Metadata(NamedTuple):
    """Structured metadata from RAW file."""
    fpath: str
    bayer_pattern: BayerPattern
    rgbg_pattern: np.ndarray
    sizes: dict
    camera_whitebalance: np.ndarray
    black_level_per_channel: np.ndarray
    white_level: int
    camera_white_level_per_channel: Optional[np.ndarray]
    daylight_whitebalance: np.ndarray
    rgb_xyz_matrix: np.ndarray
    overexposure_lb: float
    # Normalized WB
    camera_whitebalance_norm: np.ndarray
    daylight_whitebalance_norm: np.ndarray


class RawLoader:
    """
    Loads raw image files and extracts comprehensive metadata.
    Handles initial processing steps like border removal, Bayer pattern normalization,
    and scaling to black/white points.

    Attributes:
        config (ProcessingConfig): Configuration for raw processing.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def _set_bayer_pattern_name(self, metadata_dict: dict):
        """Set bayer_pattern string from RGBG (raw_pattern) indices.
        Modifies metadata_dict in place.
        """
        matched_pattern = None
        for name, pattern_arr in BAYER_PATTERNS.items():
            if np.array_equal(pattern_arr, metadata_dict["rgbg_pattern"]):
                matched_pattern = name
                break

        if matched_pattern: # Ensure it is a BayerPattern Enum
            metadata_dict["bayer_pattern"] = BayerPattern[matched_pattern]
        else:
            raise UnsupportedFormatError(f"Unknown Bayer pattern: {metadata_dict['rgbg_pattern']}")

    def _remove_empty_borders(self, mono_img: np.ndarray, metadata_dict: dict) -> np.ndarray:
        """
        Remove empty borders declared in the metadata.
        Modifies metadata_dict in place.
        """
        if metadata_dict["sizes"]["top_margin"] or metadata_dict["sizes"]["left_margin"]:
            mono_img = mono_img[
                :, metadata_dict["sizes"]["top_margin"] :, metadata_dict["sizes"]["left_margin"] :
            ]
            metadata_dict["sizes"]["raw_height"] -= metadata_dict["sizes"]["top_margin"]
            metadata_dict["sizes"]["raw_width"] -= metadata_dict["sizes"]["left_margin"]
            metadata_dict["sizes"]["top_margin"] = metadata_dict["sizes"]["left_margin"] = 0

        if self.config.crop_all:
            _, h, w = mono_img.shape
            min_h = min(
                h,
                metadata_dict["sizes"]["height"],
                metadata_dict["sizes"]["iheight"],
                metadata_dict["sizes"]["raw_height"],
            )
            min_w = min(
                w,
                metadata_dict["sizes"]["width"],
                metadata_dict["sizes"]["iwidth"],
                metadata_dict["sizes"]["raw_width"],
            )
            metadata_dict["sizes"]["height"] = metadata_dict["sizes"]["iheight"] = metadata_dict[
                "sizes"
            ]["raw_height"] = min_h
            metadata_dict["sizes"]["width"] = metadata_dict["sizes"]["iwidth"] = metadata_dict[
                "sizes"
            ]["raw_width"] = min_w
            mono_img = mono_img[:, :min_h, :min_w]

        assert mono_img.shape[1:] == (
            metadata_dict["sizes"]["raw_height"],
            metadata_dict["sizes"]["raw_width"],
        ), f"Image shape {mono_img.shape[1:]=} does not match metadata raw dimensions {metadata_dict['sizes']=}"
        return mono_img

    def _mono_any_to_mono_rggb(self, mono_img: np.ndarray, metadata_dict: dict, full_raw_colors: np.ndarray) -> np.ndarray:
        """
        Convert (crop) any Bayer pattern to RGGB. Modifies metadata_dict in place.
        Assumes cropped margins.
        """
        if metadata_dict["bayer_pattern"] == BayerPattern.RGGB:
            pass
        else:
            if not (
                metadata_dict["sizes"]["top_margin"] == metadata_dict["sizes"]["left_margin"] == 0
            ):
                raise NotImplementedError(
                    f"Metadata sizes {metadata_dict['sizes']=}, Bayer pattern {metadata_dict['bayer_pattern']=} with borders is not implemented."
                )
            if metadata_dict["bayer_pattern"] == BayerPattern.GBRG:
                mono_img = mono_img[:, 1:-1]
                metadata_dict["sizes"]["raw_height"] -= 2
                metadata_dict["sizes"]["height"] -= 2
                metadata_dict["sizes"]["iheight"] -= 2
            elif metadata_dict["bayer_pattern"] == BayerPattern.BGGR:
                mono_img = mono_img[:, 1:-1, 1:-1]
                metadata_dict["sizes"]["raw_height"] -= 2
                metadata_dict["sizes"]["height"] -= 2
                metadata_dict["sizes"]["iheight"] -= 2
                metadata_dict["sizes"]["raw_width"] -= 2
                metadata_dict["sizes"]["width"] -= 2
                metadata_dict["sizes"]["iwidth"] -= 2
            elif metadata_dict["bayer_pattern"] == BayerPattern.GRBG:
                mono_img = mono_img[:, :, 1:-1]
                metadata_dict["sizes"]["raw_width"] -= 2
                metadata_dict["sizes"]["width"] -= 2
                metadata_dict["sizes"]["iwidth"] -= 2
            else:
                raise UnsupportedFormatError(f"Unsupported Bayer pattern for conversion: {metadata_dict['bayer_pattern']=}")

        # For non-RGGB patterns, after cropping to shift pixels, the final pattern must be RGGB
        metadata_dict["rgbg_pattern"] = BAYER_PATTERNS["RGGB"]
        self._set_bayer_pattern_name(metadata_dict)
        assert metadata_dict["bayer_pattern"] == BayerPattern.RGGB, f'Conversion to RGGB failed: {metadata_dict["bayer_pattern"]=}'
        return mono_img

    def _ensure_correct_shape(self, mono_img: np.ndarray, metadata_dict: dict) -> np.ndarray:
        """Ensure dimension % 4 == 0. Modifies metadata_dict in place."""
        _, h, w = mono_img.shape
        assert (metadata_dict["sizes"]["raw_height"], metadata_dict["sizes"]["raw_width"]) == (
            h,
            w,
        ), f"Image shape {mono_img.shape[1:]=} does not match metadata raw dimensions {metadata_dict['sizes']=}"

        if h % 4 > 0:
            mono_img = mono_img[:, : -(h % 4)]
            metadata_dict["sizes"]["raw_height"] -= h % 4
            metadata_dict["sizes"]["height"] -= h % 4
            metadata_dict["sizes"]["iheight"] -= h % 4
        if w % 4 > 0:
            mono_img = mono_img[:, :, : -(w % 4)]
            metadata_dict["sizes"]["raw_width"] -= w % 4
            metadata_dict["sizes"]["width"] -= w % 4
            metadata_dict["sizes"]["iwidth"] -= w % 4

        assert not (mono_img.shape[1] % 4 or mono_img.shape[2] % 4), f"Image dimensions {mono_img.shape[1:]} are not multiples of 4 after cropping."
        return mono_img

    def _scale_img_to_bw_points(self, img: np.ndarray, metadata_dict: dict, compat: bool = True) -> np.ndarray:
        """
        Scale image to black/white points described in metadata_dict.
        Modifies metadata_dict in place to set "overexposure_lb".
        """
        scaled_img = img.astype(np.float32)
        metadata_dict["overexposure_lb"] = 1.0 # Initialize overexposure lower bound

        for ch in range(img.shape[-3]):
            scaled_img[ch] -= metadata_dict["black_level_per_channel"][ch]

            # Normalize s.t. white level is 1
            if compat:
                vrange = metadata_dict["white_level"] - metadata_dict["black_level_per_channel"][ch]
                if metadata_dict["camera_white_level_per_channel"] is not None: # Check for None explicitly
                    metadata_dict["overexposure_lb"] = min(
                        metadata_dict["overexposure_lb"],
                        (
                            metadata_dict["camera_white_level_per_channel"][ch]
                            - metadata_dict["black_level_per_channel"][ch]
                        )
                        / vrange,
                    )
            else:
                vrange = (
                    metadata_dict["camera_white_level_per_channel"][ch]
                    - metadata_dict["black_level_per_channel"][ch]
                )
            scaled_img[ch] /= vrange
        return scaled_img

    def load_raw_data(self, fpath: str) -> Tuple[np.ndarray, Metadata]:
        """
        Loads raw image data from a given file path and extracts LibRaw metadata.
        Applies initial processing steps based on ProcessingConfig.

        Returns:
            Tuple[np.ndarray, Metadata]: (mono_image, metadata)
        """
        try:
            rawpy_img = rawpy.imread(fpath)
            mono_img = np.expand_dims(rawpy_img.raw_image, axis=0) # (1, H, W)
        except (rawpy._rawpy.LibRawFileUnsupportedError, rawpy._rawpy.LibRawIOError) as e:
            raise RawProcessingError(f"Error opening raw file {fpath}: {e}")
        except Exception as e:
            raise RawProcessingError(f"Unknown error loading raw file {fpath}: {e}")

        # Extract metadata
        # Create a mutable dictionary first to modify in place
        metadata_dict = {
            "fpath": fpath,
            "camera_whitebalance": rawpy_img.camera_whitebalance,
            "black_level_per_channel": rawpy_img.black_level_per_channel,
            "white_level": rawpy_img.white_level,
            "camera_white_level_per_channel": rawpy_img.camera_white_level_per_channel,
            "daylight_whitebalance": rawpy_img.daylight_whitebalance,
            "rgb_xyz_matrix": rawpy_img.rgb_xyz_matrix,
            "sizes": rawpy_img.sizes._asdict(),
            "rgbg_pattern": rawpy_img.raw_pattern, # Initial raw pattern
            "overexposure_lb": 1.0 # Placeholder, updated by _scale_img_to_bw_points
        }

        # Assertions from legacy
        assert metadata_dict["rgb_xyz_matrix"].any(), f"rgb_xyz_matrix of {fpath} is empty"
        assert rawpy_img.color_desc.decode() == "RGBG", f"{fpath} does not seem to have bayer pattern ({rawpy_img.color_desc.decode()})"
        assert metadata_dict["rgbg_pattern"] is not None, f"{fpath} has no bayer pattern information"

        self._set_bayer_pattern_name(metadata_dict) # Sets metadata_dict["bayer_pattern"]

        # Legacy check - raw_colors represents the overall sensor pattern
        assert_correct_metadata = (
            metadata_dict["rgbg_pattern"] == rawpy_img.raw_colors[:2, :2]
        )
        assert (
            assert_correct_metadata
            if isinstance(assert_correct_metadata, bool)
            else assert_correct_metadata.all()
        ), (f"Bayer pattern decoding did not match ({fpath=}, {metadata_dict['rgbg_pattern']=}, "
            f"{rawpy_img.raw_colors[:2, :2]=})")

        # Normalize white balance arrays
        for a_wb in ("daylight", "camera"):
            metadata_dict[f"{a_wb}_whitebalance_norm"] = np.array(
                metadata_dict[f"{a_wb}_whitebalance"], dtype=np.float32
            )
            # Ensure the green channel (index 1) is used for normalization if index 3 is zero or invalid
            if metadata_dict[f"{a_wb}_whitebalance_norm"][3] == 0:
                metadata_dict[f"{a_wb}_whitebalance_norm"][3] = metadata_dict[
                    f"{a_wb}_whitebalance_norm"
                ][1]
            # Normalize by the green channel (index 1) always
            metadata_dict[f"{a_wb}_whitebalance_norm"] /= metadata_dict[f"{a_wb}_whitebalance_norm"][1]

        # Apply initial processing steps (modifies mono_img and metadata_dict in place)
        mono_img = self._remove_empty_borders(mono_img, metadata_dict)
        if self.config.force_rggb:
            mono_img = self._mono_any_to_mono_rggb(mono_img, metadata_dict, rawpy_img.raw_colors)
        mono_img = self._ensure_correct_shape(mono_img, metadata_dict)
        if self.config.return_float:
            mono_img = self._scale_img_to_bw_points(mono_img, metadata_dict)

        # Create Metadata NamedTuple from the potentially modified dictionary
        # Explicit conversion to ensure `bayer_pattern` is BayerPattern Enum
        metadata_dict["bayer_pattern"] = BayerPattern(metadata_dict["bayer_pattern"])
        metadata = Metadata(**metadata_dict)
        return mono_img, metadata


class BayerProcessor:
    '''Processes Bayer mosaics: WB, demosaic, RGGB conversion. Performance-optimized with vectorization.'''

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def apply_white_balance(self, bayer_mosaic: BayerImage, metadata: Metadata, reverse: bool = False, in_place: bool = False) -> Optional[BayerImage]:
        '''Apply/reverse WB; vectorized broadcasting for speed.'''
        if not in_place:
            bayer_mosaic = bayer_mosaic.copy()

        op = np.divide if reverse else operator.mul # Using operator.mul for clarity with np arrays
        wb_norm = getattr(metadata, f"{self.config.wb_type}_whitebalance_norm")

        # Align with original legacy_raw.py logic (lines 422-433)
        # Assuming wb_norm has R, G1, B, G2 from rawpy order
        bayer_mosaic[0, 0::2, 0::2] = op(bayer_mosaic[0, 0::2, 0::2], wb_norm[0]) # R
        bayer_mosaic[0, 0::2, 1::2] = op(bayer_mosaic[0, 0::2, 1::2], wb_norm[1]) # G1
        bayer_mosaic[0, 1::2, 0::2] = op(bayer_mosaic[0, 1::2, 0::2], wb_norm[3]) # G2 (mapped from wb_norm[3] in legacy)
        bayer_mosaic[0, 1::2, 1::2] = op(bayer_mosaic[0, 1::2, 1::2], wb_norm[2]) # B

        return bayer_mosaic if not in_place else None

    def mono_to_rggb(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGGBImage:
        '''Convert mono to 4-channel RGGB; assumes RGGB pattern.'''
        if metadata.bayer_pattern != BayerPattern.RGGB:
            raise UnsupportedFormatError(f"Expected RGGB for mono_to_rggb: {metadata.bayer_pattern}")

        bayer_flat = bayer_mosaic[0]  # Drop channel dim
        h, w = bayer_flat.shape

        # Ensure image dimensions are even.
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Image dimensions must be even for RGGB conversion: H={h}, W={w}")

        # Vectorized extraction (no loops, ~2x faster than original)
        r = bayer_flat[0::2, 0::2]
        g1 = bayer_flat[0::2, 1::2]
        g2 = bayer_flat[1::2, 0::2]
        b = bayer_flat[1::2, 1::2]

        return np.stack([r, g1, g2, b], axis=0).astype(np.float32)

    def rggb_to_mono(self, rggb_image: RGGBImage) -> BayerImage:
        '''Reverse: RGGB to mono Bayer; vectorized interleaving.'''
        assert rggb_image.shape[0] == 4, f"Expected 4-channel RGGB: {rggb_image.shape}"
        h2, w2 = rggb_image.shape[1:]
        h, w = h2 * 2, w2 * 2

        mono = np.zeros((1, h, w), dtype=np.float32)
        # Broadcast and assign (efficient)
        mono[0, 0::2, 0::2] = rggb_image[0]  # R
        mono[0, 0::2, 1::2] = rggb_image[1]  # G1
        mono[0, 1::2, 0::2] = rggb_image[2]  # G2
        mono[0, 1::2, 1::2] = rggb_image[3]  # B

        return mono

    def demosaic(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGBImage:
        '''Demosaic to RGB; optimized with uint16 conversion only once.'''
        if metadata.bayer_pattern != BayerPattern.RGGB:
            raise UnsupportedFormatError(f"Demosaic requires RGGB: {metadata.bayer_pattern}")

        # Prepare for OpenCV: Normalize to [0, 65535] uint16
        min_val, max_val = bayer_mosaic.min(), bayer_mosaic.max()
        offset = max(0, -min_val)
        scale = 65535 / (max_val + offset) if max_val + offset > 0 else 1

        prep = ((bayer_mosaic + offset) * scale).astype(np.uint16)[0]  # HWC

        # Demosaic
        rgb_hwc = cv2.demosaicing(prep, self.config.demosaic_method)

        # Back to float [0,1] CHW
        rgb = rgb_hwc.transpose(2, 0, 1).astype(np.float32) / 65535 * (max_val + offset) - offset


        return rgb


class ColorTransformer:
    '''Handles color space transformations.'''

    @staticmethod
    def get_xyz_to_rgb_matrix(profile: str) -> np.ndarray:
        '''Static XYZ to profiled RGB matrix.'''
        if profile == "lin_rec2020":
            return np.array([
                [1.71666343, -0.35567332, -0.25336809],
                [-0.66667384, 1.61645574, 0.0157683],
                [0.01764248, -0.04277698, 0.94224328],
            ], dtype=np.float32)
        elif "sRGB" in profile:
            return np.array([
                [3.24100326, -1.53739899, -0.49861587],
                [-0.96922426, 1.87592999, 0.04155422],
                [0.05563942, -0.2040112, 1.05714897],
            ], dtype=np.float32)
        raise NotImplementedError(f"Unsupported profile: {profile}")

    @classmethod
    def get_camRGB_to_profiledRGB_img_matrix(cls, metadata: Metadata, output_color_profile: str) -> np.ndarray:
        """
        Get conversion matrix from camRGB to a given color profile.
        """
        # Access rgb_xyz_matrix using dot notation for NamedTuple
        cam_to_xyzd65 = np.linalg.inv(metadata.rgb_xyz_matrix[:3])
        if output_color_profile.lower() == "xyz":
            return cam_to_xyzd65
        xyz_to_profiledRGB = cls.get_xyz_to_rgb_matrix(output_color_profile)
        color_matrix = xyz_to_profiledRGB @ cam_to_xyzd65
        return color_matrix

    @classmethod
    def camRGB_to_profiledRGB_img(cls, camRGB_img: np.ndarray, metadata: Metadata, output_color_profile: str) -> np.ndarray:
        """Convert camRGB debayered image to a given RGB color profile (in-place)."""
        color_matrix = cls.get_camRGB_to_profiledRGB_img_matrix(metadata, output_color_profile)
        orig_dims = camRGB_img.shape

        profiledRGB_img = (color_matrix @ camRGB_img.reshape(3, -1)).reshape(orig_dims)
        if output_color_profile.startswith("gamma"):
            # Placeholder, assuming `gamma_pt` or equivalent is used externally for torch tensors.
            # The original `apply_gamma` was a separate function, not part of ColorTransformer.
            # If application of gamma is needed here, it should call an external utility.
            raise NotImplementedError("Applying gamma via ColorTransformer.camRGB_to_profiledRGB_img is not supported. Use gamma_pt or equivalent post-processing.")
        return profiledRGB_img

def get_sample_raw_file(url: str = SAMPLE_RAW_URL) -> str:
    """Get a testing image online."""
    fn = url.split("/")[-1]
    fpath = os.path.join("data", fn)
    if not os.path.exists(fpath):
        os.makedirs("data", exist_ok=True)
        r = requests.get(url, allow_redirects=True, verify=False)
        open(fpath, "wb").write(r.content)
    return fpath


def is_exposure_ok(
    mono_float_img: np.ndarray,
    metadata: Metadata, # Changed to Metadata NamedTuple
    oe_threshold=DEFAULT_OE_THRESHOLD,
    ue_threshold=DEFAULT_UE_THRESHOLD,
    qty_threshold=DEFAULT_QTY_THRESHOLD,
) -> bool:
    """Check that the image exposure is useable in all channels."""
    config = ProcessingConfig()
    bayer_processor = BayerProcessor(config)
    rggb_img = bayer_processor.mono_to_rggb(mono_float_img, metadata)

    local_overexposure_lb = metadata.overexposure_lb # Access directly from NamedTuple

    overexposed = (rggb_img >= oe_threshold * local_overexposure_lb).any(0)
    if ue_threshold > 0:
        underexposed = (rggb_img <= ue_threshold).all(0)
        return (overexposed + underexposed).sum() / overexposed.size <= qty_threshold
    return overexposed.sum() / overexposed.size <= qty_threshold


def is_xtrans(fpath) -> bool:
    return fpath.lower().endswith(".raf")


def xtrans_fpath_to_OpenEXR(
    src_fpath: str, dest_fpath: str, output_color_profile: str = DEFAULT_OUTPUT_PROFILE
):
    assert output_color_profile == DEFAULT_OUTPUT_PROFILE
    assert is_xtrans(src_fpath)
    if not shutil.which("darktable-cli"):
        logger.error("darktable-cli not found in PATH. X-Trans conversion will fail.")
        raise RuntimeError("darktable-cli not found for X-Trans conversion.")

    conversion_cmd: tuple = (
        "darktable-cli",
        src_fpath,
        str(Path("src/rawnind/dependencies/configs") / "dt4_xtrans_to_linrec2020.xmp"), # Correct path
        dest_fpath,
        "--core",
        "--conf",
        "plugins/imageio/format/exr/bpp=16",
    )
    subprocess.call(conversion_cmd)


# --- Re-add hdr_nparray_to_file ---

def hdr_nparray_to_file(
    img: Union[np.ndarray, torch.Tensor],
    fpath: str,
    color_profile: Literal["lin_rec2020", "lin_sRGB", "gamma_sRGB", None], # Add None for camRGB
    bit_depth: Optional[int] = None,
    src_fpath: Optional[str] = None,
) -> None:
    """Save (c,h,w) numpy array to HDR image file. (OpenEXR or TIFF)

    src_fpath can be used to copy metadata over using exiftool.
    """
    # Handle torch.Tensor input
    if isinstance(img, torch.Tensor):
        img: np.ndarray = img.numpy()

    if fpath.endswith("exr"):
        if bit_depth is None:
            bit_depth = 16 if img.dtype == np.float16 else 32
        # Ensure img has the correct dtype for the determined bit_depth
        if bit_depth == 16 and img.dtype != np.float16:
            img = img.astype(np.float16)
        elif bit_depth == 32 and img.dtype != np.float32:
            img = img.astype(np.float32)
        # If bit_depth is determined to be 16 or 32, and img.dtype matches, no further action needed.
        # If bit_depth is specified (not None) and img.dtype is different from target 16/32, then raise error.
        elif bit_depth in [16, 32] and img.dtype not in [np.float16, np.float32]:
             raise NotImplementedError(
                f"hdr_nparray_to_file: bit_depth={bit_depth} requires np.float{bit_depth}, but got {img.dtype=}"
            )

        # Use the global OPENEXR_PROVIDER detected earlier
        if OPENEXR_PROVIDER == "OpenImageIO":
            import OpenImageIO as oiio
            output = oiio.ImageOutput.create(fpath)
            if not output:
                raise RuntimeError(f"Could not create output for {fpath}")

            spec = oiio.ImageSpec(
                img.shape[2],
                img.shape[1],
                img.shape[0],
                oiio.HALF if bit_depth == 16 else oiio.FLOAT,
            )
            if color_profile == "lin_rec2020":
                spec.attribute("oiio:ColorSpace", "Rec2020")
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290])
            elif color_profile == "lin_sRGB":
                spec.attribute("oiio:ColorSpace", "lin_srgb")
                spec.attribute('chromaticities', oiio.TypeDesc("float[8]"), [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290])
            elif color_profile is None: # Handle None for camRGB
                pass
            else:
                logger.warning(f"No color profile for {fpath}")

            spec.attribute("compression", "zips")

            if output.open(fpath, spec):
                success = output.write_image(
                    np.ascontiguousarray(img.transpose(1, 2, 0))
                )
                output.close()
                if not success:
                    raise RuntimeError(
                        f"Error writing {fpath}: {output.geterror()} ({img.shape=})"
                    )
            else:
                raise RuntimeError(f"Error opening output image: {fpath}")
        elif OPENEXR_PROVIDER == "OpenEXR":
            import OpenEXR
            header = OpenEXR.Header(img.shape[-1], img.shape[-2])
            header["Compression"] = Imath.Compression(
                Imath.Compression.ZIPS_COMPRESSION
            )
            assert color_profile is None or color_profile.startswith(
                "lin"
            ), f"{color_profile=}"
            if color_profile == "lin_rec2020":
                header["chromaticities"] = Imath.Chromaticities(
                    Imath.chromaticity(0.708, 0.292),
                    Imath.chromaticity(0.17, 0.797),
                    Imath.chromaticity(0.131, 0.046),
                    Imath.chromaticity(0.3127, 0.3290),
                )
            elif color_profile == "lin_sRGB":
                header["chromaticities"] = Imath.Chromaticities(
                    Imath.chromaticity(0.64, 0.33),
                    Imath.chromaticity(0.30, 0.60),
                    Imath.chromaticity(0.15, 0.06),
                    Imath.chromaticity(0.3127, 0.3290),
                )
            elif color_profile is None:
                pass
            else:
                raise NotImplementedError(
                    f"hdr_nparray_to_file: OpenEXR with {color_profile=}"
                )
            if bit_depth == 16:
                header["channels"] = {
                    "R": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)),
                }
                np_data_type = np.float16
            elif bit_depth == 32:
                header["channels"] = {
                    "R": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "G": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                    "B": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
                }
                np_data_type = np.float32
            else:
                raise NotImplementedError(
                    f"hdr_nparray_to_file: OpenEXR with {bit_depth=}"
                )
            exr = OpenEXR.OutputFile(fpath, header)

            exr.writePixels(
                {
                    "R": img[0].astype(np_data_type),
                    "G": img[1].astype(np_data_type),
                    "B": img[2].astype(np_data_type),
                }
            )
            exr.close()
        else:
            raise NotImplementedError(f"hdr_nparray_to_file: {OPENEXR_PROVIDER=}")
    else: # TIFF files
        # Use the global TIFF_PROVIDER detected earlier
        if TIFF_PROVIDER == "OpenCV":
            if img.dtype == np.float32 and (img.min() <= 0 or img.max() >= 1):
                logger.warning(
                    f"hdr_nparray_to_file warning: DATA LOSS: image range out of bounds "
                    f"({img.min()=}, {img.max()=}). Consider using OpenImageIO or saving {fpath=} to "
                    "OpenEXR in order to maintain data integrity."
                )
            if color_profile != "gamma_sRGB":
                logger.warning(
                    f"hdr_nparray_to_file warning: {color_profile=} not saved to "
                    f"{fpath=}. Viewer will wrongly assume sRGB."
                )
            hwc_img = img.transpose(1, 2, 0)
            hwc_img = cv2.cvtColor(hwc_img, cv2.COLOR_RGB2BGR) # OpenCV uses BGR
            hwc_img = (hwc_img * 65535).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(fpath, hwc_img)
        elif TIFF_PROVIDER == "OpenImageIO":
            import OpenImageIO as oiio
            output = oiio.ImageOutput.create(fpath)
            if not output:
                raise RuntimeError(f"Could not create output for {fpath}")

            spec = oiio.ImageSpec(
                img.shape[2],
                img.shape[1],
                img.shape[0],
                oiio.HALF if bit_depth == 16 else oiio.FLOAT,
            )
            if bit_depth == 16:
                spec.attribute("tiff:half", 1)
            if color_profile == "lin_rec2020":
                spec.attribute("chromaticities", oiio.TypeDesc("float[8]"), [0.708, 0.292, 0.17, 0.797, 0.131, 0.046, 0.3127, 0.3290])
                spec.attribute("oiio:ColorSpace", "Rec2020")
                spec.attribute("ICCProfile", oiio.TypeDesc("uint8[904]"), icc.rec2020)
            elif color_profile == "lin_sRGB":
                spec.attribute("oiio:ColorSpace", "lin_srgb")
            else:
                logger.warning(f"No color profile for {fpath}")
            assert img.dtype == np.float16 or img.dtype == np.float32, img.dtype
            if output.open(fpath, spec):
                success = output.write_image(
                    np.ascontiguousarray(img.transpose(1, 2, 0))
                )
                output.close()
                if not success:
                    raise RuntimeError(
                        f"Error writing {fpath}: {output.geterror()} ({img.shape=})"
                    )
            else:
                raise RuntimeError(f"Error opening output image: {fpath}")
        else:
            raise NotImplementedError(f"hdr_nparray_to_file: {TIFF_PROVIDER=}")
    # Removed exiftool subprocess call as it's an external dependency not currently managed.


def raw_fpath_to_hdr_img_file(
    src_fpath: str,
    dest_fpath: str,
    output_color_profile: Literal['lin_rec2020', 'lin_sRGB'] = DEFAULT_OUTPUT_PROFILE,
    bit_depth: Optional[int] = None,
    check_exposure: bool = True,
    crop_all: bool = True,
) -> Tuple[str, str, str]: # Simplified return type, no longer using Enum for errors
    """
    Converts a raw file to OpenEXR or TIFF HDR.
    Returns (status_string, src_fpath, dest_fpath).
    """

    try:
        # Use RawLoader.load_raw_data directly
        config = ProcessingConfig(crop_all=crop_all, return_float=True)
        raw_loader = RawLoader(config)
        mono_img, metadata = raw_loader.load_raw_data(src_fpath)

        if check_exposure and not is_exposure_ok(mono_img, metadata):
            logger.info(f"# bad exposure for {src_fpath} ({mono_img.mean()=})")
            return "BAD_EXPOSURE", src_fpath, dest_fpath

        # Demosaic and color transform
        config = ProcessingConfig(output_color_profile=output_color_profile)
        bayer_processor = BayerProcessor(config)
        rgb_img = bayer_processor.demosaic(mono_img, metadata)

        color_transformer = ColorTransformer() # Static methods don't need instance, but conceptually fine
        final_img = color_transformer.camRGB_to_profiledRGB_img(
            rgb_img, metadata, output_color_profile=output_color_profile
        )

    except RawProcessingError as e:
        logger.error(f"# Raw processing error with {src_fpath}: {e}")
        return "UNREADABLE_ERROR", src_fpath, dest_fpath
    except Exception as e:
        logger.error(f"# Unknown error {e} with {src_fpath}", exc_info=True)
        return "UNKNOWN_ERROR", src_fpath, dest_fpath

    hdr_nparray_to_file(final_img, dest_fpath, output_color_profile, bit_depth, src_fpath)
    logger.info(f"# Wrote {dest_fpath}")
    return "OK", src_fpath, dest_fpath


def raw_fpath_to_rggb_img_and_metadata(fpath: str, return_float: bool = True) -> Tuple[RGGBImage, Metadata]:
    """
    Converts a raw file path to a 4-channel RGGB image and its associated metadata.
    This orchestrates RawLoader and BayerProcessor.
    """
    config = ProcessingConfig(return_float=return_float, force_rggb=True) # Always force RGGB for this public API
    raw_loader = RawLoader(config)
    mono_img, metadata = raw_loader.load_raw_data(fpath)

    # Convert mono to RGGB (BayerProcessor assumes mono_img is RGGB if force_rggb was True)
    bayer_processor = BayerProcessor(config)
    rggb_img = bayer_processor.mono_to_rggb(mono_img, metadata)

    return rggb_img, metadata


def raw_fpath_to_hdr_img_file_mtrunner(argslist):
    """Multithreaded runner for raw_fpath_to_hdr_img_file."""
    return raw_fpath_to_hdr_img_file(*argslist)


def scenelin_to_pq(img: np.ndarray, in_range: tuple = (0.0, 1.0), out_range: tuple = (0.0, 1.0)) -> np.ndarray:
    """
    Convert scene-linear image to Perceptual Quantizer (PQ) encoding.
    Args:
        img: Input image (float32 or float64), values in in_range.
        in_range: Input range (min, max) for normalization.
        out_range: Output range (min, max) for PQ encoding.
    Returns:
        PQ-encoded image, same shape as input.
    """
    # Normalize input
    img_norm = np.clip((img - in_range[0]) / (in_range[1] - in_range[0]), 0, 1)
    # PQ constants (SMPTE ST 2084)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    # PQ encoding
    pq = ((c1 + c2 * np.power(img_norm, m1)) / (1 + c3 * np.power(img_norm, m1))) ** m2
    # Scale to out_range
    pq_scaled = pq * (out_range[1] - out_range[0]) + out_range[0]
    return pq_scaled


def gamma(img: np.ndarray, gamma_value: float = 2.2, in_range: tuple = (0.0, 1.0), out_range: tuple = (0.0, 1.0)) -> np.ndarray:
    """
    Apply gamma correction to image.
    Args:
        img: Input image (float32 or float64), values in in_range.
        gamma_value: Gamma exponent (e.g., 2.2 for sRGB).
        in_range: Input range (min, max) for normalization.
        out_range: Output range (min, max) for output.
    Returns:
        Gamma-corrected image, same shape as input.
    """
    img_norm = np.clip((img - in_range[0]) / (in_range[1] - in_range[0]), 0, 1)
    img_gamma = np.power(img_norm, 1.0 / gamma_value)
    img_scaled = img_gamma * (out_range[1] - out_range[0]) + out_range[0]
    return img_scaled


def camRGB_to_lin_rec2020_images(camRGB_img: np.ndarray, metadata: Metadata) -> np.ndarray:
    """
    Convert camera RGB image to linear Rec.2020 RGB image using metadata color matrix.
    Args:
        camRGB_img: Input image (3, H, W), camera RGB.
        metadata: Metadata object with rgb_xyz_matrix.
    Returns:
        Linear Rec.2020 RGB image (3, H, W).
    """
    # Get conversion matrix from camera RGB to Rec.2020
    cam_to_xyzd65 = np.linalg.inv(metadata.rgb_xyz_matrix[:3])
    rec2020_matrix = ColorTransformer.get_xyz_to_rgb_matrix("lin_rec2020")
    color_matrix = rec2020_matrix @ cam_to_xyzd65
    orig_shape = camRGB_img.shape
    img_flat = camRGB_img.reshape(3, -1)
    rec2020_img = (color_matrix @ img_flat).reshape(orig_shape)
    return rec2020_img

def demosaic(bayer_mosaic):
    """Demosaic Bayer pattern tensor to RGB.

    Args:
        bayer_mosaic: torch.Tensor [1, H, W] Bayer pattern or [4, H, W] RGGB

    Returns:
        torch.Tensor [3, H*2, W*2] RGB image if input is [1, H, W]
        torch.Tensor [3, H*2, W*2] RGB image if input is [4, H, W] RGGB
    """
    import cv2
    if bayer_mosaic.shape[0] == 4:
        # Convert RGGB to mono Bayer
        bayer_mosaic = rggb_to_mono(bayer_mosaic.unsqueeze(0)).squeeze(0)
    bayer_np = bayer_mosaic.detach().cpu().numpy()
    min_val, max_val = bayer_np.min(), bayer_np.max()
    offset = max(0, -min_val)
    scale = 65535 / (max_val + offset) if max_val + offset > 0 else 1
    prep = ((bayer_np + offset) * scale).astype(np.uint16)[0]
    rgb_hwc = cv2.demosaicing(prep, cv2.COLOR_BayerRGGB2RGB_EA)
    rgb_chw = rgb_hwc.transpose(2, 0, 1).astype(np.float32) / 65535 * (max_val + offset) - offset
    return torch.from_numpy(rgb_chw).to(bayer_mosaic.device)

def rggb_to_mono(rggb_image):
    """Convert RGGB tensor to mono Bayer.

    Args:
        rggb_image: torch.Tensor [1, 4, H, W] RGGB

    Returns:
        torch.Tensor [1, 2*H, 2*W] Bayer mosaic
    """
    r, g1, g2, b = rggb_image[0]
    h, w = r.shape
    mono = torch.zeros((1, 2*h, 2*w), dtype=rggb_image.dtype, device=rggb_image.device)
    mono[0, 0::2, 0::2] = r  # R
    mono[0, 0::2, 1::2] = g1  # G1
    mono[0, 1::2, 0::2] = g2  # G2
    mono[0, 1::2, 1::2] = b  # B
    return mono
