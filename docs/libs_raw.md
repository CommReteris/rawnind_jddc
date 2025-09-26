# API Reference: raw.py Module

This document details the public API for the `src/rawnind/libs/raw` module, including types, classes, functions, and exceptions. For usage, see [`docs/raw.md`](../raw.md). All arrays use NumPy float32 [0,1] unless noted; shapes are CHW for consistency with ML frameworks.

## Types and Aliases

### Core Types
- `BayerImage = np.ndarray`  
  Shape: `(1, H, W)`  
  Description: Monochrome Bayer mosaic from sensor. H, W divisible by 4 post-processing. Represents raw CFA data after normalization (subtract black, divide by white range, clip [0,1]).

- `RGGBImage = np.ndarray`  
  Shape: `(4, H//2, W//2)`  
  Description: Separated Bayer channels in RGGB order: index 0=R, 1=G1 (top green), 2=G2 (bottom green), 3=B. Used for per-channel analysis (e.g., denoising).

- `RGBImage = np.ndarray`  
  Shape: `(3, H, W)`  
  Description: Full-color image in CHW format (0=R, 1=G, 2=B). Linear [0,1] unless gamma-corrected.

All types are `dtype=np.float32`; contiguous for efficiency.

### Enums
```python
from enum import Enum

class BayerPattern(Enum):
    """Bayer color filter array patterns.  
    Defines 2x2 repeating tile with channel indices (0=R, 1=G, 2=B, 3=G).  
    Used for pattern detection and forcing RGGB alignment.  
    """
    RGGB = "RGGB"  # [[0,1],[3,2]] – Red top-left (Sony, Canon common)
    GBRG = "GBRG"  # [[1,0],[2,3]] – Green top-left (Nikon)
    BGGR = "BGGR"  # [[2,3],[1,0]] – Blue top-left (Olympus)
    GRBG = "GRBG"  # [[3,2],[0,1]] – Green top-left variant (Fuji non-X-Trans)
```

### Data Structures
```python
from typing import NamedTuple, Optional, Union
from pathlib import Path
import numpy as np

class Metadata(NamedTuple):
    """Immutable container for LibRaw-extracted RAW metadata.  
    All arrays np.float32; WB normalized to G1=1.0.  
    Derived from rawpy.RawImage attrs with validation/normalization.  
    """
    bayer_pattern: BayerPattern  
        # Detected or forced pattern (always RGGB post-load if force_rggb=True).
    rgbg_pattern: np.ndarray  
        # Shape: (2,2) int – CFA tile (e.g., np.array([[0,1],[3,2]]) for RGGB).
    sizes: dict[str, int]  
        # Sensor geometry:  
        # - 'raw_width', 'raw_height': Full mosaic dims.  
        # - 'width', 'height': Active image area.  
        # - 'iwidth', 'iheight': Output after scaling (usually same).  
        # - 'top_margin', 'left_margin': Black border offsets (cropped out).  
    camera_whitebalance: np.ndarray  
        # Shape: (4,) – Multipliers (R, G1, B, G2) from camera profile.
    black_level_per_channel: np.ndarray  
        # Shape: (4,) – Subtract values per CFA position (optical black).
    white_level: int  
        # Global saturation (e.g., 16383 for 14-bit).
    camera_white_level_per_channel: Optional[np.ndarray]  
        # Shape: (4,) or None – Per-channel saturation; fallback to white_level.
    daylight_whitebalance: np.ndarray  
        # Shape: (4,) – Standard 6500K multipliers.
    rgb_xyz_matrix: np.ndarray  
        # Shape: (3,3) – Camera RGB to CIE XYZ (D65); validated sum != 0.
    overexposure_lb: float  
        # Scaling factor for overexposure threshold (1.0 default; min per-channel range).
    camera_whitebalance_norm: np.ndarray  
        # Shape: (4,) – camera_whitebalance / G1; G2 = G1 if 0.
    daylight_whitebalance_norm: np.ndarray  
        # Shape: (4,) – daylight_whitebalance / G1.
```

## Exceptions
```python
class RawProcessingError(Exception):
    """Base exception for processing failures (e.g., invalid metadata, OOM)."""

class UnsupportedFormatError(RawProcessingError):
    """Raised for non-Bayer RAWs, unknown patterns, or LibRaw decode errors  
    (e.g., corrupted file, X-Trans without fallback).  
    Includes LibRaw error msg for debugging.  
    """
```

## Configuration
```python
from dataclasses import dataclass
from typing import Literal, Optional
import cv2

@dataclass
class ProcessingConfig:
    """Pipeline parameters. Defaults optimized for HDR workflows.  
    Passed to loaders/processors; immutable after init.  
    """
    force_rggb: bool = True  
        # Align to RGGB via cropping (1-2 pixels loss); simplifies demosaicing.
    crop_all: bool = True  
        # Crop to min(active, raw) dims; removes black borders/artifacts.
    return_float: bool = True  
        # Normalize to [0,1] float32; disable for integer pipelines.
    wb_type: str = "daylight"  
        # "camera" (proprietary), "daylight" (6500K), "compat" (global white).  
        # Selects norm WB from Metadata.
    demosaic_method: int = cv2.COLOR_BayerRGGB2RGB_EA  
        # OpenCV demosaic code (EA: edge-adaptive, high quality).  
        # Alternatives: COLOR_BayerRG2RGB (bilinear, fast).
    output_color_profile: str = "lin_rec2020"  
        # Target: "lin_rec2020" (wide linear), "lin_sRGB", "gamma_sRGB", "xyz".  
        # Dictates matrix/gamma in export.
    oe_threshold: float = 0.99  
        # Over-clip fraction in exposure check.
    ue_threshold: float = 0.001  
        # Under-clip threshold (near-black).
    qty_threshold: float = 0.75  
        # Max bad pixels ratio for is_exposure_ok.
    bit_depth: Optional[int] = None  
        # EXR: 16 (half), 32 (float); TIFF: 16 (uint16). Auto from dtype.
    check_exposure: bool = True  
        # Enable quality validation in convenience funcs.
```

## Classes

### RawLoader
```python
class RawLoader:
    """RAW decoder and preprocessor.  
    Handles file I/O, LibRaw decoding, metadata extraction, cropping,  
    normalization, and RGGB forcing. Thread-safe.  
    """

    def __init__(self, config: ProcessingConfig):
        """Store config for behavior (e.g., cropping, float output)."""

    def load(self, input_file_path: Union[str, Path]) -> Tuple[BayerImage, Metadata]:
        """Decode and preprocess RAW to Bayer mosaic.  

        Validation: File exists, <500MB (OOM guard).  
        Decoding: rawpy.imread (auto-closes). Extracts raw_image (visible uint16),  
        raw_pattern (2x2), raw_colors (H,W indices), metadata attrs.  
        Preprocessing:  
        - Expand to (1,H,W) float32.  
        - Crop margins (top/left slices), active area (min dims), align %4 (trim borders).  
        - Force RGGB if enabled (pattern-specific crops, e.g., GBRG: rows[1:-1]).  
        - Normalize if return_float: Per-channel $p' = \frac{p - b_c}{w_c - b_c}$, clip [0,1];  
          $b_c$=black, $w_c$=white per CFA. Compat mode uses global w. Update lb.  
        - Validate: Shape matches, %4=0, RGGB.  

        Args:  
            input_file_path: RAW path (.ARW, .CR2, etc.).  

        Returns:  
            (BayerImage, Metadata)  

        Raises:  
            FileNotFoundError, RawProcessingError (size/matrix),  
            UnsupportedFormatError (LibRaw/pattern), NotImplementedError (margins+force).  
        """
```

### BayerProcessor
```python
class BayerProcessor:
    """Bayer ops: WB application, channel mux/demux, demosaicing, exposure QA.  
    Vectorized for speed; assumes RGGB input.  
    """

    def __init__(self, config: ProcessingConfig):
        """Store config (e.g., demosaic method, thresholds)."""

    def apply_white_balance(self, bayer_mosaic: BayerImage, metadata: Metadata, reverse: bool = False, in_place: bool = False) -> Optional[BayerImage]:
        """Scale CFA positions by WB multipliers (or inverse).  

        WB: metadata.{wb_type}_whitebalance_norm (R,G1,B,G2 normalized to G1=1).  
        Map to RGGB: wb_map = [R, G1, G2, B] = norm[[0,1,3,2]].  
        Apply via strides (no loops):  
        - even_row even_col *= wb[0] (R)  
        - even_row odd_col *= wb[1] (G1)  
        - odd_row even_col *= wb[3] (G2)  
        - odd_row odd_col *= wb[2] (B)  
        Reverse: Divide (op = /). Clip [0,1].  

        Args:  
            bayer_mosaic: (1,H,W) input.  
            metadata: WB source.  
            reverse: Inverse op (post-demosaic linearization).  
            in_place: Modify in-situ (memory save).  

        Returns:  
            Balanced image or None (in_place).  
        """

    def mono_to_rggb(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGGBImage:
        """Demultiplex to 4 channels.  

        Strided views: R = [0::2,0::2], G1=[0::2,1::2], G2=[1::2,0::2], B=[1::2,1::2].  
        Stack axis=0; validate RGGB.  

        Args/Returns/Raises: As above.  
        """

    def rggb_to_mono(self, rggb_image: RGGBImage) -> BayerImage:
        """Mux channels to mosaic.  

        Zeros (1,2h,2w); assign strides: [0::2,0::2]=R, etc. Validate 4-ch.  

        Args/Returns: As inverse.  
        """

    def demosaic(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGBImage:
        """CFA interpolation to RGB.  

        Prep: offset = max(0, -min); scale = 65535 / (max+offset); uint16 clip.  
        Demosaic: cv2.demosaicing(uint16, method) → HWC RGB.  
        Post: CHW float32; reverse scale - offset, clip [0,1]. Warn bounds.  

        Args/Returns/Raises: As above.  
        """

    def is_exposure_ok(self, bayer_mosaic: BayerImage, metadata: Metadata) -> bool:
        """QA: Bad pixels ratio.  

        To RGGB; over_mask = any >= oe * lb; under = all <= ue.  
        return (over.sum() + under.sum()) / total <= qty.  

        Args/Returns: As above.  
        """
```

### ColorTransformer
```python
class ColorTransformer:
    """Color basis transforms via matrices.  

    camRGB → XYZ (inv camera matrix) → profiled (XYZ-to-target).  
    """

    @staticmethod
    def get_xyz_to_rgb_matrix(profile: str) -> np.ndarray:
        """3x3 primaries matrix.  

        lin_rec2020: BT.2020.  
        lin_sRGB: sRGB linear.  
        $ M = \begin{bmatrix} 1.7167 & -0.3557 & -0.2534 \\ \dots \end{bmatrix} $ (Rec.2020 ex.).  
        """

    def cam_rgb_to_profiled(self, cam_rgb: RGBImage, metadata: Metadata, profile: str) -> RGBImage:
        """Batch matrix transform.  

        inv_cam = np.linalg.inv(rgb_xyz[:3,:3]).  
        transform = xyz_to_rgb @ inv_cam.  
        profiled = (transform @ cam_rgb.reshape(3,-1)).reshape(shape).  
        Gamma if "gamma_*" (_apply_gamma in-place).  

        Args/Returns/Raises: As above (LinAlgError on singular).  
        """

    def _apply_gamma(self, img: RGBImage, profile: str) -> None:
        """sRGB gamma: mask = img > 0.0031308; img[mask] = 1.055 * img[mask]^(1/2.4) - 0.055; else *12.92.  
        Vectorized mask/power.  
        """
```

### HdrExporter
```python
class HdrExporter:
    """Format-specific writers with metadata. Providers auto-detected.  
    """

    def __init__(self, exr_provider: str = "OpenImageIO", tiff_provider: str = "OpenImageIO"):
        """Fallbacks: EXR to OpenEXR, TIFF to OpenCV."""

    def save(self, img: RGBImage, output_path: Union[str, Path], profile: str, bit_depth: Optional[int] = None, src_path: Optional[Path] = None) -> None:
        """Dispatch to EXR/TIFF; mkdir; exiftool copy if src.  
        bit_depth: 16/32 EXR (half/float), 16 TIFF (uint16).  
        Warn TIFF range/loss, non-sRGB.  
        """

    def _save_exr(self, img: RGBImage, path: Path, profile: str, bit_depth: Optional[int] = None):
        """ZIPS compression; chromaticities (Imath/OIIO).  
        OIIO: Spec(3-ch HALF/FLOAT), attrs ColorSpace/chromaticities (8 floats: Rx,Ry,...).  
        OpenEXR: Header channels R/G/B, Chromaticities(V2f).  
        HWC transpose for write.  
        """

    def _save_tiff(self, img: RGBImage, path: Path, profile: str, bit_depth: Optional[int] = None):
        """uint16 [0,65535]; OpenCV BGR imwrite or OIIO w/ ICC (rec2020 bytes).  
        Assume sRGB; warn gamma/other.  
        """
```

## Functions

### Convenience Pipelines
```python
def raw_fpath_to_mono_img_and_metadata(
    file_path: str, force_rggb: bool = True, crop_all: bool = True, return_float: bool = True
) -> Tuple[BayerImage, Metadata]:
    """Config wrapper for RawLoader.load()."""

def raw_fpath_to_rggb_img_and_metadata(file_path: str, return_float: bool = True) -> Tuple[RGGBImage, Metadata]:
    """Load + mono_to_rggb."""

def raw_fpath_to_hdr_img_file(
    src_path: str,
    dest_path: str,
    output_profile: Literal["lin_rec2020", "lin_sRGB"] = "lin_rec2020",
    bit_depth: Optional[int] = None,
    check_exposure: bool = True,
    crop_all: bool = True,
) -> Tuple[str, str, str]:
    """End-to-end: load → WB (camera) → exposure? → demosaic → transform → save.  
    Returns ("OK"/"BAD_EXPOSURE"/"UNREADABLE_ERROR"/"UNKNOWN_ERROR", src, dest).  
    """
```

### X-Trans Fallback
```python
def is_xtrans(input_file_path: Union[str, Path]) -> bool:
    """RAF suffix + header b'RAF' magic."""

def xtrans_to_openexr(src_path: Union[str, Path], dest_path: Union[str, Path], profile: str = "lin_rec2020") -> None:
    """Darktable CLI: darktable-cli src xmp dest --conf exr/bpp=16.  
    Requires dt4_xtrans_to_linrec2020.xmp; only lin_rec2020.  
    """
```

### Utilities
```python
def get_sample_raw_file(url: str = "https://.../DSC01568.ARW") -> str:
    """Download/cache sample to data/."""

def libraw_process(raw_path: str, out_path: str) -> None:
    """rawpy postprocess (VNG, sRGB, no WB) → imageio; skip if no imageio."""
```

## Constants
- `BAYER_PATTERNS: dict[str, np.ndarray]` – 2x2 layouts.
- `DEFAULT_OUTPUT_PROFILE = "lin_rec2020"`
- Thresholds: `DEFAULT_OE_THRESHOLD=0.99`, etc.
- `MAX_FILE_SIZE_MB = 500`
- `OPENEXR_PROVIDER`, `TIFF_PROVIDER`: Detected strings.

For implementation details, see source code docstrings.