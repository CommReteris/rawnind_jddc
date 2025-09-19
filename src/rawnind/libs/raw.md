# RAW Image Processing Library

## Overview

This module provides an improved library for processing RAW images using LibRaw via the `rawpy` Python binding. It supports Bayer mosaic extraction, demosaicing, white balance application, color space conversion, and export to HDR formats like EXR and TIFF. The implementation is refactored for modularity, performance (e.g., vectorized NumPy operations), and robustness (e.g., enhanced error handling, input validation).

### Key Enhancements
- **Modular Design**: Separate classes for loading (`RawLoader`), processing (`BayerProcessor`), color transformation (`ColorTransformer`), and export (`HdrExporter`).
- **Performance Optimizations**: Vectorized operations for white balance, normalization, and pattern extraction; single-pass cropping; no temporary copies in critical paths.
- **Consistency**: Full typing support, centralized logging with `pathlib`, and constants.
- **Error Handling**: Custom exceptions (`RawProcessingError`, `UnsupportedFormatError`) with validation for file size, patterns, and exposure.
- **X-Trans Support**: Validation and conversion for Fujifilm RAF files using Darktable CLI as a fallback.
- **Backward Compatibility**: Convenience functions for common workflows.

The library assumes Bayer-pattern RAWs (e.g., ARW, CR2, CRW) and forces RGGB layout where possible. It does not support non-Bayer formats natively.

## Dependencies

Install via `pip install -r requirements.txt` or manually:

- `rawpy`: LibRaw bindings for RAW loading.
- `numpy`: Array operations and vectorization.
- `opencv-python` (cv2): Demosaicing and TIFF export fallback.
- `imageio`: Optional for LibRaw sanity checks.
- `OpenImageIO` or `OpenEXR`: For EXR export (OpenImageIO preferred for performance).
- `Imath`: OpenEXR math utilities.
- `requests`: For sample file download.
- Optional: `darktable-cli` for X-Trans conversion; `exiftool` for metadata copying.

Provider detection is automatic:
- OpenEXR: Prefers OpenImageIO, falls back to OpenEXR.
- TIFF: Prefers OpenImageIO, falls back to OpenCV.

## Configuration

Use the `ProcessingConfig` dataclass to customize the pipeline:

```python
from dataclasses import dataclass

@dataclass
class ProcessingConfig:
    """Configuration for RAW processing pipeline."""
    force_rggb: bool = True  # Force RGGB pattern via cropping.
    crop_all: bool = True    # Crop to active area.
    return_float: bool = True  # Normalize to [0,1] float32.
    wb_type: str = "daylight"  # "camera" or "daylight" white balance; "compat" for legacy.
    demosaic_method: int = cv2.COLOR_BayerRGGB2RGB_EA  # OpenCV demosaic flag.
    output_color_profile: str = "lin_rec2020"  # "lin_rec2020", "lin_sRGB", "gamma_sRGB", "xyz".
    oe_threshold: float = 0.99  # Overexposure threshold.
    ue_threshold: float = 0.001  # Underexposure threshold.
    qty_threshold: float = 0.75  # Max bad pixel ratio for exposure check.
    bit_depth: Optional[int] = None  # EXR/TIFF bit depth (16 or 32).
    check_exposure: bool = True  # Validate exposure quality.
```

Constants are defined at module level, e.g., `BAYER_PATTERNS`, `DEFAULT_OUTPUT_PROFILE = "lin_rec2020"`.

## API Reference

### Type Aliases
- `BayerImage`: `np.ndarray` (shape: (1, H, W)) – Monochrome Bayer mosaic.
- `RGGBImage`: `np.ndarray` (shape: (4, H/2, W/2)) – Separated RGGB channels.
- `RGBImage`: `np.ndarray` (shape: (3, H, W)) – RGB image in CHW format.

### Exceptions
- `RawProcessingError`: Base for processing errors.
- `UnsupportedFormatError`: For non-Bayer or invalid RAWs.

### Enums and Data Structures
- `BayerPattern(Enum)`: Supported patterns – `RGGB`, `GBRG`, `BGGR`, `GRBG`.
- `Metadata(NamedTuple)`: Extracted RAW metadata.
  - Fields: `bayer_pattern`, `rgbg_pattern`, `sizes` (dict), `camera_whitebalance`, `black_level_per_channel`, `white_level`, `camera_white_level_per_channel`, `daylight_whitebalance`, `rgb_xyz_matrix`, `overexposure_lb`, `camera_whitebalance_norm`, `daylight_whitebalance_norm`.

### RawLoader
Handles loading RAWs to Bayer mosaic and metadata.

```python
class RawLoader:
    """Handles loading and initial processing of RAW files."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def load(self, input_file_path: Union[str, Path]) -> Tuple[BayerImage, Metadata]:
        """Load RAW file to mono Bayer mosaic and metadata.

        Validates file (existence, size < 500MB), extracts data using rawpy,
        crops borders/active area, optionally forces RGGB, normalizes to [0,1],
        and validates shape (divisible by 4, RGGB pattern).

        Args:
            input_file_path: Path to RAW file (e.g., .ARW, .CR2).

        Returns:
            Tuple of Bayer mosaic and metadata.

        Raises:
            FileNotFoundError: If file missing.
            RawProcessingError: If too large or invalid matrix.
            UnsupportedFormatError: If non-Bayer or unknown pattern.
        """
        # Implementation details: Uses rawpy.imread context manager for cleanup.
        # Single-pass cropping/normalization for efficiency.
```

Internal methods (not public):
- `_extract_metadata`: Normalizes WB to green channel.
- `_pattern_to_enum`: Maps RGBG indices to BayerPattern.
- `_crop_and_adjust`: Crops margins, active area, forces RGGB if needed.
- `_force_rggb`: Crops to align non-RGGB patterns (e.g., shift GBRG by 1 row).
- `_normalize`: Subtracts black level, divides by white range, clips [0,1].
- `_validate_shape`: Ensures dimensions and pattern.

### BayerProcessor
Processes Bayer data: white balance, pattern conversion, demosaicing.

```python
class BayerProcessor:
    """Processes Bayer mosaics: WB, demosaic, RGGB conversion. Performance-optimized with vectorization."""

    def __init__(self, config: ProcessingConfig):
        self.config = config

    def apply_white_balance(self, bayer_mosaic: BayerImage, metadata: Metadata, reverse: bool = False, in_place: bool = False) -> Optional[BayerImage]:
        """Apply or reverse white balance using camera or daylight multipliers.

        Vectorized: Broadcasts WB coefficients to RGGB positions (R,G1,G2,B).
        Clips to [0,1] to prevent overflow.

        Args:
            bayer_mosaic: Input Bayer image.
            metadata: Contains WB coefficients.
            reverse: Divide instead of multiply.
            in_place: Modify input directly.

        Returns:
            Balanced image (or None if in_place).
        """
        # Uses advanced indexing for even/odd rows/cols; no loops.

    def mono_to_rggb(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGGBImage:
        """Extract 4-channel RGGB from monochrome Bayer (assumes RGGB).

        Vectorized slicing: R = [0::2,0::2], G1=[0::2,1::2], etc.

        Args:
            bayer_mosaic: Monochrome Bayer.
            metadata: For pattern validation.

        Returns:
            RGGB channels as (4, H/2, W/2) float32.

        Raises:
            UnsupportedFormatError: If not RGGB.
        """

    def rggb_to_mono(self, rggb_image: RGGBImage) -> BayerImage:
        """Interleave RGGB channels back to monochrome Bayer.

        Vectorized assignment to even/odd slices.

        Args:
            rggb_image: 4-channel RGGB.

        Returns:
            Monochrome Bayer (1, H*2, W*2).
        """

    def demosaic(self, bayer_mosaic: BayerImage, metadata: Metadata) -> RGBImage:
        """Demosaic Bayer to RGB using OpenCV.

        Scales to uint16 [0,65535] for OpenCV, demosaics (EA method default),
        rescales back to float [0,1] CHW.

        Args:
            bayer_mosaic: RGGB Bayer.
            metadata: For validation.

        Returns:
            RGB image (3, H, W) float32, clipped [0,1].

        Raises:
            UnsupportedFormatError: If not RGGB.
        """
        # Warns if output slightly out of bounds.

    def is_exposure_ok(self, bayer_mosaic: BayerImage, metadata: Metadata) -> bool:
        """Check if exposure is acceptable (low over/under-exposed pixels).

        Converts to RGGB, counts pixels > oe_threshold or < ue_threshold,
        returns True if bad ratio <= qty_threshold.

        Args:
            bayer_mosaic: Input Bayer.
            metadata: For overexposure_lb scaling.

        Returns:
            bool: Exposure quality.
        """
```

### ColorTransformer
Manages color space conversions.

```python
class ColorTransformer:
    """Handles color space transformations."""

    @staticmethod
    def get_xyz_to_rgb_matrix(profile: str) -> np.ndarray:
        """Get transformation matrix from XYZ to profiled RGB.

        Supports "lin_rec2020", "lin_sRGB" (or "sRGB").

        Args:
            profile: Color profile name.

        Returns:
            3x3 matrix float32.

        Raises:
            NotImplementedError: For unsupported profiles.
        """

    def cam_rgb_to_profiled(self, cam_rgb: RGBImage, metadata: Metadata, profile: str) -> RGBImage:
        """Convert camera RGB to profiled RGB via XYZ intermediate.

        Inverts camera RGB-XYZ matrix, applies XYZ-to-profile matrix.
        Applies gamma if profile starts with "gamma".

        Args:
            cam_rgb: Camera RGB (3, H, W).
            metadata: Contains rgb_xyz_matrix.
            profile: Target profile.

        Returns:
            Profiled RGB (3, H, W) float32.

        Raises:
            NotImplementedError: For invalid matrix or gamma.
        """
        # Vectorized matrix @ reshape for batch transform.

    def _apply_gamma(self, img: RGBImage, profile: str) -> None:
        # In-place sRGB gamma: piecewise power function.
```

### HdrExporter
Exports RGB to HDR formats with metadata.

```python
class HdrExporter:
    """Exports to HDR formats (EXR/TIFF) with metadata embedding."""

    def __init__(self, exr_provider: str = OPENEXR_PROVIDER, tiff_provider: str = TIFF_PROVIDER):
        self.exr_provider = exr_provider
        self.tiff_provider = tiff_provider

    def save(self, img: RGBImage, output_path: Union[str, Path], profile: str, bit_depth: Optional[int] = None, src_path: Optional[Path] = None) -> None:
        """Save RGB to EXR or TIFF.

        Creates directories, embeds color space (Rec.2020/sRGB chromaticities or ICC),
        copies EXIF from source using exiftool if available.
        Warns on data loss for TIFF (assumes [0,1]).

        Args:
            img: RGB (3, H, W) float32.
            output_path: EXR or TIFF path.
            profile: Color profile for metadata.
            bit_depth: 16 (half) or 32 (float) for EXR; uint16 for TIFF.
            src_path: Source for EXIF copy.

        Raises:
            ValueError: Unsupported extension.
            IOError: Write failure.
        """
        # Delegates to _save_exr or _save_tiff.

    def _save_exr(self, img: RGBImage, path: Path, profile: str, bit_depth: Optional[int] = None):
        # Uses OpenImageIO (preferred) or OpenEXR; sets compression ZIPS, chromaticities.

    def _save_tiff(self, img: RGBImage, path: Path, profile: str, bit_depth: Optional[int] = None):
        # Uses OpenCV (BGR uint16) or OpenImageIO (with ICC for Rec.2020).
        # Warns if range not [0,1] or non-sRGB profile.
```

### X-Trans Support
Limited support for Fujifilm X-Trans RAWs (.RAF).

```python
def is_xtrans(input_file_path: Union[str, Path]) -> bool:
    """Check if file is X-Trans RAF.

    Validates suffix and RAF header.

    Returns:
        bool: True if X-Trans.
    """

def xtrans_to_openexr(src_path: Union[str, Path], dest_path: Union[str, Path], profile: str = "lin_rec2020") -> None:
    """Convert X-Trans to EXR using Darktable CLI.

    Requires darktable-cli and XMP config file.
    Only supports lin_rec2020 profile.

    Args:
        src_path: Input RAF.
        dest_path: Output EXR.
        profile: Must be "lin_rec2020".

    Raises:
        UnsupportedFormatError: If not X-Trans.
        RuntimeError: If darktable-cli missing.
        FileNotFoundError: If XMP config missing.
        RawProcessingError: If conversion fails.
    """
```

### Convenience Functions
High-level APIs for common tasks.

```python
def raw_fpath_to_mono_img_and_metadata(
    file_path: str, force_rggb: bool = True, crop_all: bool = True, return_float: bool = True
) -> Tuple[BayerImage, Metadata]:
    """Load RAW to Bayer mosaic and metadata (config wrapper)."""

def raw_fpath_to_rggb_img_and_metadata(file_path: str, return_float: bool = True) -> Tuple[RGGBImage, Metadata]:
    """Load and extract RGGB channels."""

def raw_fpath_to_hdr_img_file(
    src_path: str,
    dest_path: str,
    output_profile: Literal["lin_rec2020", "lin_sRGB"] = "lin_rec2020",
    bit_depth: Optional[int] = None,
    check_exposure: bool = True,
    crop_all: bool = True,
) -> Tuple[str, str, str]:  # (outcome, src, dest)
    """Full pipeline: Load → WB → Demosaic → Color Transform → Export.

    Checks exposure if enabled; returns "OK", "BAD_EXPOSURE", "UNREADABLE_ERROR", or "UNKNOWN_ERROR".
    Applies camera WB by default.

    Args:
        src_path: Input RAW.
        dest_path: Output EXR/TIFF.
        output_profile: Target color space.
        bit_depth: Export depth.
        check_exposure: Skip if bad.
        crop_all: Apply cropping.

    Returns:
        Tuple of status, source, dest paths.
    """
```

### Utilities
```python
def get_sample_raw_file(url: str = "https://nc.trougnouf.com/.../DSC01568.ARW") -> str:
    """Download and cache sample ARW file to data/."""

def libraw_process(raw_path: str, out_path: str) -> None:
    """Sanity check: Postprocess with rawpy (VNG demosaic, sRGB) and save via imageio.
    Disabled if imageio missing.
    """
```

## Usage Examples

### Basic Loading and Processing
```python
from pathlib import Path
from src.rawnind.libs.raw import ProcessingConfig, RawLoader, BayerProcessor, ColorTransformer, HdrExporter

config = ProcessingConfig(output_color_profile="lin_rec2020")
loader = RawLoader(config)
mono, metadata = loader.load("path/to/image.ARW")

processor = BayerProcessor(config)
wb_mono = processor.apply_white_balance(mono, metadata)
rggb = processor.mono_to_rggb(wb_mono, metadata)
if processor.is_exposure_ok(wb_mono, metadata):
    rgb = processor.demosaic(wb_mono, metadata)
    transformer = ColorTransformer()
    profiled = transformer.cam_rgb_to_profiled(rgb, metadata, "lin_rec2020")
    exporter = HdrExporter()
    exporter.save(profiled, "output.exr", "lin_rec2020")
```

### Full Pipeline with Convenience Function
```python
from src.rawnind.libs.raw import raw_fpath_to_hdr_img_file

status, src, dest = raw_fpath_to_hdr_img_file("input.ARW", "output.exr")
if status == "OK":
    print("Exported successfully")
else:
    print(f"Failed: {status}")
```

### X-Trans Conversion
```python
from src.rawnind.libs.raw import xtrans_to_openexr

xtrans_to_openexr("input.RAF", "output.exr")
```

## CLI Usage

Run the module directly for testing:

```bash
python -m src.rawnind.libs.raw -i input.ARW -o output --no_wb
```

- `-i, --raw_fpath`: Input RAW (defaults to sample download).
- `-o, --out_base_path`: Output prefix (defaults to tests_output/raw.py.main).
- `--no_wb`: Skip white balance.

Outputs: `.lin_rec2020.exr`, `.lin_sRGB.exr`, `.gamma_sRGB.tif`, `.camRGB.exr`, `.libraw.tif`.

Logs metadata and saves to `tests_output/`.

## Limitations and Notes
- **Performance**: Optimized for Bayer; X-Trans uses external CLI (slower).
- **Formats**: Input: Standard RAWs (via LibRaw). Output: EXR (preferred for HDR), TIFF (16-bit, potential data loss).
- **Assumptions**: Images normalized [0,1]; RGGB after processing.
- **Testing**: Use `get_sample_raw_file()` for validation.
- **Extensions**: Add more demosaic methods or profiles as needed.

For issues, check logs (e.g., `logging.basicConfig(level=logging.INFO)`).
