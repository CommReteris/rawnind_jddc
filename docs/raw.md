
# RAW Image Processing Library

## Fundamentals of RAW Image Processing

RAW image files contain unprocessed data directly from a camera's sensor, providing greater flexibility for post-processing compared to compressed formats like JPEG. This data includes the full dynamic range captured by the sensor, typically 12 to 16 bits per channel, without the automatic adjustments (such as tone mapping or sharpening) applied in-camera. The primary structure of a RAW file is a Bayer mosaic, derived from the camera's color filter array (CFA), which allows the sensor to record color information.

### The Bayer Color Filter Array
The Bayer CFA, developed by Bryce Bayer in 1976, overlays a mosaic of color filters on the sensor pixels. In this arrangement:
- Approximately 50% of pixels are filtered for green (G) to align with human visual sensitivity to luminance.
- 25% are for red (R) and 25% for blue (B).

The standard 2x2 repeating pattern, known as RGGB, positions the colors as follows:

|   | Col 0 | Col 1 |
|---|-------|-------|
| Row 0 | R | G |
| Row 1 | G | B |

This table represents the 2x2 tile: R (red) top-left, G (green) top-right and bottom-left, B (blue) bottom-right. The pattern repeats across the sensor.

Each pixel captures intensity for only one color channel, resulting in a subsampled representation where full RGB values must be interpolated during demosaicing. Other patterns exist (e.g., GBRG for some Nikon cameras), but the library normalizes to RGGB where possible.

| Pattern | 2x2 Layout | Top-Left Pixel | Common Cameras |
|---------|------------|----------------|---------------|
| RGGB   | [[R,G],[G,B]] | R | Sony, Canon |
| GBRG   | [[G,R],[B,G]] | G | Nikon |
| BGGR   | [[B,G],[G,R]] | B | Olympus |
| GRBG   | [[G,B],[R,G]] | G | Fuji (non-X-Trans) |

RAW files store this mosaic data in proprietary formats (e.g., Sony ARW, Canon CR2), along with metadata such as sensor dimensions, black levels, white balance coefficients, and color transformation matrices. The advantages of RAW include:
- Preservation of highlights and shadows for recovery in editing.
- Access to camera-specific calibration for accurate color reproduction.
- Support for advanced workflows, such as HDR merging or integration with machine learning models for denoising.

However, processing RAW files involves challenges, including decoding compressed data, handling pattern variations, and avoiding artifacts from interpolation.

### Overview of the Processing Pipeline
The library implements a modular pipeline for RAW processing, designed to handle these challenges efficiently. The steps are visualized in the following text-based flowchart:

```
RAW File --> Decode (LibRaw) --> Extract Mosaic & Metadata
           |
           v
Crop Borders & Align RGGB --> Normalize to [0,1]
           |
           v
Apply White Balance --> Demosaic to RGB --> Transform Color Space
           |
           v
Check Exposure Quality --> {OK?} --> Yes: Export to EXR/TIFF --> Output HDR File
                       | No
                       v
                       Flag Error --> Retry or Skip
```

1. **Decoding**: Use LibRaw (via rawpy) to extract the Bayer mosaic and metadata from the file.
2. **Preprocessing**: Crop non-image areas, normalize pixel values to [0,1], and align the pattern to RGGB.
3. **White Balance**: Apply color correction multipliers to the mosaic based on metadata.
4. **Demosaicing**: Interpolate the mosaic to a full RGB image using an edge-aware algorithm.
5. **Color Space Transformation**: Convert from camera-specific RGB to a standard color space via an XYZ intermediate.
6. **Exposure Assessment**: Evaluate the image for over- or under-exposure to ensure quality.
7. **Export**: Save the result in HDR formats like EXR or TIFF, embedding relevant metadata.

This pipeline is optimized for performance through vectorized operations and early cropping, reducing memory usage and computation time. For example, a 24-megapixel RAW file (approximately 6000x4000 pixels) requires about 100 MB in raw form and up to 300 MB after processing to float32 RGB.

For detailed API documentation, including classes, functions, and types, refer to [`API/libs_raw.md`](API/libs_raw.md).

## Dependencies

The library requires the following core dependencies, installable via pip:
- `rawpy`: Provides bindings to the LibRaw C library for decoding over 500 RAW formats, including handling of compression and metadata extraction.
- `numpy`: Enables vectorized array operations essential for efficient processing, such as broadcasting white balance coefficients across the mosaic.
- `opencv-python`: Supplies demosaicing functions and serves as a fallback for TIFF export.

Optional dependencies improve functionality:
- `OpenImageIO`: Recommended for EXR and TIFF I/O due to its support for metadata embedding and faster performance.
- `OpenEXR` and `Imath`: Fallback for EXR export if OpenImageIO is unavailable.
- `imageio`: Used for baseline processing comparisons.
- System-level tools: `darktable-cli` for X-Trans support and `exiftool` for copying EXIF metadata.

Provider detection occurs at module import, with logging for configuration details. If optional components are missing, the library degrades gracefully (e.g., using OpenCV for TIFF).

## Configuration

Configuration is managed through the `ProcessingConfig` dataclass, which allows customization of pipeline behavior. Key parameters and their purposes include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| force_rggb | True | Aligns non-RGGB patterns to RGGB through minimal cropping (typically 1-2 pixels), standardizing input for demosaicing. |
| crop_all | True | Reduces the image to the active sensor area, excluding calibration borders to eliminate artifacts and save memory (5-10% reduction). |
| return_float | True | Normalizes output to [0,1] float32 for mathematical operations; set to False for integer-based workflows to conserve memory. |
| wb_type | "daylight" | Selects white balance source ("camera" for shot-specific, "daylight" for standard 6500K, "compat" for uniform scaling). |
| demosaic_method | cv2.COLOR_BayerRGGB2RGB_EA | Specifies the interpolation algorithm; EA provides higher quality by adapting to edges, though it is slower than bilinear. |
| output_color_profile | "lin_rec2020" | Defines the target color space; linear profiles preserve dynamic range for HDR, while gamma-corrected ones suit display. |
| oe_threshold | 0.99 | Controls overexposure detection; values near 1.0 identify clipped highlights. |
| ue_threshold | 0.001 | Sets underexposure threshold; low values flag noisy shadows. |
| qty_threshold | 0.75 | Maximum allowable bad pixel ratio for passing quality checks. |

These defaults are tuned for HDR and machine learning applications but can be adjusted for specific use cases, such as stricter exposure validation for batch processing.

## Technical Deep Dive

The following sections provide a detailed examination of each pipeline stage, including underlying algorithms and numerical considerations.

### 1. Decoding and Preprocessing: From Bits to Bayer (RawLoader)

Decoding uses LibRaw to parse the file and produce:
- `raw_image`: A uint16 array of the visible Bayer mosaic.
- `raw_colors`: A (H, W) array mapping each pixel to its CFA index (0=R, 1=G, 2=B, 3=G).
- Metadata: Including