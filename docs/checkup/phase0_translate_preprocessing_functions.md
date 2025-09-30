# Phase 0: Add Preprocessing Functions to raw_processing.py

## Objective
Add 5 missing preprocessing functions from legacy_rawproc.py to raw_processing.py. These functions are used by tools/prep_image_dataset.py for offline YAML metadata generation.

## Prerequisites
None - Phase 0 is independent of all other phases. The 5 preprocessing functions are used by tools/prep_image_dataset.py for offline YAML generation, NOT by ConfigurableDataset at runtime.

## File to Modify
`src/rawnind/dependencies/raw_processing.py`

## Tasks

### Task 0.1: Add Missing Constants

**Location**: After existing constants in raw_processing.py (find appropriate location near other module-level constants)

**Add these 14 constants from legacy_rawproc.py lines 16-36**:

```python
# Preprocessing constants (from legacy_rawproc.py)
BW_POINT = 16383
MAX_GAIN = 10
MIN_BRIGHT_PIXELS_PROPORTION = 0.01
OVEREXPOSURE_LB = 0.997
DC_MAX_LOSS = 0.005
ALIGNMENT_LOSS_LB = 1
ALIGNMENT_MAX_LOSS = 0.002
BAYER_PATTERN_NAME = 'RGGB'
MASK_MEAN_MIN = 0.9995
PROC_IMGS_DPATH = 'proc'
MIN_L1_FOR_DC = 0.01
MIN_L1_FOR_DENOISE = 0.005
MIN_MS_SSIM_FOR_DC = 0.97
MIN_MS_SSIM_FOR_DENOISE = 0.98
```

### Task 0.2: Add make_overexposure_mask()

**Location**: After gamma() function or in preprocessing utilities section

**Translate from legacy_rawproc.py lines 293-295**:

```python
def make_overexposure_mask(img: np.ndarray, overexposure_lb: float = OVEREXPOSURE_LB) -> np.ndarray:
    """Create binary mask marking overexposed pixels.
    
    Args:
        img: Image array to analyze
        overexposure_lb: Threshold for overexposure (default from constant)
        
    Returns:
        Boolean mask where False indicates overexposed pixels
    """
    return img < overexposure_lb
```

### Task 0.3: Add make_loss_mask()

**Location**: After make_overexposure_mask()

**Translate from legacy_rawproc.py lines 334-371**:

```python
def make_loss_mask(
    img_gt: np.ndarray,
    img_noisy: np.ndarray, 
    best_alignment: tuple,
    overexposure_lb: float = OVEREXPOSURE_LB
) -> np.ndarray:
    """Generate loss mask combining overexposure check and alignment validation.
    
    Applies alignment shift, checks for overexposure in both images, and ensures
    mask has sufficient valid pixels for training.
    
    Args:
        img_gt: Ground truth image
        img_noisy: Noisy image
        best_alignment: (v_shift, h_shift) tuple from find_best_alignment()
        overexposure_lb: Overexposure threshold
        
    Returns:
        Boolean mask suitable for loss computation
        
    Raises:
        AssertionError: If mask has insufficient valid pixels (mean < MASK_MEAN_MIN)
    """
    # Apply alignment shifts
    img_gt_shifted, img_noisy_shifted = shift_images(img_gt, img_noisy, best_alignment)
    
    # Create overexposure masks
    mask_gt = make_overexposure_mask(img_gt_shifted, overexposure_lb)
    mask_noisy = make_overexposure_mask(img_noisy_shifted, overexposure_lb)
    
    # Combine masks (both must be valid)
    combined_mask = np.logical_and(mask_gt, mask_noisy)
    
    # Validate sufficient valid pixels
    mask_mean = combined_mask.mean()
    assert mask_mean > MASK_MEAN_MIN, \
        f"Insufficient valid pixels in loss mask: {mask_mean} <= {MASK_MEAN_MIN}"
    
    return combined_mask
```

### Task 0.4: Add find_best_alignment()

**Location**: After make_loss_mask()

**Translate from legacy_rawproc.py lines 374-428**:

```python
def find_best_alignment(
    img_gt: np.ndarray,
    img_noisy: np.ndarray,
    max_v_shift: int = 8,
    max_h_shift: int = 8
) -> tuple:
    """Find optimal pixel shift minimizing L1 loss between images.
    
    Performs grid search over possible shifts, computing L1 distance for each,
    and returns shift with minimum loss.
    
    Args:
        img_gt: Ground truth reference image
        img_noisy: Noisy image to align
        max_v_shift: Maximum vertical shift to search (default 8)
        max_h_shift: Maximum horizontal shift to search (default 8)
        
    Returns:
        Tuple (best_v_shift, best_h_shift, min_loss) where shifts are in pixels
    """
    min_loss = float('inf')
    best_v_shift = 0
    best_h_shift = 0
    
    # Grid search over all possible shifts
    for v_shift in range(-max_v_shift, max_v_shift + 1):
        for h_shift in range(-max_h_shift, max_h_shift + 1):
            # Apply shift
            shifted_gt, shifted_noisy = shift_images(img_gt, img_noisy, (v_shift, h_shift))
            
            # Compute L1 loss
            loss = np.abs(shifted_gt - shifted_noisy).mean()
            
            # Update best if improved
            if loss < min_loss:
                min_loss = loss
                best_v_shift = v_shift
                best_h_shift = h_shift
    
    return (best_v_shift, best_h_shift, min_loss)
```

### Task 0.5: Add get_best_alignment_compute_gain_and_make_loss_mask()

**Location**: After find_best_alignment()

**Translate from legacy_rawproc.py lines 437-519**:

```python
def get_best_alignment_compute_gain_and_make_loss_mask(
    gt_fpath: str,
    noisy_fpath: str,
    overexposure_lb: float = OVEREXPOSURE_LB,
    max_v_shift: int = 8,
    max_h_shift: int = 8
) -> dict:
    """Preprocessing master function: find alignment, compute gain, generate mask.
    
    This function coordinates all preprocessing steps for a clean-noisy image pair:
    1. Load images from file paths
    2. Find optimal alignment between them
    3. Compute gain normalization factor
    4. Generate combined loss mask
    
    Used by tools/prep_image_dataset.py for offline YAML metadata generation.
    
    Args:
        gt_fpath: Path to ground truth image
        noisy_fpath: Path to noisy image
        overexposure_lb: Overexposure threshold
        max_v_shift: Maximum vertical alignment search range
        max_h_shift: Maximum horizontal alignment search range
        
    Returns:
        Dictionary containing:
            - 'best_alignment': (v_shift, h_shift) tuple
            - 'best_alignment_loss': Minimum L1 loss achieved
            - 'gain': Gain normalization factor from match_gain()
            - 'mask_fpath': Path where mask was saved
            - 'mask_mean': Fraction of valid pixels in mask
    """
    from . import numpy_operations as np_ops
    
    # Load images
    img_gt = np_ops.img_fpath_to_np_flt(gt_fpath)
    img_noisy = np_ops.img_fpath_to_np_flt(noisy_fpath)
    
    # Find optimal alignment
    best_v_shift, best_h_shift, alignment_loss = find_best_alignment(
        img_gt, img_noisy, max_v_shift, max_h_shift
    )
    best_alignment = (best_v_shift, best_h_shift)
    
    # Apply alignment for gain computation
    img_gt_aligned, img_noisy_aligned = shift_images(img_gt, img_noisy, best_alignment)
    
    # Compute gain normalization
    gain = match_gain(img_gt_aligned, img_noisy_aligned)
    
    # Generate loss mask
    loss_mask = make_loss_mask(img_gt, img_noisy, best_alignment, overexposure_lb)
    
    # Save mask to file (generate path from noisy_fpath)
    mask_fpath = noisy_fpath.replace('.tif', '_mask.tif').replace('.exr', '_mask.exr')
    np_ops.np_to_img(loss_mask.astype(np.float32), mask_fpath)
    
    # Compute mask statistics
    mask_mean = float(loss_mask.mean())
    
    return {
        'best_alignment': best_alignment,
        'best_alignment_loss': float(alignment_loss),
        'gain': float(gain),
        'mask_fpath': mask_fpath,
        'mask_mean': mask_mean
    }
```

### Task 0.6: Add dt_proc_img()

**Location**: After get_best_alignment_compute_gain_and_make_loss_mask()

**Translate from legacy_rawproc.py lines 496-512**:

```python
def dt_proc_img(
    raw_fpath: str,
    output_fpath: str,
    xmp_fpath: str,
    darktable_cli: str = 'darktable-cli'
) -> None:
    """Process RAW image using darktable CLI with XMP settings.
    
    Executes external darktable-cli command to process RAW files according to
    specified XMP sidecar settings. Used for preprocessing reference images.
    
    Args:
        raw_fpath: Path to input RAW file
        output_fpath: Path for output processed image
        xmp_fpath: Path to XMP sidecar with processing settings
        darktable_cli: Path to darktable-cli executable (default 'darktable-cli')
        
    Raises:
        RuntimeError: If darktable-cli execution fails
    """
    import subprocess
    
    cmd = [
        darktable_cli,
        raw_fpath,
        xmp_fpath,
        output_fpath
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"darktable-cli failed processing {raw_fpath}: {e.stderr}"
        ) from e
```

## Verification

```bash
# Import check - all functions available
python -c "
from rawnind.dependencies.raw_processing import (
    make_overexposure_mask,
    make_loss_mask,
    find_best_alignment,
    get_best_alignment_compute_gain_and_make_loss_mask,
    dt_proc_img,
    BW_POINT,
    ALIGNMENT_MAX_LOSS,
    MASK_MEAN_MIN
)
print('✓ All preprocessing functions and constants imported successfully')
"

# Tools can now import functions
python -c "
import sys
sys.path.insert(0, 'src')
# This would have failed before Phase 0
from rawnind.dependencies.raw_processing import get_best_alignment_compute_gain_and_make_loss_mask
print('✓ tools/prep_image_dataset.py can now import preprocessing functions')
"
```

## Estimated Time
90 minutes