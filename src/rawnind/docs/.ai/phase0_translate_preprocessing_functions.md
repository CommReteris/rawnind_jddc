# Phase 0: Translate Dataset Preprocessing Functions

**Estimated Time**: 30 minutes  
**Priority**: CRITICAL - Blocks tools/prep_image_dataset.py execution  
**File**: [`src/rawnind/dependencies/raw_processing.py`](../../dependencies/raw_processing.py)

## Context

Dataset preprocessing functions from [`legacy_rawproc.py`](../../../legacy_rawproc.py) generate YAML metadata files used by ConfigurableDataset. These are SEPARATE from runtime dataset loading - they run BEFORE training to compute alignments and masks.

## Objective

Translate 7 functions + constants from legacy_rawproc.py into dependencies/raw_processing.py, making them available for tools that need to preprocess raw image datasets.

## Functions to Translate

### 1. Constants (Add near top of file after existing constants)

From legacy_rawproc.py lines 16-36:
```python
LOSS_THRESHOLD: float = 0.4
GT_OVEREXPOSURE_LB: float = 1.0
KEEPERS_QUANTILE: float = 0.9999
MAX_SHIFT_SEARCH: int = 128
NEIGHBORHOOD_SEARCH_WINDOW = 3
GAMMA = 2.2
DS_DN = "RawNIND"
DATASETS_ROOT = os.path.join("..", "..", "datasets")
DS_BASE_DPATH: str = os.path.join(DATASETS_ROOT, DS_DN)
BAYER_DS_DPATH: str = os.path.join(DS_BASE_DPATH, "src", "Bayer")
LINREC2020_DS_DPATH: str = os.path.join(DS_BASE_DPATH, "proc", "lin_rec2020")
MASKS_DPATH = os.path.join(DS_BASE_DPATH, f"masks_{LOSS_THRESHOLD}")
RAWNIND_CONTENT_FPATH = os.path.join(
    DS_BASE_DPATH, "RawNIND_masks_and_alignments.yaml"
)
EXTRARAW_DS_DPATH = os.path.join("..", "..", "datasets", "extraraw")
EXTRARAW_CONTENT_FPATHS = (
    os.path.join(EXTRARAW_DS_DPATH, "trougnouf", "crops_metadata.yaml"),
    os.path.join(EXTRARAW_DS_DPATH, "raw-pixls", "crops_metadata.yaml"),
)
```

**Action**: Add these constants to dependencies/raw_processing.py after line 20 (after existing constants).

### 2. np_l1() Function (legacy_rawproc.py lines 40-43)

```python
def np_l1(img1: np.ndarray, img2: np.ndarray, avg=True) -> Union[float, np.ndarray]:
    """Compute L1 distance between two images."""
    if avg:
        return np.abs(img1 - img2).mean()
    return np.abs(img1 - img2)
```

**Action**: Add after shift_mask() function in dependencies/raw_processing.py.

### 3. make_overexposure_mask() Function (legacy_rawproc.py lines 293-295)

```python
def make_overexposure_mask(
    anchor_img: np.ndarray, gt_overexposure_lb: float = GT_OVEREXPOSURE_LB
):
    """Create binary mask marking non-overexposed pixels."""
    return (anchor_img < gt_overexposure_lb).all(axis=0)
```

**Action**: Add after np_l1() in dependencies/raw_processing.py.

### 4. make_loss_mask() Function (legacy_rawproc.py lines 334-371)

```python
def make_loss_mask(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    loss_threshold: float = LOSS_THRESHOLD,
    keepers_quantile: float = KEEPERS_QUANTILE,
    verbose: bool = False,
) -> Union[np.ndarray, tuple]:
    """Return a loss mask between two aligned images.

    loss_map is the sum of l1 loss over all channels

    0: ignore if loss_map >= threshold
    1: apply loss

    Applies morphological opening to clean up the mask.
    """
    import scipy.ndimage
    
    loss_map = np_l1(
        gamma(anchor_img), gamma(match_gain(anchor_img, target_img)), avg=False
    )
    loss_map = loss_map.sum(axis=0)
    loss_mask = np.ones_like(loss_map)
    reject_threshold = min(loss_threshold, np.quantile(loss_map, keepers_quantile))
    if reject_threshold == 0:
        reject_threshold = 1.0
    if verbose:
        print(f"{reject_threshold=}")
    loss_mask[loss_map >= reject_threshold] = 0.0
    loss_mask = scipy.ndimage.binary_opening(loss_mask.astype(np.uint8)).astype(
        np.float32
    )
    return loss_mask
```

**Action**: Add after make_overexposure_mask() in dependencies/raw_processing.py.

**Note**: Requires scipy.ndimage import at top of file.

### 5. find_best_alignment() Function (legacy_rawproc.py lines 374-428)

```python
def find_best_alignment(
    anchor_img: np.ndarray,
    target_img: np.ndarray,
    max_shift_search: int = MAX_SHIFT_SEARCH,
    return_loss_too: bool = False,
    verbose: bool = False,
) -> Union[tuple, tuple]:
    """Find best alignment (minimal loss) between anchor_img and target_img."""
    target_img = match_gain(anchor_img, target_img)
    assert np.isclose(anchor_img.mean(), target_img.mean(), atol=1e-07), (
        f"{anchor_img.mean()=}, {target_img.mean()=}"
    )
    current_best_shift: tuple = (0, 0)
    shifts_losses: dict = {
        current_best_shift: np_l1(anchor_img, target_img, avg=True)
    }
    if verbose:
        print(f"{shifts_losses=}")

    def explore_neighbors(
        initial_shift: tuple,
        shifts_losses: dict = shifts_losses,
        anchor_img: np.ndarray = anchor_img,
        target_img: np.ndarray = target_img,
        search_window=NEIGHBORHOOD_SEARCH_WINDOW,
    ) -> None:
        """Explore initial_shift's neighbors and update shifts_losses."""
        for yshift in range(-search_window, search_window + 1, 1):
            for xshift in range(-search_window, search_window + 1, 1):
                current_shift = (initial_shift[0] + yshift, initial_shift[1] + xshift)
                if current_shift in shifts_losses:
                    continue
                shifts_losses[current_shift] = np_l1(
                    *shift_images(anchor_img, target_img, current_shift)
                )
                if verbose:
                    print(f"{current_shift=}, {shifts_losses[current_shift]}")

    while (
        min(shifts_losses.values()) > 0
        and abs(current_best_shift[0]) + abs(current_best_shift[1]) < max_shift_search
    ):
        explore_neighbors(current_best_shift)
        new_best_shift = min(shifts_losses, key=shifts_losses.get)
        if new_best_shift == current_best_shift:
            if return_loss_too:
                return new_best_shift, float(min(shifts_losses.values()))
            return new_best_shift
        current_best_shift = new_best_shift
    if return_loss_too:
        return current_best_shift, float(min(shifts_losses.values()))
    return current_best_shift
```

**Action**: Add after make_loss_mask() in dependencies/raw_processing.py.

### 6. img_fpath_to_np_mono_flt_and_metadata() Function (legacy_rawproc.py lines 431-434)

This function has dependencies on `np_imgops` and `raw` modules. Need to check refactored equivalents.

```python
def img_fpath_to_np_mono_flt_and_metadata(fpath: str):
    """Load image as mono float array with metadata."""
    if fpath.endswith(".exr"):
        # For EXR files, use existing loader and construct basic metadata
        # Need to find refactored equivalent of np_imgops.img_fpath_to_np_flt
        pass  # IMPLEMENTATION PENDING - check what's available
    # For RAW files, use RawLoader
    loader = RawLoader()
    return loader.load_mono_and_metadata(fpath)
```

**Action**: SKIP this function - check if equivalent exists in refactored code. May need custom implementation.

### 7. get_best_alignment_compute_gain_and_make_loss_mask() Function (legacy_rawproc.py lines 437-519)

This is the main preprocessing pipeline that:
- Loads GT and noisy images
- Computes alignment using find_best_alignment()
- Generates loss masks
- Saves masks to disk
- Returns metadata dict for YAML

**Dependencies**: Uses `np_imgops.np_to_img()`, `raw.demosaic()`. Need refactored equivalents.

**Action**: Translate after verifying all dependencies are available in refactored code.

### 8. dt_proc_img() Function (legacy_rawproc.py lines 496-512)

```python
def dt_proc_img(src_fpath: str, dest_fpath: str, xmp_fpath: str, compression=True):
    """Process image using darktable-cli."""
    assert shutil.which("darktable-cli")
    assert dest_fpath.endswith(".tif")
    assert not os.path.isfile(dest_fpath), f"{dest_fpath} already exists"
    conversion_cmd: tuple = (
        "darktable-cli",
        src_fpath,
        xmp_fpath,
        dest_fpath,
        "--core",
        "--conf",
        "plugins/imageio/format/tiff/bpp=16",
    )
    subprocess.call(conversion_cmd, timeout=15 * 60)
    assert os.path.isfile(dest_fpath), f"{dest_fpath} was not written by darktable-cli"
```

**Action**: Add after find_best_alignment() in dependencies/raw_processing.py.

## Imports Needed

Add to top of dependencies/raw_processing.py:
```python
import scipy.ndimage  # For make_loss_mask morphological operations
```

## __all__ Exports

Add to __all__ list in dependencies/raw_processing.py:
```python
'np_l1',
'make_overexposure_mask', 
'make_loss_mask',
'find_best_alignment',
'get_best_alignment_compute_gain_and_make_loss_mask',
'dt_proc_img',
# Constants
'LOSS_THRESHOLD',
'KEEPERS_QUANTILE',
'MAX_SHIFT_SEARCH',
'NEIGHBORHOOD_SEARCH_WINDOW',
'GAMMA',
'DS_DN',
'DATASETS_ROOT',
'DS_BASE_DPATH',
'BAYER_DS_DPATH',
'LINREC2020_DS_DPATH',
'MASKS_DPATH',
'RAWNIND_CONTENT_FPATH',
'EXTRARAW_DS_DPATH',
'EXTRARAW_CONTENT_FPATHS',
```

## Verification

After translation, verify tools can import:
```python
from rawnind.dependencies import raw_processing as rawproc
# Should work:
rawproc.get_best_alignment_compute_gain_and_make_loss_mask
rawproc.RAWNIND_CONTENT_FPATH
rawproc.LOSS_THRESHOLD
```

## Dependencies to Resolve

Before implementing function #6 and #7, need to identify refactored equivalents of:
- `np_imgops.img_fpath_to_np_flt()` - likely in image_analysis or external_libraries
- `np_imgops.np_to_img()` - likely in image_analysis or external_libraries
- `raw.demosaic()` - check if already in RawLoader/BayerProcessor classes

These may already exist in the refactored codebase under different names/locations.