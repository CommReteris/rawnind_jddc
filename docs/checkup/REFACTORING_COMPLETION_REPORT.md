# RawNIND Clean API Refactoring - Final Report

## Executive Summary

The RawNIND dataset package refactoring is **80% complete**. The clean API architecture exists with proper configuration classes, wrapper infrastructure, and factory functions. Four legacy dataset classes contain fully functional reference implementations that must be **translated with 100% domain logic fidelity** into a unified [`ConfigurableDataset`](../../src/rawnind/dataset/clean_api.py:70) class.

## Current State Analysis

### What Works
✅ Base dataset classes extracted (`base_dataset.py`)
✅ Configuration system (`DatasetConfig`, `BayerDatasetConfig`, `RgbDatasetConfig`)
✅ Wrapper infrastructure (`CleanDataset`, `CleanValidationDataset`, `CleanTestDataset`)
✅ Factory functions (`create_training_dataset`, `create_validation_dataset`, `create_test_dataset`)
✅ Batch format standardization in `_standardize_batch_format()`
✅ Training and inference clean APIs ready to consume datasets

### What's Broken
❌ **ConfigurableDataset generates random data** instead of loading real images
❌ **create_training_datasets() missing** - training code cannot get dataloaders
❌ **3 syntax errors** block code execution
❌ **Duplicate class definitions** violate "single source of truth" anti-pattern rule
❌ **Invalid attribute access** (`self.config.config.is_bayer`) causes AttributeError

## Architecture Understanding

### Translation Pattern (Not Wrapping)

The refactoring translates legacy implementations into ConfigurableDataset:

```
BEFORE:
┌─────────────────────────────────────────┐
│ legacy_rawds.py (1540 lines)           │
│ ├── CleanProfiledRGBCleanBayerImageCropsDataset │
│ ├── CleanProfiledRGBCleanProfiledRGBImageCropsDataset │
│ ├── CleanProfiledRGBNoisyBayerImageCropsDataset │
│ └── CleanProfiledRGBNoisyProfiledRGBImageCropsDataset │
│                                         │
│ Extracted (with duplication) to:       │
│ ├── bayer_datasets.py (2 classes)      │
│ ├── rgb_datasets.py (2 classes)        │
│ ├── clean_datasets.py (2 duplicates)   │
│ └── noisy_datasets.py (1 duplicate)    │
└─────────────────────────────────────────┘

AFTER:
┌─────────────────────────────────────────┐
│ clean_api.py                            │
│ └── ConfigurableDataset                 │
│     ├── Translated __init__ (from 4)    │
│     ├── Translated _load_dataset (from 4) │
│     └── Translated __getitem__ (4 branches) │
│         ├── Branch: clean_noisy + bayer │
│         ├── Branch: clean_noisy + rgb   │
│         ├── Branch: clean_clean + bayer │
│         └── Branch: clean_clean + rgb   │
│                                         │
│ Legacy files: DELETED                   │
└─────────────────────────────────────────┘
```

### Key Insight
**ConfigurableDataset uses config-driven conditional branching to unify all 4 dataset type behaviors into a single implementation.**

## Critical Issues Identified

### Blocking Errors (Phase 1)

1. **Incomplete statement** - `src/rawnind/dataset/noisy_datasets.py:198`
   - Code: `output["gain"]`  
   - Fix: `output["gain"] = 1.0`

2. **Unreachable code** - `src/rawnind/dataset/rgb_datasets.py:300-305`
   - Delete lines 300-305 after `return output` on line 299

3. **Missing import** - `src/rawnind/dataset/clean_api.py`
   - Add: `import random`

### Implementation Gaps (Phase 2-3)

4. **ConfigurableDataset not loading real data**
   - Current: Generates `torch.randn()` random tensors
   - Required: Translate YAML loading and image loading from 4 reference classes

5. **Missing integration function**
   - Name: `create_training_datasets` (plural)
   - Referenced: `training/clean_api.py:867`, `training/training_loops.py:963`
   - Purpose: Bridge clean API to training code

### Anti-Pattern Violations (Phase 4)

6. **Duplicate class definitions**
   - Same classes in multiple files
   - Violates: "Never have more than one implementation of the same thing"
   - Resolution: Delete duplicate files after translation

## Implementation Roadmap

### Phase 1: Fix Syntax Errors (10 min)
**Guide**: `phase1_syntax_fixes.md`
- 3 simple fixes to unblock compilation

### Phase 2: Translate ConfigurableDataset (180 min) ⭐ MAIN WORK
**Guide**: `phase2_translate_configurabledataset.md`
- Merge 4 legacy __init__ methods into _load_dataset()
- Merge 4 legacy __getitem__ methods into __getitem__() with branches
- Preserve ALL domain logic with bug fixes applied

### Phase 3: Integration Function (45 min)
**Guide**: `phase3_integration_function.md`
- Implement create_training_datasets()
- Connect to training code

### Phase 4: Delete Legacy Files (15 min)
**Guide**: `phase4_delete_legacy_files.md`  
- Remove duplicates and reference implementations
- Update __init__.py

### Phase 5: Simplify Wrappers (10 min)
**Guide**: `phase5_simplify_cleandataset.md`
- Simplify CleanDataset now that ConfigurableDataset contains all logic

**Total**: ~4.5 hours

## Domain Logic Preservation Requirements

ConfigurableDataset translation MUST preserve:

### Image Loading
- ✓ Load from YAML file metadata
- ✓ Load images using pt_helpers.fpath_to_tensor()
- ✓ Handle ValueError with retry on different image

### Quality Filtering
- ✓ MS-SSIM score thresholds (min/max)
- ✓ Alignment loss threshold (clean-noisy only)
- ✓ Mask mean threshold (clean-noisy only)
- ✓ Bayer-only filtering (skip non-Bayer images)
- ✓ Crop availability check

### Test/Train Splitting
- ✓ Test reserve filtering
- ✓ Training mode: exclude test_reserve images
- ✓ Testing mode: include only test_reserve images

### Image Processing
- ✓ Alignment shifts using rawproc.shift_images() (clean-noisy only)
- ✓ Mask loading from file (clean-noisy) or computation (clean-clean)
- ✓ Random crop selection with retry on insufficient valid pixels
- ✓ Dynamic dataset modification (remove bad crops/images)
- ✓ Arbitrary processing for experiments (if configured)

### Gain Handling
- ✓ Bayer datasets: use `raw_gain` from image metadata
- ✓ RGB datasets: use `rgb_gain` from image metadata
- ✓ match_gain=True: multiply y_crops by gain, set output gain=1.0
- ✓ match_gain=False: leave y_crops unchanged, set output gain=image gain

### Data Pairing Modes
- ✓ Mode x_y: gt vs noisy (standard)
- ✓ Mode x_x: gt vs gt_bayer/gt_rgb (self-supervised clean)
- ✓ Mode y_y: noisy vs noisy (self-supervised noisy)
- ✓ Different file paths per mode
- ✓ Mask from file (x_y) or torch.ones_like (x_x, y_y)

### Output Format
- ✓ Clean-noisy: Returns x_crops, y_crops, mask_crops, rgb_xyz_matrix (Bayer), gain
- ✓ Clean-clean Bayer: Returns x_crops, y_crops, mask_crops, rgb_xyz_matrix, gain
- ✓ Clean-clean RGB: Returns x_crops, mask_crops, gain (NO y_crops)

### Special Cases
- ✓ Toy dataset limiting (first 25 images)
- ✓ Crop coordinate sorting (deterministic for testing)
- ✓ Bayer pattern alignment (even coordinates)
- ✓ Resolution handling (Bayer 4-channel vs RGB 3-channel)

## Reference Implementation Locations

Use these for copying logic during translation:

### Clean-Clean + Bayer
- Primary: `legacy_rawds.py` lines 335-416
- Fallback: `src/rawnind/dataset/bayer_datasets.py` lines 37-119

### Clean-Clean + RGB
- Primary: `legacy_rawds.py` lines 419-503
- Fallback: `src/rawnind/dataset/rgb_datasets.py` lines 31-114

### Clean-Noisy + Bayer
- Primary: `legacy_rawds.py` lines 506-679
- Fallback: `src/rawnind/dataset/bayer_datasets.py` lines 128-307

### Clean-Noisy + RGB
- Primary: `legacy_rawds.py` lines 682-874 (has bug line 868-874)
- Fallback: `src/rawnind/dataset/rgb_datasets.py` lines 117-304 (same bug lines 299-305)
- **Bug**: Unreachable return statement - FIX during translation

## Files Changed Summary

### Modified
- `src/rawnind/dataset/clean_api.py` - Translate ConfigurableDataset, add create_training_datasets()
- `src/rawnind/dataset/__init__.py` - Update imports/exports
- `src/rawnind/dataset/noisy_datasets.py` - Fix syntax (then DELETE)
- `src/rawnind/dataset/rgb_datasets.py` - Fix syntax (then DELETE)

### Deleted
- `src/rawnind/dataset/clean_datasets.py` - All duplicates
- `src/rawnind/dataset/noisy_datasets.py` - After syntax fix
- `src/rawnind/dataset/bayer_datasets.py` - After translation complete
- `src/rawnind/dataset/rgb_datasets.py` - After translation complete
- Possibly: `validation_datasets.py`, `test_dataloaders.py` (if logic now in CleanValidationDataset/CleanTestDataset)

### Unchanged
- `src/rawnind/dataset/base_dataset.py` - Shared utilities (correct extraction)
- `src/rawnind/dataset/dataset_config.py` - Configuration classes

## Post-Completion State

After all phases:

```python
# Public API (src/rawnind/dataset/__init__.py)
from rawnind.dataset import (
    # Clean API (recommended)
    create_training_dataset,      # Single dataset
    create_validation_dataset,
    create_test_dataset,
    create_training_datasets,     # All three dataloaders (NEW)
    
    # Configuration
    DatasetConfig,
    DatasetMetadata,
    
    # Datasets (if needed directly)
    ConfigurableDataset,          # Unified implementation
    CleanDataset,                 # Wrapper with standardization
    CleanValidationDataset,
    CleanTestDataset,
    
    # Base classes (for extension)
    RawImageDataset,
    
    # Constants
    ALIGNMENT_MAX_LOSS,
    MASK_MEAN_MIN,
    TOY_DATASET_LEN,
)
```

No more imports from deleted legacy files.

## Next Steps After Completion

1. User validates tests pass
2. Code review for domain logic fidelity
3. Performance profiling (should match legacy performance)
4. Documentation update in README.md
5. Delete `legacy_rawds.py` (original source, now obsolete)

## Contact/Questions

If unclear during implementation:
- Re-read relevant collective memory entities
- Check phase-specific .md guides
- Compare with reference implementations in legacy_rawds.py
- Verify against domain logic preservation checklist