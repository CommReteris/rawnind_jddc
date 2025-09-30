# Phase 4: Delete Legacy Files

## Context
Read from collective memory: "File Organization Issues", "Dataset Class Location Map", "Legacy Dataset Classes"

## Objective
Remove all duplicate class definitions and legacy reference implementation files after translation is complete.

## Prerequisites
- Phase 2 must be complete (ConfigurableDataset fully translated)
- Phase 3 must be complete (create_training_datasets implemented)
- Verify ConfigurableDataset contains all logic before deleting reference implementations

## Files to Delete

### Immediate Deletion (Duplicates)

**File 1**: `src/rawnind/dataset/clean_datasets.py`
- Reason: ALL classes are duplicates of those in bayer_datasets.py and rgb_datasets.py
- Classes lost: CleanProfiledRGBCleanBayerImageCropsDataset (duplicate), CleanProfiledRGBCleanProfiledRGBImageCropsDataset (duplicate)
- Impact: NONE - duplicates already exist elsewhere
- **Action**: `rm src/rawnind/dataset/clean_datasets.py`

**File 2**: `src/rawnind/dataset/noisy_datasets.py`
- Reason: CleanProfiledRGBNoisyBayerImageCropsDataset is duplicate, has incomplete gain bug
- Classes lost: CleanProfiledRGBNoisyBayerImageCropsDataset (duplicate)
- Impact: NONE - working version in bayer_datasets.py
- **Action**: `rm src/rawnind/dataset/noisy_datasets.py`

### Post-Translation Deletion (Reference Implementations)

**File 3**: `src/rawnind/dataset/bayer_datasets.py`
- Reason: Reference implementation now fully translated to ConfigurableDataset
- Classes lost: CleanProfiledRGBCleanBayerImageCropsDataset, CleanProfiledRGBNoisyBayerImageCropsDataset, ProfiledRGBBayerImageDataset
- Impact: Logic preserved in ConfigurableDataset translation
- **Action**: `rm src/rawnind/dataset/bayer_datasets.py`

**File 4**: `src/rawnind/dataset/rgb_datasets.py`  
- Reason: Reference implementation now fully translated to ConfigurableDataset
- Classes lost: CleanProfiledRGBCleanProfiledRGBImageCropsDataset, CleanProfiledRGBNoisyProfiledRGBImageCropsDataset, ProfiledRGBProfiledRGBImageDataset
- Impact: Logic preserved in ConfigurableDataset translation
- Note: Has unreachable code bug (lines 300-305)
- **Action**: `rm src/rawnind/dataset/rgb_datasets.py`

## Files to Update After Deletion

### Update __init__.py

**File**: `src/rawnind/dataset/__init__.py`

**Remove these imports**:
```python
# DELETE all imports from:
# - .clean_datasets
# - .noisy_datasets  
# - .bayer_datasets
# - .rgb_datasets
```

**Keep only**:
```python
from .clean_api import (
    create_training_dataset,
    create_validation_dataset,
    create_test_dataset,
    create_training_datasets,  # Should be added in Phase 3
    DatasetConfig,
    DatasetMetadata,
    CleanDataset,
    CleanValidationDataset,
    CleanTestDataset,
    ConfigurableDataset,
    # ... other clean_api exports
)

from .base_dataset import (
    RawImageDataset,
    CleanCleanImageDataset,
    CleanNoisyDataset,
    RawDatasetOutput,
    # ... constants
)
```

**Update __all__ to export only from clean_api and base_dataset**

### Check validation_datasets.py

**File**: `src/rawnind/dataset/validation_datasets.py`

**Current imports** (lines 31-32):
```python
from .bayer_datasets import CleanProfiledRGBNoisyBayerImageCropsDataset
from .rgb_datasets import CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
```

**Options**:
1. If validation logic already in CleanValidationDataset: DELETE this file
2. If additional logic needed: Update imports or translate into CleanValidationDataset

**Action**: Review file, likely DELETE (logic in CleanValidationDataset)

### Check test_dataloaders.py

**File**: `src/rawnind/dataset/test_dataloaders.py`

**Current imports** (lines 15-16):
```python
from .bayer_datasets import CleanProfiledRGBNoisyBayerImageCropsDataset
from .rgb_datasets import CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
```

**Options**:
1. If test logic already in CleanTestDataset: DELETE this file
2. If get_images() iterator needed: Translate into CleanTestDataset

**Action**: Review file, likely DELETE (logic in CleanTestDataset with get_images support)

## Execution Order

1. Delete clean_datasets.py (duplicates)
2. Delete noisy_datasets.py (duplicates)  
3. Update validation_datasets.py and test_dataloaders.py OR delete them
4. Delete bayer_datasets.py (reference impl)
5. Delete rgb_datasets.py (reference impl)
6. Update __init__.py imports
7. Verify no broken imports

## Verification

```bash
# Check no broken imports
python -c "from rawnind.dataset import *; print('✓ All imports work')"

# Check no duplicate class definitions
grep -r "class CleanProfiledRGB" src/rawnind/dataset/*.py | wc -l
# Should output: 0 (all in ConfigurableDataset now)

# Check files deleted
ls src/rawnind/dataset/clean_datasets.py 2>&1 | grep "No such file"
ls src/rawnind/dataset/noisy_datasets.py 2>&1 | grep "No such file"
ls src/rawnind/dataset/bayer_datasets.py 2>&1 | grep "No such file"
ls src/rawnind/dataset/rgb_datasets.py 2>&1 | grep "No such file"

# Check clean_api exports work
python -c "
from rawnind.dataset import (
    ConfigurableDataset,
    CleanDataset,
    create_training_dataset,
    create_training_datasets
)
print('✓ Clean API exports verified')
"
```

## Estimated Time
15 minutes