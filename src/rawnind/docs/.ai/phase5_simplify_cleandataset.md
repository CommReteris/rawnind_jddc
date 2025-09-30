# Phase 5: Simplify CleanDataset

## Context
Read from collective memory: "Clean Solution Architecture", "Integration Architecture", "Batch Format Standardization"

## Objective
Simplify CleanDataset wrapper now that ConfigurableDataset contains all domain logic.

## Prerequisites
Phase 2 complete (ConfigurableDataset translated)

## File to Modify
`src/rawnind/dataset/clean_api.py`

## Changes Required

### Change 5.1: Simplify _create_underlying_dataset() Method

**Location**: Lines ~158-175 in CleanDataset class

**Current code**:
```python
def _create_underlying_dataset(self):
    """Create the appropriate underlying dataset based on configuration."""
    if self._data_loader_override:
        self._underlying_dataset = self._data_loader_override
        return

    # Select appropriate dataset class based on configuration
    if "bayer" in self.config.dataset_type:
        self._underlying_dataset = ConfigurableDataset(self.config, self.data_paths)
    elif "rgb" in self.config.dataset_type:
        self._underlying_dataset = ConfigurableDataset(self.config, self.data_paths)
    elif self.config.dataset_type == "rawnind_academic":
        self._underlying_dataset = ConfigurableDataset(self.config, self.data_paths)
    else:
        raise ValueError(f"Unsupported dataset type: {self.config.dataset_type}")
    if len(self._underlying_dataset) == 0:
        raise ValueError("No images found in the dataset.")
```

**Replace with** (simplified):
```python
def _create_underlying_dataset(self):
    """Create ConfigurableDataset (now contains all translated logic)."""
    if self._data_loader_override:
        self._underlying_dataset = self._data_loader_override
        return

    # ConfigurableDataset handles all dataset types via config-driven branching
    self._underlying_dataset = ConfigurableDataset(self.config, self.data_paths)
    
    if len(self._underlying_dataset) == 0:
        raise ValueError("No images found in dataset")
```

**Reason**: ConfigurableDataset now handles all cases internally - no need for selection logic

### Change 5.2: Verify _standardize_batch_format() Handles Clean-Clean

**Location**: Lines ~197-248 in CleanDataset class

**Check this method handles**:
1. Dict input with keys: `x_crops`, `y_crops` (optional), `mask_crops`, `rgb_xyz_matrix` (optional), `gain`
2. Maps to: `clean_images`, `noisy_images`, `masks`, `rgb_xyz_matrices`, `gain`
3. Handles missing `y_crops` for clean-clean RGB datasets

**Verify mapping logic**:
```python
def _standardize_batch_format(self, batch: Any) -> Dict[str, Any]:
    """Standardize batch format to consistent structure."""
    if isinstance(batch, dict):
        standardized = batch.copy()
    elif isinstance(batch, (tuple, list)):
        # Handle tuple format
        if len(batch) >= 3:
            standardized = {
                'noisy_images': batch[0],  # x_crops
                'clean_images': batch[1],  # y_crops (may be None)
                'masks': batch[2],  # mask_crops
            }
            if len(batch) > 3:
                standardized['rgb_xyz_matrices'] = batch[3]
        else:
            raise ValueError(f"Unexpected batch format: {batch}")
    else:
        raise ValueError(f"Unknown batch type: {type(batch)}")
    
    # Rename legacy keys to clean API keys
    if 'x_crops' in standardized:
        standardized['clean_images'] = standardized.pop('x_crops')
    if 'y_crops' in standardized:
        # y_crops may be None for clean-clean RGB
        standardized['noisy_images'] = standardized.pop('y_crops')
    if 'mask_crops' in standardized:
        standardized['masks'] = standardized.pop('mask_crops')
    if 'rgb_xyz_matrix' in standardized:
        standardized['rgb_xyz_matrices'] = standardized.pop('rgb_xyz_matrix')
    
    # Add expected metadata...
```

**If missing y_crops handling, add**:
```python
# Handle clean-clean RGB (no y_crops)
if 'noisy_images' not in standardized and 'y_crops' not in batch:
    # Clean-clean RGB: use clean_images for both
    standardized['noisy_images'] = standardized['clean_images']
```

## Verification

```bash
# Test with mock data
python -c "
from rawnind.dataset.clean_api import CleanDataset, DatasetConfig
from rawnind.dataset.dataset_config import BayerDatasetConfig

# Create mock override
def mock_loader():
    yield {
        'x_crops': torch.randn(1, 3, 128, 128),
        'y_crops': torch.randn(1, 3, 128, 128),
        'mask_crops': torch.ones(1, 3, 128, 128, dtype=torch.bool),
        'gain': 1.0
    }

config = DatasetConfig(
    dataset_type='rgb_pairs',
    data_format='clean_noisy',
    input_channels=3,
    output_channels=3,
    crop_size=128,
    num_crops_per_image=1,
    batch_size=1,
    test_reserve_images=[]
)

ds = CleanDataset(config, {}, data_loader_override=mock_loader())
batch = next(iter(ds))

assert 'clean_images' in batch
assert 'noisy_images' in batch
assert 'masks' in batch
print('✓ Batch format standardization works')
"
```

## Estimated Time
10 minutes