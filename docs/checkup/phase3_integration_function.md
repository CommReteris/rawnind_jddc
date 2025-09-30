# Phase 3: Implement create_training_datasets Integration Function

## Context
Read from collective memory: "create_training_datasets Function", "Training Integration Pattern", "Clean API Integration Points"

## Objective
Implement the bridge function that connects clean API to training code.

## Prerequisites
Phase 2 must be complete (ConfigurableDataset fully translated).

## File to Modify
`src/rawnind/dataset/clean_api.py`

## Implementation

### Step 3.1: Add Function After Existing Factory Functions

**Location**: After `create_test_dataset()` function, before validation utility functions (around line 485)

**Add this complete function**:

```python
def create_training_datasets(
    input_channels: int,
    output_channels: int,
    crop_size: int,
    batch_size: int,
    clean_dataset_yamlfpaths: List[str],
    noise_dataset_yamlfpaths: List[str],
    test_reserve: List[str],
    **kwargs
) -> Dict[str, Any]:
    """Create train/validation/test dataloaders for training integration.
    
    This function bridges the clean API to training code by creating all three
    dataloaders with proper configuration. Supports both legacy dual-dataloader
    pattern and modern unified dataloader approach.
    
    Args:
        input_channels: Number of input channels (4=Bayer, 3=RGB)
        output_channels: Number of output channels  
        crop_size: Size of random crops to extract
        batch_size: Batch size for DataLoaders
        clean_dataset_yamlfpaths: Paths to clean dataset YAML files
        noise_dataset_yamlfpaths: Paths to noisy dataset YAML files
        test_reserve: List of image_set names reserved for testing
        **kwargs: Additional config (toy_dataset, match_gain, bayer_only, data_pairing, etc.)
        
    Returns:
        Dict with keys:
            - train_dataloader: PyTorch DataLoader for training
            - validation_dataloader: PyTorch DataLoader for validation
            - test_dataloader: PyTorch DataLoader for testing
    """
    # Determine data format and dataset type from parameters
    has_noisy = bool(noise_dataset_yamlfpaths)
    data_format = 'clean_noisy' if has_noisy else 'clean_clean'
    dataset_type = 'bayer_pairs' if input_channels == 4 else 'rgb_pairs'
    
    # Extract kwargs with sensible defaults
    num_crops_per_image = kwargs.get('num_crops_per_image', 1)
    toy_dataset = kwargs.get('toy_dataset', False)
    match_gain = kwargs.get('match_gain', False)
    bayer_only = kwargs.get('bayer_only', input_channels == 4)
    data_pairing = kwargs.get('data_pairing', 'x_y')
    arbitrary_proc_method = kwargs.get('arbitrary_proc_method', None)
    
    # Create nested configuration for dataset-specific parameters
    if input_channels == 4:
        from .dataset_config import BayerDatasetConfig
        nested_config = BayerDatasetConfig(is_bayer=True, bayer_only=bayer_only)
    else:
        from .dataset_config import RgbDatasetConfig
        nested_config = RgbDatasetConfig(is_bayer=False)
    
    # Build main dataset configuration
    base_config = DatasetConfig(
        dataset_type=dataset_type,
        data_format=data_format,
        input_channels=input_channels,
        output_channels=output_channels,
        crop_size=crop_size,
        num_crops_per_image=num_crops_per_image,
        batch_size=batch_size,
        test_reserve_images=test_reserve,
        match_gain=match_gain,
        max_samples=25 if toy_dataset else None,
        config=nested_config,
        content_fpaths=noise_dataset_yamlfpaths or clean_dataset_yamlfpaths
    )
    
    # Add optional attributes for data pairing and arbitrary processing
    if hasattr(base_config, 'data_pairing'):
        base_config.data_pairing = data_pairing
    if hasattr(base_config, 'arbitrary_proc_method'):
        base_config.arbitrary_proc_method = arbitrary_proc_method
    
    # Prepare data paths dictionary
    data_paths = {
        'noise_dataset_yamlfpaths': noise_dataset_yamlfpaths or clean_dataset_yamlfpaths
    }
    
    # Create training dataset (random crops, shuffled)
    train_config = DatasetConfig(**vars(base_config))
    train_config.save_individual_results = False  # Training mode
    train_config.center_crop = False
    train_dataset = create_training_dataset(train_config, data_paths)
    
    # Create validation dataset (center crop, deterministic)
    val_config = DatasetConfig(**vars(base_config))
    val_config.save_individual_results = False  # Not test mode
    val_config.center_crop = True
    val_config.num_crops_per_image = 1
    val_dataset = create_validation_dataset(val_config, data_paths)
    
    # Create test dataset (center crop, save results)
    test_config = DatasetConfig(**vars(base_config))
    test_config.save_individual_results = True  # Test mode
    test_config.center_crop = True
    test_config.num_crops_per_image = 1
    test_dataset = create_test_dataset(test_config, data_paths)
    
    # Wrap with PyTorch DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=kwargs.get('num_workers', 0)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=kwargs.get('num_workers', 0)
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=kwargs.get('num_workers', 0)
    )
    
    return {
        'train_dataloader': train_loader,
        'validation_dataloader': val_loader,
        'test_dataloader': test_loader
    }
```

### Step 3.2: Update Exports in __init__.py

**File**: `src/rawnind/dataset/__init__.py`

**Add to imports section**:
```python
from .clean_api import (
    # ... existing imports ...
    create_training_datasets,  # ADD THIS
)
```

**Add to __all__ list**:
```python
__all__ = [
    # ... existing exports ...
    'create_training_datasets',  # ADD THIS
]
```

## Verification

```bash
# Import check
python -c "from rawnind.dataset import create_training_datasets; print('✓ Function exported')"

# Type signature check
python -c "
from rawnind.dataset import create_training_datasets
import inspect
sig = inspect.signature(create_training_datasets)
print(f'Parameters: {list(sig.parameters.keys())}')
assert 'input_channels' in sig.parameters
assert 'noise_dataset_yamlfpaths' in sig.parameters
print('✓ Signature correct')
"

# Return type check  
python -c "
from rawnind.dataset.clean_api import create_training_datasets
result = create_training_datasets(
    input_channels=3,
    output_channels=3,
    crop_size=128,
    batch_size=1,
    clean_dataset_yamlfpaths=[],
    noise_dataset_yamlfpaths=[],
    test_reserve=[],
    toy_dataset=True
)
assert 'train_dataloader' in result
assert 'validation_dataloader' in result
assert 'test_dataloader' in result
print('✓ Return structure correct')
"
```

## Estimated Time
45 minutes