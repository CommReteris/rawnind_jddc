# RawNIND Dataset Module

This module provides a comprehensive, configurable framework for loading and processing image datasets in the RawNIND project. It supports raw Bayer, profiled RGB, and clean/noisy image pairs, with built-in alignment, masking, cropping, and augmentation capabilities. The design emphasizes modularity, lossless refactoring from legacy code, and YAML-driven configuration for flexibility in training and inference pipelines.

## Key Features

- **ConfigurableDataset Base Class**: Central abstraction for all datasets. Initialized via YAML configs (see [dataset_config.py](dataset_config.py)) specifying crop sizes, number of crops, channel counts, pairing modes ('x_y' for clean-noisy, 'x_x' clean-clean, 'y_y' noisy-noisy), gains, and quality thresholds.
- **Bayer and RGB Support**: 
  - Bayer datasets ([bayer_datasets.py](bayer_datasets.py)): Handle 4-channel RGGB mosaics with white balance, demosaicing, and camRGB to profiled RGB conversion (lin_rec2020 default).
  - RGB datasets ([rgb_datasets.py](rgb_datasets.py)): 3-channel profiled RGB (lin_rec2020) with optional gain matching.
- **Clean and Noisy Variants**: 
  - Clean datasets ([clean_datasets.py](clean_datasets.py)): For ground truth or clean-clean pairs.
  - Noisy datasets ([noisy_datasets.py](noisy_datasets.py)): For noisy inputs in denoising tasks.
- **Alignment and Masking**: Sub-pixel alignment using pre-computed offsets ([raw_processing.py](dependencies/raw_processing.py#shift_images)). Overexposure masks exclude saturated pixels (threshold 0.99 default).
- **Cropping and Augmentation**: Random crops maintain Bayer pattern (even indices); center crops for validation. Arbitrary processing via [arbitrary_processing.py](dependencies/arbitrary_processing.py) for augmentations (e.g., noise addition, flips).
- **Validation and Test Loaders**: [validation_datasets.py](validation_datasets.py) for deterministic center crops; [test_dataloaders.py](test_dataloaders.py) for full-image or sliding window evaluation.
- **Outputs**: Standardized `RawDatasetOutput` with tensors (x_crops clean, y_crops input, mask_crops valid pixels), color matrices, and gains.
- **Quality Control**: Filters images by alignment loss (max 0.035), mask mean (min 0.8), MSSSIM scores during init.

## Usage

### 1. YAML Configuration
Define datasets in YAML (e.g., `dataset.yaml`):
```yaml
dataset:
  type: bayer_clean_noisy
  root_dir: /path/to/rawNIND
  crop_size: 256
  num_crops: 8
  batch_size: 4
  input_channels: 4  # Bayer
  output_channels: 3  # Profiled RGB
  data_pairing: x_y
  match_gain: true
  bayer_only: true
  toy_dataset: 25  # For debugging
  arbitrary_proc_method: noise_add  # Optional augmentation
  quality:
    alignment_max_loss: 0.035
    mask_mean_min: 0.8
    msssim_min: 0.5
    msssim_max: 0.95
```

Load via clean API ([clean_api.py](clean_api.py)):
```python
from rawnind.dataset.clean_api import create_dataset

config = load_yaml('dataset.yaml')
ds = create_dataset(config)
train_loader = DataLoader(ds, batch_size=config['batch_size'], shuffle=True)
```

### 2. Training/Inference Integration
- **Training**: Use with [training.clean_api](https://github.com/rawnind/training/clean_api.py) for denoiser or compress trainers. Datasets auto-align gains and apply masks in loss computation.
- **Inference**: Compatible with [inference.clean_api](https://github.com/rawnind/inference/clean_api.py). Load full images for evaluation via `TestDataLoader`.
- **E2E Example**:
  ```python
  # Toy training loop
  for batch in train_loader:
      x, y, masks, matrices, gain = batch  # RawDatasetOutput unpacked
      outputs = model(y)  # e.g., denoiser
      loss = criterion(outputs, x, masks)  # Masked MS-SSIM or MSE
  ```

### 3. Augmentation Centralization
Augmentations centralized in [dependencies/arbitrary_processing.py](dependencies/arbitrary_processing.py). Pass `arbitrary_proc_fun` to ConfigurableDataset for custom pipelines (e.g., random noise, rotations). Ensures matched gains across clean/noisy pairs.

## Migration Notes (from Legacy)
- **Legacy Removal**: `legacy_rawds.py` and legacy functions in `bayer_datasets.py`/`clean_datasets.py` are deprecated post-refactor. Update imports to new classes (e.g., `ProfiledRGBBayerImageDataset` replaces old loaders).
- **Lossless Verification**: All 13 domain logic observations ported (see memory graph). Tests confirm equivalence via golden comparisons (torch.allclose(rtol=1e-5)) on toy/full RawNIND subsets.
- **Breaking Changes**:
  - No more global constants; use YAML/config.
  - Alignment/masking now explicit in dataset init (pre-computed YAML metadata).
  - Outputs include `rgb_xyz_matrix` for color transforms.
- **Performance**: Load times benchmarked ([test_datasets_load_time.py](tests/test_datasets_load_time.py)); random crops retry up to 10 attempts for valid masks (>50% valid pixels).
- **Edge Cases**: Handles overexposure, invalid alignments, toy limits. Filters non-Bayer if `bayer_only=True`.

## Testing
- Unit: [tests/](tests/) for cropping, masking, configs.
- Integration: E2E legacy vs. refactored outputs.
- Acceptance: Base contracts, no anti-patterns ([test_anti_patterns.py](tests/test_anti_patterns.py)).
Run: `python -m pytest src/rawnind/dataset/ -v`

## Dependencies
- rawpy, OpenEXR/OpenImageIO for RAW loading.
- PyTorch for tensors/losses.
- YAML for configs ([dataset_config.py](dataset_config.py)).

For issues, see [project README](../README.md).