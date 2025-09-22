# CLI to Clean API Migration Guide

## Overview

This guide helps users transition from the legacy CLI-based interfaces to the new clean programmatic APIs introduced in the RawNIND package refactoring. The clean APIs eliminate command-line parsing dependencies and provide modern, explicit configuration interfaces.

## 🎯 Original Problem Solved

**Before Refactoring:**
- `ImageToImageNN` classes retained legacy CLI dependencies (`--arch`, `--match_gain`, `--loss`)
- Command-line argument parsing was required even for programmatic usage
- Instantiation in test environments and integration scenarios failed due to CLI expectations
- Package interdependencies through shared CLI parsing infrastructure

**After Refactoring:**
- Clean factory functions eliminate CLI dependencies entirely
- Configuration classes provide explicit, typed parameters
- Zero CLI parsing required for any operation
- Complete interface modernization while preserving all functionality

## 📦 Package-by-Package Migration

### Inference Package

#### Before (Legacy CLI Interface):
```python
# ❌ Old way - required CLI arguments
from rawnind.inference.model_factory import get_and_load_test_object
import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument("--arch", required=True, choices=["unet", "utnet3"])
parser.add_argument("--in_channels", type=int, required=True, choices=[3, 4])
parser.add_argument("--config", required=True)
args = parser.parse_args()

# This would fail in programmatic contexts
test_obj = get_and_load_test_object()
```

#### After (Clean API):
```python
# ✅ New way - clean factory functions
from rawnind.inference import create_rgb_denoiser, create_bayer_denoiser, load_model_from_checkpoint

# Create models without CLI
rgb_denoiser = create_rgb_denoiser('unet', device='cpu')
bayer_denoiser = create_bayer_denoiser('unet', device='cpu')

# Load trained models with auto-detection
model_path = "path/to/model/directory"
trained_denoiser = load_model_from_checkpoint(model_path, device='cpu')
```

#### Available Factory Functions:
- `create_rgb_denoiser(architecture, device='cpu', filter_units=48, **kwargs)`
- `create_bayer_denoiser(architecture, device='cpu', filter_units=48, **kwargs)`
- `load_model_from_checkpoint(checkpoint_path, device='cpu', **kwargs)`
- `compute_image_metrics(predicted, ground_truth, metrics_list, mask=None)`

### Training Package

#### Before (Legacy CLI Interface):
```python
# ❌ Old way - CLI-based training setup
import configargparse
from rawnind.training.training_loops import ImageToImageNNTraining

parser = configargparse.ArgumentParser()
parser.add_argument("--arch", required=True)
parser.add_argument("--init_lr", type=float, required=True)
parser.add_argument("--tot_steps", type=int, required=True)
# ... many more required CLI arguments
args = parser.parse_args()

trainer = ImageToImageNNTraining()  # Would parse CLI internally
```

#### After (Clean API):
```python
# ✅ New way - explicit configuration
from rawnind.training import create_denoiser_trainer, TrainingConfig, create_experiment_manager, ExperimentConfig

# Explicit configuration replaces CLI parsing
config = TrainingConfig(
    model_architecture='unet',
    input_channels=3,
    output_channels=3,
    learning_rate=1e-4,
    batch_size=4,
    crop_size=128,
    total_steps=1000,
    validation_interval=100,
    loss_function='ms_ssim',
    device='cpu'
)

# Create trainer without CLI dependencies
trainer = create_denoiser_trainer('rgb_to_rgb', config)

# Set up experiment management
exp_config = ExperimentConfig(
    experiment_name='my_experiment',
    save_directory='/path/to/experiments',
    checkpoint_interval=100,
    keep_best_n_models=3
)
experiment_manager = create_experiment_manager(exp_config)
```

#### Available Factory Functions:
- `create_denoiser_trainer(training_type, config)`
- `create_denoise_compress_trainer(training_type, config)`
- `create_experiment_manager(config)`
- `TrainingConfig(...)` - Explicit configuration class
- `ExperimentConfig(...)` - Experiment management configuration

### Dataset Package

#### Before (Legacy Interface):
```python
# ❌ Old way - complex initialization through training classes
from rawnind.dataset.base_dataset import RawImageDataset
import configargparse

# Required complex setup with CLI parsing
args = configargparse.Namespace(
    noise_dataset_yamlfpaths=['path1.yaml'],
    clean_dataset_yamlfpaths=['path2.yaml'],
    # ... many parameters
)
dataset = RawImageDataset(args)
```

#### After (Clean API):
```python
# ✅ New way - clean factory functions
from rawnind.dataset import create_training_dataset, DatasetConfig

# Explicit configuration
config = DatasetConfig(
    dataset_type='rgb_pairs',
    data_format='clean_noisy',
    input_channels=3,
    output_channels=3,
    crop_size=128,
    batch_size=4,
    device='cpu'
)

# Create dataset without CLI dependencies
dataset = create_training_dataset(config)
```

#### Available Factory Functions:
- `create_training_dataset(config)`
- `create_validation_dataset(config)`
- `create_test_dataset(config)`
- `DatasetConfig(...)` - Explicit dataset configuration

### Dependencies Package

#### Before (Legacy Interface):
```python
# ❌ Old way - CLI script execution
import subprocess
subprocess.run(["python", "-m", "rawnind.dependencies.raw_processing", "--arg1", "value1"])
```

#### After (Clean API):
```python
# ✅ New way - class-based interface
from rawnind.dependencies.raw_processing import RawLoader, BayerProcessor, ColorTransformer, ProcessingConfig

# Configure and process
config = ProcessingConfig(force_rggb=True, crop_all=True)
loader = RawLoader(config)
mono, metadata = loader.load("input.raw")

processor = BayerProcessor(config)
rggb = processor.mono_to_rggb(mono, metadata)
```

## 🔧 Migration Patterns

### Pattern 1: Model Instantiation

**Before:**
```python
# Required CLI parsing even for simple instantiation
import sys
sys.argv = ['script.py', '--arch', 'unet', '--in_channels', '3', '--config', 'config.yaml']
model = SomeModelClass()
```

**After:**
```python
# Direct instantiation with explicit parameters
denoiser = create_rgb_denoiser('unet', device='cpu')
```

### Pattern 2: Configuration Management

**Before:**
```python
# CLI arguments scattered throughout codebase
parser.add_argument("--learning_rate", type=float, required=True)
parser.add_argument("--batch_size", type=int, required=True)
# ... dozens of arguments
args = parser.parse_args()
```

**After:**
```python
# Typed configuration classes
config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=4,
    model_architecture='unet',
    # All parameters explicit and validated
)
```

### Pattern 3: Model Loading

**Before:**
```python
# Complex CLI-based loading
sys.argv = ['script.py', '--config', 'model_dir/args.yaml', '--load_path', 'model_dir']
model = get_and_load_test_object()
```

**After:**
```python
# Simple checkpoint loading with auto-detection
denoiser = load_model_from_checkpoint('model_dir', device='cpu')
```

## 🚀 Benefits of Clean APIs

### 1. **Eliminates CLI Dependencies**
- No more `configargparse` or `argparse` requirements
- Works perfectly in Jupyter notebooks, tests, and integration scenarios
- No need to mock command-line arguments

### 2. **Type Safety and Validation**
- Configuration classes provide type hints and validation
- IDE auto-completion for all parameters
- Compile-time error detection

### 3. **Explicit Configuration**
- All parameters are explicitly specified
- No hidden defaults or implicit CLI parsing
- Clear documentation of all required parameters

### 4. **Better Testing**
- Easy to create test configurations
- No CLI mocking required
- Hermetic test execution

### 5. **Improved Documentation**
- Configuration classes serve as documentation
- Clear parameter types and validation rules
- Examples in docstrings

## 📋 Migration Checklist

### For Inference Usage:
- [ ] Replace `get_and_load_test_object()` with `create_rgb_denoiser()` or `create_bayer_denoiser()`
- [ ] Replace `ImageToImageNN(...)` with factory functions
- [ ] Use `load_model_from_checkpoint()` for trained models
- [ ] Replace custom metrics code with `compute_image_metrics()`

### For Training Usage:
- [ ] Replace CLI argument parsing with `TrainingConfig` class
- [ ] Use `create_denoiser_trainer()` instead of direct class instantiation
- [ ] Set up experiment management with `ExperimentConfig`
- [ ] Replace file-based configuration with typed configuration classes

### For Dataset Usage:
- [ ] Replace complex dataset initialization with `create_training_dataset()`
- [ ] Use `DatasetConfig` for explicit dataset parameters
- [ ] Eliminate CLI-based dataset configuration

### For Dependencies Usage:
- [ ] Replace CLI script calls with class-based APIs
- [ ] Use `ProcessingConfig` for RAW processing configuration
- [ ] Instantiate processing classes directly (`RawLoader`, `BayerProcessor`, etc.)

## 🔍 Troubleshooting

### Common Migration Issues:

#### Issue: "get_and_load_test_object is deprecated"
**Solution:** Use clean factory functions:
```python
# Instead of
test_obj = get_and_load_test_object()

# Use
denoiser = create_rgb_denoiser('unet', device='cpu')
# or
denoiser = load_model_from_checkpoint('model_path', device='cpu')
```

#### Issue: "Module requires CLI arguments"
**Solution:** The module is using legacy interfaces. Check for clean API alternatives:
```python
# Instead of importing modules that expect CLI
from rawnind.inference.model_factory import Denoiser

# Use clean factory functions
from rawnind.inference import create_rgb_denoiser
```

#### Issue: "Configuration not found"
**Solution:** Use explicit configuration classes:
```python
# Instead of
args = parse_args()  # Requires CLI

# Use
config = TrainingConfig(
    model_architecture='unet',
    learning_rate=1e-4,
    # ... explicit parameters
)
```

## 📈 Performance Benefits

The clean APIs provide several performance benefits:

1. **Faster Startup**: No CLI parsing overhead
2. **Memory Efficiency**: No argument parser objects retained
3. **Better Caching**: Configuration objects can be reused
4. **Parallel Execution**: No global CLI state conflicts

## 🎉 Summary

The CLI to Clean API migration provides:

- ✅ **Complete elimination of CLI dependencies** in core functionality
- ✅ **Modern, typed interfaces** with validation and documentation
- ✅ **Backward compatibility** where necessary through deprecation warnings
- ✅ **Enhanced testability** and integration capabilities
- ✅ **Preserved functionality** while improving developer experience

The original `ImageToImageNN` CLI instantiation problem is **completely solved** through the introduction of clean factory functions and configuration classes.

## 🔗 Reference Documentation

- **Inference Package**: See `src/rawnind/inference/clean_api.py` for complete API reference
- **Training Package**: See `src/rawnind/training/clean_api.py` for training interfaces
- **Dataset Package**: See `src/rawnind/dataset/clean_api.py` for dataset interfaces
- **Test Examples**: See `src/rawnind/tests/test_e2e_*_clean_api.py` for usage examples