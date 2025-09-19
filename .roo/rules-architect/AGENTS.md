# Project Architecture Rules (Non-Obvious Only)

## Core Components
- **abstract_trainer.py**: Centralizes PyTorch model management and training loops
- **pt_helpers.py**: Provides device abstraction and PyTorch utilities
- **rawtestlib.py**: Custom test framework for rapid testing

## Model Hierarchy
- Base: `ImageToImageNN` - handles device setup, logging, and model loading
- Intermediate: `BayerImageToImageNNTraining`, `PRGBImageToImageNNTraining`
- Specialized: `DenoiseCompressTraining`, `DenoiserTraining`

## PyTorch Integration
- **Device Management**: Centralized via `pt_helpers.get_device()`
- **Model Loading**: Standardized through `load_model()` static method
- **Inference**: Unified `infer()` method across all models

## Training Pipeline
- Argument parsing with configargparse and YAML configs
- Automatic experiment management (logging, checkpointing)
- Resource monitoring via psutil integration

## Testing Strategy
- Test classes inherit from training classes but override `get_dataloaders()`
- Use `offline_custom_test()` for controlled evaluation
- Focus on model initialization and forward pass validation

## Critical Patterns
- Always call `super().__init__()` in model subclasses
- Ensure proper device placement before tensor operations
- Use PyTorch's built-in evaluation mode during inference