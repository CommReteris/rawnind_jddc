# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Build/Lint/Test Commands
- **Run all tests**: `python src/rawnind/tools/test_all_known.py --tests test_manproc --model_types dc`
- **Run a single test**: `python src/rawnind/tests/test_manproc_dc_bayer2prgb.py --config path/to/config.yaml`
- **Train a model**: `python src/rawnind/train_dc_bayer2prgb.py --config path/to/config.yaml`

## Code Style Guidelines
- **Imports**: Group imports by source (standard library, third-party, project)
- **PyTorch Device Handling**: Always use `pt_helpers.get_device()` for device selection
- **Model Initialization**: Use the abstract_trainer's `instantiate_model()` method
- **Tensor Operations**: Prefer PyTorch tensor operations over NumPy where possible
- **Error Handling**: Use custom `error_handler()` function for critical errors

## PyTorch Integration
- **Device Management**: Use `pt_helpers.get_device()` to get the current device
- **Model Loading**: Use `ImageToImageNN.load_model()` static method
- **Inference**: Models have an `infer()` method that handles device placement
- **Transfer Functions**: Use `get_transfer_function()` for gamma/PQ encoding

## Testing Framework
- **Test Classes**: Extend from `rawtestlib.*` classes for quick testing
- **Custom Dataloaders**: Override `get_dataloaders()` to return None for fast tests
- **Test Execution**: Use `offline_custom_test()` method for controlled testing
- **Assertions**: Verify model outputs using PyTorch tensor assertions

## Project Structure
- **src/rawnind/models/**: Contains all PyTorch model implementations
- **src/rawnind/libs/abstract_trainer.py**: Core training framework with PyTorch integration
- **src/rawnind/tests/**: Test files following naming patterns like `test_manproc_dc_bayer2prgb.py`
- **config/**: YAML configuration files for different training scenarios

## Critical Patterns
- **Model Initialization**: Always call `super().__init__()` in model classes
- **Device Consistency**: Ensure tensors are moved to the correct device before operations
- **Batch Handling**: Single images should be converted to batches with `.unsqueeze(0)`
- **Evaluation Mode**: Use `model.eval()` during inference

## Testing Best Practices
- **Unit Tests**: Create small, focused tests that verify specific functionality
- **Integration Tests**: Use the custom test framework for end-to-end testing
- **Mocking**: Mock external dependencies while preserving PyTorch model behavior
- **Coverage**: Focus on testing model initialization, forward passes, and key methods

This guide provides essential information for working with this project's PyTorch-based image processing pipeline.