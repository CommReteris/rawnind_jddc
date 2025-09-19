# Project Documentation Rules (Non-Obvious Only)

## Code Organization
- **src/rawnind/models/**: Contains PyTorch model implementations, not web app code
- **config/**: YAML files for training configurations, not runtime settings

## Key Components
- **abstract_trainer.py**: Core framework with PyTorch integration
- **rawtestlib.py**: Custom test classes that override dataloaders
- **pt_helpers.py**: Device management and PyTorch utilities

## Testing Infrastructure
- **Test Files**: Follow naming pattern `test_manproc_dc_bayer2prgb.py`
- **Test Runner**: Use `test_all_known.py` for comprehensive testing
- **Custom Dataloaders**: Return None in test subclasses for fast execution

## PyTorch Integration
- Models inherit from `torch.nn.Module`
- Training loops use PyTorch's autograd and optimizer functions
- Device management handled by `pt_helpers.get_device()`

## Configuration System
- Uses configargparse for command-line argument parsing
- YAML files in config/ directory override default values