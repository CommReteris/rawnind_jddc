# Registry Pattern Extensions for RawNind

Based on my analysis of the codebase, I've identified several elegant patterns that could complement your existing
augmentations pipeline approach. The augmentations pipeline is indeed very elegant - it uses a global registry (
`AUGMENTATION_REGISTRY`) with a configurable pipeline that supports deterministic seeding and GPU acceleration.

## Current Registry Patterns in the Codebase

Your codebase already uses similar registry patterns in several places:

1. **Augmentations Registry** (`src/rawnind/dependencies/augmentations.py`)
2. **Loss Functions Registry** (`src/rawnind/dependencies/pt_losses.py`) - has `losses` and `metrics` dictionaries
3. **Multiple Training Classes** - suggesting a training strategy pattern

## Suggested New Registry Patterns

### 1. Model Registry and Factory Pattern

**Problem**: Multiple model types (denoising, compression, etc.) are instantiated manually throughout the codebase.

**Solution**: Create a unified model registry similar to augmentations.

```python
# src/rawnind/dependencies/model_registry.py
import torch
from typing import Dict, Any, Type
from ..models.raw_denoiser import UtNet2, ImageDenoisingModel
from ..models.compression_autoencoders import BalleEncoder, BalleDecoder
from ..models.bm3d_denoiser import BM3DDenoiser

# Global registry of available models
MODEL_REGISTRY = {
    'utnet2': UtNet2,
    'bm3d': BM3DDenoiser,
    'balle_encoder': BalleEncoder,
    'balle_decoder': BalleDecoder,
}


class ModelPipeline:
    """Configurable model instantiation pipeline."""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.model_configs = config
        self.models = {}

    def create_model(self, model_name: str, **override_params) -> torch.nn.Module:
        """Create a model instance from registry."""
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")

        model_class = MODEL_REGISTRY[model_name]
        params = self.model_configs.get(model_name, {})
        params.update(override_params)

        return model_class(**params)

    def create_pipeline(self, model_sequence: list) -> torch.nn.Sequential:
        """Create a sequential pipeline of models."""
        models = [self.create_model(name) for name in model_sequence]
        return torch.nn.Sequential(*models)

# Usage example:
# config = {
#     'utnet2': {'in_channels': 4, 'out_channels': 3},
#     'balle_encoder': {'num_filters': 192}
# }
# pipeline = ModelPipeline(config)
# denoiser = pipeline.create_model('utnet2')
```

### 2. Quality Checks Registry and Pipeline

**Problem**: DatasetConfig has a `quality_checks` field but no systematic way to apply them.

**Solution**: Create a quality checks pipeline similar to augmentations.

```python
# src/rawnind/dependencies/quality_checks.py
import torch
from typing import Dict, Any, Callable


def check_overexposure(image: torch.Tensor, threshold: float = 0.01) -> Dict[str, Any]:
    """Check for overexposed pixels."""
    overexposed = (image >= 1.0).float().mean()
    return {
        'overexposure_ratio': overexposed.item(),
        'passed': overexposed <= threshold
    }


def check_underexposure(image: torch.Tensor, threshold: float = 0.1) -> Dict[str, Any]:
    """Check for underexposed pixels."""
    underexposed = (image <= 0.0).float().mean()
    return {
        'underexposure_ratio': underexposed.item(),
        'passed': underexposed <= threshold
    }


def check_noise_level(image: torch.Tensor, max_std: float = 0.1) -> Dict[str, Any]:
    """Estimate noise level."""
    # Simple noise estimation using local variance
    kernel = torch.ones(1, 1, 3, 3, device=image.device) / 9
    smoothed = torch.nn.functional.conv2d(image, kernel, padding=1)
    noise_var = ((image - smoothed) ** 2).mean()
    return {
        'estimated_noise_std': noise_var.sqrt().item(),
        'passed': noise_var.sqrt() <= max_std
    }


# Global registry of quality checks
QUALITY_CHECKS_REGISTRY = {
    'overexposure': check_overexposure,
    'underexposure': check_underexposure,
    'noise_level': check_noise_level,
}


class QualityChecksPipeline:
    """Configurable image quality assessment pipeline."""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.checks = []
        for check_name, params in config.items():
            if check_name not in QUALITY_CHECKS_REGISTRY:
                raise ValueError(f"Unknown quality check: {check_name}")
            self.checks.append((QUALITY_CHECKS_REGISTRY[check_name], params))

    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        """Apply all configured quality checks."""
        results = {}
        all_passed = True

        for check_fn, params in self.checks:
            result = check_fn(image, **params)
            check_name = check_fn.__name__.replace('check_', '')
            results[check_name] = result
            all_passed = all_passed and result['passed']

        results['overall_passed'] = all_passed
        return results
```

### 3. Preprocessing Steps Registry and Pipeline

**Problem**: DatasetConfig has a `preprocessing_steps` field but no systematic pipeline.

**Solution**: Create a preprocessing pipeline similar to augmentations.

```python
# src/rawnind/dependencies/preprocessing_steps.py
import torch
from typing import Dict, Any


def normalize_image(image: torch.Tensor, mean: list, std: list) -> torch.Tensor:
    """Normalize image with given mean and std."""
    mean = torch.tensor(mean, device=image.device).view(-1, 1, 1)
    std = torch.tensor(std, device=image.device).view(-1, 1, 1)
    return (image - mean) / std


def gamma_correction(image: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Apply gamma correction."""
    return torch.pow(image.clamp(min=1e-8), 1.0 / gamma)


def white_balance(image: torch.Tensor, wb_gains: list) -> torch.Tensor:
    """Apply white balance gains."""
    gains = torch.tensor(wb_gains, device=image.device).view(-1, 1, 1)
    return image * gains


def demosaic_bilinear(bayer: torch.Tensor, pattern: str = 'RGGB') -> torch.Tensor:
    """Simple bilinear demosaicing for Bayer patterns."""
    # Simplified implementation - in practice use a proper demosaicing algorithm
    if bayer.shape[1] == 4:  # Already demosaiced to 4 channels
        # Convert 4-channel to 3-channel RGB
        r = bayer[:, 0:1]  # Red
        g = (bayer[:, 1:2] + bayer[:, 2:3]) / 2  # Average of two greens
        b = bayer[:, 3:4]  # Blue
        return torch.cat([r, g, b], dim=1)
    return bayer


# Global registry of preprocessing steps
PREPROCESSING_REGISTRY = {
    'normalize': normalize_image,
    'gamma_correction': gamma_correction,
    'white_balance': white_balance,
    'demosaic': demosaic_bilinear,
}


class PreprocessingPipeline:
    """Configurable image preprocessing pipeline."""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.steps = []
        for step_name, params in config.items():
            if step_name not in PREPROCESSING_REGISTRY:
                raise ValueError(f"Unknown preprocessing step: {step_name}")
            self.steps.append((PREPROCESSING_REGISTRY[step_name], params))

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Apply all configured preprocessing steps."""
        for step_fn, params in self.steps:
            image = step_fn(image, **params)
        return image
```

### 4. Training Strategy Registry

**Problem**: Multiple training classes (DenoiserTraining, PRGBImageToImageNNTraining) with similar patterns.

**Solution**: Create a training strategy registry.

```python
# src/rawnind/dependencies/training_strategies.py
from typing import Dict, Any, Type
from ..training.training_loops import (
    ImageToImageNNTraining,
    PRGBImageToImageNNTraining,
    DenoiserTraining,
    DenoiseCompressTraining
)

# Global registry of training strategies
TRAINING_STRATEGIES_REGISTRY = {
    'image_to_image': ImageToImageNNTraining,
    'prgb_image_to_image': PRGBImageToImageNNTraining,
    'denoiser': DenoiserTraining,
    'denoise_compress': DenoiseCompressTraining,
}


class TrainingStrategyFactory:
    """Factory for creating training strategy instances."""

    @staticmethod
    def create_trainer(strategy_name: str, config: Any) -> Any:
        """Create a trainer instance from registry."""
        if strategy_name not in TRAINING_STRATEGIES_REGISTRY:
            raise ValueError(f"Unknown training strategy: {strategy_name}")

        trainer_class = TRAINING_STRATEGIES_REGISTRY[strategy_name]
        return trainer_class(config)

    @staticmethod
    def list_available_strategies() -> list:
        """List all available training strategies."""
        return list(TRAINING_STRATEGIES_REGISTRY.keys())
```

### 5. Enhanced Transfer Function Registry

**Problem**: Training loops have `get_transfer_function` but limited extensibility.

**Solution**: Extend with a proper registry pattern.

```python
# src/rawnind/dependencies/transfer_functions.py
import torch
from typing import Dict, Any


def identity_transfer(x: torch.Tensor) -> torch.Tensor:
    """Identity transfer function."""
    return x


def log_transfer(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Logarithmic transfer function."""
    return torch.log(x + epsilon)


def gamma_transfer(x: torch.Tensor, gamma: float = 2.2) -> torch.Tensor:
    """Gamma transfer function."""
    return torch.pow(x.clamp(min=1e-8), 1.0 / gamma)


def sigmoid_transfer(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Sigmoid transfer function."""
    return torch.sigmoid(x * scale)


# Global registry of transfer functions
TRANSFER_FUNCTIONS_REGISTRY = {
    'identity': identity_transfer,
    'log': log_transfer,
    'gamma': gamma_transfer,
    'sigmoid': sigmoid_transfer,
}


class TransferFunctionPipeline:
    """Configurable transfer function pipeline."""

    def __init__(self, config: Dict[str, Dict[str, Any]]):
        self.functions = []
        for func_name, params in config.items():
            if func_name not in TRANSFER_FUNCTIONS_REGISTRY:
                raise ValueError(f"Unknown transfer function: {func_name}")
            self.functions.append((TRANSFER_FUNCTIONS_REGISTRY[func_name], params))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply configured transfer functions in sequence."""
        for func, params in self.functions:
            x = func(x, **params)
        return x
```

## Benefits of These Patterns

1. **Consistency**: All use the same registry pattern as augmentations
2. **Extensibility**: Easy to add new components without modifying core code
3. **Configuration-driven**: Can be configured via YAML/JSON files
4. **Testability**: Each component can be tested independently
5. **Composability**: Components can be combined in pipelines
6. **Maintainability**: Clear separation of concerns

## Integration with Existing Code

These patterns would integrate seamlessly with your existing DatasetConfig:

```python
# Enhanced DatasetConfig usage
config = DatasetConfig(
    # Existing fields...
    augmentations=['flip', 'rotate'],
    preprocessing_steps=['normalize', 'gamma_correction'],
    quality_checks=['overexposure', 'noise_level'],
    # New fields for other patterns
    model_pipeline=['utnet2'],
    training_strategy='denoiser',
    transfer_functions=['gamma'],
)
```

The patterns maintain the same elegant philosophy as your augmentations pipeline: registry-based, configurable, and
pipeline-oriented.