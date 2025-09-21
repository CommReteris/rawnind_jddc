"""RawNIND package.

A modular PyTorch-based framework for RAW image denoising and compression.
This package provides a clean, maintainable architecture for training and inference
of neural networks on RAW image data.

Subpackages:
- rawnind.dependencies: Shared utilities, configurations, and common functionality
- rawnind.dataset: Dataset loading, preprocessing, and data handling
- rawnind.training: Training loops, optimization, and experiment management
- rawnind.inference: Model inference, loading, and deployment utilities
- rawnind.models: Neural network model definitions and architectures
- rawnind.tools: Command-line utilities and dataset preparation scripts
- rawnind.tests: Comprehensive test suite

Import submodules explicitly where needed, e.g.:
    from rawnind.dependencies import raw_processing
    from rawnind.training import training_loops
    from rawnind.inference import model_factory
"""

# Intentionally avoid eager imports to keep `import rawnind` lightweight
# and avoid requiring optional runtime dependencies at import time.

__all__ = ["dependencies", "dataset", "training", "inference", "models", "tools", "tests"]
