"""RawNIND: Source code for learning joint denoising, demosaicing, and compression from raw images.

This package contains implementations of neural network models and utilities for processing
raw camera images. The code supports the research presented in "Learning Joint Denoising,
Demosaicing, and Compression from the Raw Natural Image Noise Dataset".

The package has two main components:
    - common: Shared utilities and libraries used across the project
    - rawnind: Core functionality for raw image processing and neural networks

The implementation focuses on:
    - Processing both Bayer and X-Trans sensor data
    - Joint denoising, demosaicing, and compression of raw images
    - Training and evaluation of neural networks for image processing tasks
    - Dataset handling for paired (clean-noisy) and unpaired (clean-clean) image sets

For detailed usage instructions, refer to the README files in the repository.

"""


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
