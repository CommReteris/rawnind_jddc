"""Inference package for model loading, prediction, and evaluation.

This package contains all components related to model inference, including:
- Model loading and management
- Inference engines
- Evaluation and metrics
- Deployment utilities
<<<<<<< HEAD
"""

from .base_inference import ImageToImageNN, BayerImageToImageNN
from .inference_engine import InferenceEngine
from .model_factory import get_and_load_test_object, get_and_load_model
from .model_loader import ModelLoader

__all__ = [
    'ImageToImageNN',
    'BayerImageToImageNN',
    'ModelLoader',
=======

Clean API (recommended):
- create_rgb_denoiser(), create_bayer_denoiser(), create_compressor()
- load_model_from_checkpoint()
- compute_image_metrics()

Legacy API (deprecated):
- ImageToImageNN, BayerImageToImageNN (CLI-dependent)
- get_and_load_test_object(), get_and_load_model() (CLI-dependent)
"""

# Clean modern API (no CLI dependencies)
from .clean_api import (
    create_rgb_denoiser,
    create_bayer_denoiser,
    create_compressor,
    load_model_from_checkpoint,
    compute_image_metrics,
    list_available_models,
    find_best_model_in_directory,
    InferenceConfig,
    ModelCheckpoint,
    CleanDenoiser,
    CleanBayerDenoiser,
    CleanCompressor
)

# Legacy CLI-dependent API (deprecated)
from .base_inference import ImageToImageNN, BayerImageToImageNN
from .inference_engine import InferenceEngine
from .model_factory import get_and_load_test_object, get_and_load_model


__all__ = [
    # Clean modern API (recommended)
    'create_rgb_denoiser',
    'create_bayer_denoiser', 
    'create_compressor',
    'load_model_from_checkpoint',
    'compute_image_metrics',
    'list_available_models',
    'find_best_model_in_directory',
    'InferenceConfig',
    'ModelCheckpoint',
    'CleanDenoiser',
    'CleanBayerDenoiser',
    'CleanCompressor',
    
    # Legacy API (deprecated - will be removed)
    'ImageToImageNN',
    'BayerImageToImageNN',
    
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
    'InferenceEngine',
    'get_and_load_test_object',
    'get_and_load_model'
]
