"""Training package for neural network training loops and optimization.

This package contains all components related to model training, including:
- Training loops and optimization routines
- Hyperparameter tuning
- Experiment management
- Model training classes
<<<<<<< HEAD
"""

from .experiment_manager import ExperimentManager
from .training_loops import TrainingLoops

__all__ = [
=======

Recommended Clean API (no CLI dependencies):
    from rawnind.training import (
        create_denoiser_trainer,
        create_denoise_compress_trainer, 
        create_experiment_manager,
        TrainingConfig,
        ExperimentConfig
    )

Legacy API (deprecated - will be removed):
    from rawnind.training import (
        TrainingLoops,           # CLI-dependent
        ExperimentManager        # CLI-dependent
    )
"""

# Clean modern API (recommended) - no CLI dependencies
from .clean_api import (
    create_denoiser_trainer,
    create_denoise_compress_trainer,
    create_experiment_manager,
    TrainingConfig,
    ExperimentConfig,
    CleanTrainer,
    CleanDenoiserTrainer,
    CleanDenoiseCompressTrainer,
    CleanExperimentManager,
    validate_training_type_and_config,
    create_training_config_from_yaml
)

# Legacy API (deprecated - contains CLI dependencies)
from .experiment_manager import ExperimentManager
from .training_loops import TrainingLoops

# Export both APIs with clear distinction
__all__ = [
    # Clean modern API (recommended)
    'create_denoiser_trainer',
    'create_denoise_compress_trainer', 
    'create_experiment_manager',
    'TrainingConfig',
    'ExperimentConfig',
    'CleanTrainer',
    'CleanDenoiserTrainer', 
    'CleanDenoiseCompressTrainer',
    'CleanExperimentManager',
    'validate_training_type_and_config',
    'create_training_config_from_yaml',
    
    # Legacy API (deprecated)
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
    'TrainingLoops',
    'ExperimentManager'
]
