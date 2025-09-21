"""Training package for neural network training loops and optimization.

This package contains all components related to model training, including:
- Training loops and optimization routines
- Hyperparameter tuning
- Experiment management
- Model training classes
"""

from .experiment_manager import ExperimentManager
from .training_loops import TrainingLoops

__all__ = [
    'TrainingLoops',
    'ExperimentManager'
]
