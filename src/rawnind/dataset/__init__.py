"""Dataset package for RawNIND project.

This package contains all dataset-related functionality extracted from
the original rawds.py file, organized into logical modules for better
maintainability and modularity.

Recommended Clean API (no CLI dependencies):
    from rawnind.dataset import (
        create_training_dataset,
        create_validation_dataset,
        create_test_dataset,
        DatasetConfig,
        DatasetMetadata,
        validate_dataset_format,
        prepare_dataset_splits
    )

Legacy API (available but complex):
    from rawnind.dataset import (
        RawImageDataset,              # Base classes
        CleanNoisyDataset,            # Complex class hierarchy
        ProfiledRGBBayerImageDataset  # Requires detailed knowledge
    )

Extracted from rawds.py as part of the codebase refactoring.
"""

# Clean modern API (recommended) - simplified interfaces
from .clean_api import (
    create_training_dataset,
    create_validation_dataset,
    create_test_dataset,
    DatasetConfig,
    DatasetMetadata,
    CleanDataset,
    CleanValidationDataset,
    CleanTestDataset,
    validate_dataset_format,
    prepare_dataset_splits,
    convert_dataset_format,
    create_dataset_config_from_yaml,
    load_rawnind_test_reserve_config,
    validate_training_type_and_dataset_config,
    ConfigurableDataset
)

# Legacy API - base dataset classes and utilities (more complex but complete)








# Clean dataset implementations


# Noisy dataset implementations




# Validation and test dataset implementations


# Constants
MAX_MASKED = 0.5
MAX_RANDOM_CROP_ATTEMPS = 10
MASK_MEAN_MIN = 0.8
ALIGNMENT_MAX_LOSS = 0.035
OVEREXPOSURE_LB = 0.99
TOY_DATASET_LEN = 25
COLOR_PROFILE = "lin_rec2020"


__all__ = [
    # Clean modern API (recommended)
    'create_training_dataset',
    'create_validation_dataset',
    'create_test_dataset',
    'DatasetConfig',
    'DatasetMetadata',
    'CleanDataset',
    'CleanValidationDataset',
    'CleanTestDataset',
    'validate_dataset_format',
    'prepare_dataset_splits',
    'convert_dataset_format',
    'create_dataset_config_from_yaml',
    'load_rawnind_test_reserve_config',
    'validate_training_type_and_dataset_config',
    'ConfigurableDataset',

    # Constants
    "MAX_MASKED",
    "MAX_RANDOM_CROP_ATTEMPS",
    "MASK_MEAN_MIN",
    "ALIGNMENT_MAX_LOSS",
    "OVEREXPOSURE_LB",
    "TOY_DATASET_LEN",
    "COLOR_PROFILE",
]
