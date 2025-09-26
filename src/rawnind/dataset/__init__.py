"""Dataset package for RawNIND project.

This package contains all dataset-related functionality extracted from
the original rawds.py file, organized into logical modules for better
maintainability and modularity.

<<<<<<< HEAD
Extracted from rawds.py as part of the codebase refactoring.
"""

# Base dataset classes and utilities
from .base_dataset import (CleanCleanImageDataset, CleanNoisyDataset, ProfiledRGBBayerImageDataset,
                           ProfiledRGBProfiledRGBImageDataset, RawDatasetOutput, RawImageDataset, TestDataLoader)
# Clean dataset implementations
from .clean_datasets import (
    CleanProfiledRGBCleanBayerImageCropsDataset,
    CleanProfiledRGBCleanProfiledRGBImageCropsDataset,
)
# Noisy dataset implementations
from .noisy_datasets import (
    CleanProfiledRGBNoisyBayerImageCropsDataset,
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset,
)
# Validation and test dataset implementations
from .validation_datasets import (CleanProfiledRGBNoisyBayerImageCropsTestDataloader,
                                  CleanProfiledRGBNoisyBayerImageCropsValidationDataset,
                                  CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader,
                                  CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset)
=======
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

>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

# Constants
MAX_MASKED = 0.5
MAX_RANDOM_CROP_ATTEMPS = 10
MASK_MEAN_MIN = 0.8
ALIGNMENT_MAX_LOSS = 0.035
OVEREXPOSURE_LB = 0.99
TOY_DATASET_LEN = 25
COLOR_PROFILE = "lin_rec2020"

<<<<<<< HEAD
__all__ = [
    # Base classes
    "RawImageDataset",
    "RawDatasetOutput",
    "ProfiledRGBBayerImageDataset",
    "ProfiledRGBProfiledRGBImageDataset",
    "CleanCleanImageDataset",
    "CleanNoisyDataset",
    "TestDataLoader",

    # Clean dataset implementations
    "CleanProfiledRGBCleanBayerImageCropsDataset",
    "CleanProfiledRGBCleanProfiledRGBImageCropsDataset",

    # Noisy dataset implementations
    "CleanProfiledRGBNoisyBayerImageCropsDataset",
    "CleanProfiledRGBNoisyProfiledRGBImageCropsDataset",

    # Validation and test dataset implementations
    "CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset",
    "CleanProfiledRGBNoisyBayerImageCropsValidationDataset",
    "CleanProfiledRGBNoisyBayerImageCropsTestDataloader",
    "CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader",
=======

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
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

    # Constants
    "MAX_MASKED",
    "MAX_RANDOM_CROP_ATTEMPS",
    "MASK_MEAN_MIN",
    "ALIGNMENT_MAX_LOSS",
    "OVEREXPOSURE_LB",
    "TOY_DATASET_LEN",
    "COLOR_PROFILE",
]
