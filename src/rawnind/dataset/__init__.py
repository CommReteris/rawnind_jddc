"""Dataset package for RawNIND project.

This package contains all dataset-related functionality extracted from
the original rawds.py file, organized into logical modules for better
maintainability and modularity.

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

# Constants
MAX_MASKED = 0.5
MAX_RANDOM_CROP_ATTEMPS = 10
MASK_MEAN_MIN = 0.8
ALIGNMENT_MAX_LOSS = 0.035
OVEREXPOSURE_LB = 0.99
TOY_DATASET_LEN = 25
COLOR_PROFILE = "lin_rec2020"

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

    # Constants
    "MAX_MASKED",
    "MAX_RANDOM_CROP_ATTEMPS",
    "MASK_MEAN_MIN",
    "ALIGNMENT_MAX_LOSS",
    "OVEREXPOSURE_LB",
    "TOY_DATASET_LEN",
    "COLOR_PROFILE",
]
