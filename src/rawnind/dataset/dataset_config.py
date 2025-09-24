"""
This module defines the configuration classes for the dataset package.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BayerDatasetConfig:
    """Configuration specific to Bayer datasets."""
    is_bayer: bool = True
    bayer_only: bool = True


@dataclass
class RgbDatasetConfig:
    """Configuration specific to RGB datasets."""
    is_bayer: bool = False

@dataclass
class DatasetConfig:
    """Configuration for dataset creation with explicit parameters."""

    # Dataset type and format
    dataset_type: str  # "rgb_pairs", "bayer_pairs", "self_supervised", "rawnind_academic", "hdr_pairs"
    data_format: str  # "clean_noisy", "clean_clean", "bayer_rgb_pairs"

    # Image specifications
    input_channels: int
    output_channels: int
    crop_size: int
    num_crops_per_image: int
    batch_size: int
    content_fpaths: List[str] = field(default_factory=list)

    # Color and processing
    color_profile: str = "lin_rec2020"
    device: str = "cpu"
    match_gain: bool = False

    # Augmentation
    augmentations: List[str] = field(default_factory=list)
    augmentation_probability: float = 0.5

    # Quality control
    min_valid_pixels_ratio: float = 0.8
    max_crop_attempts: int = 10

    # Dataset splitting
    validation_split: float = 0.2
    test_reserve_images: List[str] = field(default_factory=list)
    enforce_test_reserve: bool = True

    # Performance
    lazy_loading: bool = True
    cache_size: int = 100  # Number of images to cache
    cache_size_mb: Optional[int] = None  # Alternative: cache size in MB
    num_workers: int = 1

    # Advanced options
    center_crop: bool = False  # Use center crop instead of random
    save_individual_results: bool = False
    compute_statistics: bool = False
    analyze_noise_levels: bool = False
    enable_caching: bool = False
    cache_strategy: str = "lru"

    # Format-specific options
    file_format: str = "exr"  # "exr", "jpg", "png", "tiff"
    dynamic_range: str = "sdr"  # "sdr", "hdr"
    tone_mapping: str = "none"  # "none", "reinhard", "aces"

    # Bayer-specific
    demosaicing_method: str = "bilinear"
    bayer_pattern: str = "RGGB"
    maintain_bayer_alignment: bool = True

    # Color space
    input_color_profile: str = "lin_rec2020"
    output_color_profile: str = "lin_rec2020"
    apply_color_conversion: bool = False

    # Noise handling
    noise_augmentation: Optional[str] = None  # "synthetic", "real", None
    handle_missing_files: str = "skip"  # "skip", "error"

    # Academic dataset specific
    academic_dataset_path: Optional[str] = None
    academic_dataset_version: str = "v1.0"
    load_camera_metadata: bool = False
    test_reserve_config_path: Optional[str] = None

    # Quality control thresholds
    quality_checks: List[str] = field(default_factory=list)
    quality_thresholds: dict[str, float] = field(default_factory=dict)

    # Processing pipeline
    preprocessing_steps: List[str] = field(default_factory=list)

    # Dataset size limits
    max_samples: Optional[int] = None
    config: Optional[BayerDatasetConfig | RgbDatasetConfig] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.crop_size <= 0 or self.crop_size % 2 != 0:
            raise ValueError("Crop size must be positive and even")
        if self.num_crops_per_image <= 0:
            raise ValueError("Number of crops per image must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.input_channels <= 0:
            raise ValueError("Input channels must be positive")
        if self.output_channels <= 0:
            raise ValueError("Output channels must be positive")

        # Validate dataset type and channel compatibility
        if self.dataset_type == "bayer_pairs" and self.input_channels != 4:
            raise ValueError("Bayer datasets require 4 input channels")
        if self.dataset_type == "rgb_pairs" and self.input_channels != 3:
            raise ValueError("RGB datasets require 3 input channels")

        # Set reasonable defaults
        defaults = {
            "max_alignment_error"    : 0.035,
            "max_overexposure_ratio" : 0.01,
            "min_image_quality_score": 0.7,
            "max_image_quality_score": 1.0
        }
        # Update with provided thresholds, keeping defaults for missing keys
        defaults.update(self.quality_thresholds)
        self.quality_thresholds = defaults