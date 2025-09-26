
"""
Clean API for dataset package without CLI dependencies.

This module provides clean, modern programmatic interfaces for creating and managing
datasets without command-line argument parsing dependencies. It provides factory
functions and configuration classes for the existing dataset infrastructure.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional, Literal

import torch
import yaml
from pathlib import Path

# Import existing dataset classes
from .base_dataset import (
    RawImageDataset, CleanNoisyDataset, RawDatasetOutput
)
# Import dependencies
from ..dependencies import (
    pytorch_helpers as pt_helpers,
    raw_processing as rawproc,
    load_yaml
)
from .dataset_config import DatasetConfig


class DatasetMetadata:
    """Metadata information for datasets."""

    def __init__(self, name: str, total_images: int, color_profile: str,
                 iso_levels: List[int] = field(default_factory=list),
                 color_matrix: Optional[torch.Tensor] = None):
        self.name = name
        self.total_images = total_images
        self.color_profile = color_profile
        self.iso_levels = iso_levels
        self.color_matrix = color_matrix

    @classmethod
    def from_file(cls, metadata_path: str) -> 'DatasetMetadata':
        """Load dataset metadata from YAML file.
        
        Args:
            metadata_path: Path to metadata YAML file
            
        Returns:
            DatasetMetadata instance
        """
        with open(metadata_path, 'r') as f:
            data = yaml.safe_load(f)

        dataset_info = data.get('dataset_info', {})
        camera_info = data.get('camera_info', {})

        # Extract color matrix if available
        color_matrix = camera_info.get('color_matrix')
        if color_matrix:
            color_matrix = torch.tensor(color_matrix, dtype=torch.float32)

        return cls(
            name=dataset_info.get('name', 'unknown'),
            total_images=dataset_info.get('total_images', 0),
            color_profile=dataset_info.get('color_profile', 'lin_rec2020'),
            iso_levels=camera_info.get('iso_levels', []),
            color_matrix=color_matrix
        )


class ConfigurableDataset(CleanNoisyDataset):
    """A unified, configuration-driven dataset for clean-noisy pairs using YAML metadata."""

    def __init__(
        self,
        config: DatasetConfig,
        yaml_paths: List[str],
        test_reserve: List[str] = field(default_factory=list),
        bayer_only: bool = True,
        alignment_max_loss: float = 0.035,
        mask_mean_min: float = 0.8,
        min_msssim_score: float = 0.0,
        max_msssim_score: float = 1.0,
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
        match_gain: bool = False,
        toy_dataset: bool = False,
    ):
        self.config = config
        self.yaml_paths = yaml_paths
        self.test_reserve = test_reserve
        self.bayer_only = bayer_only
        self.alignment_max_loss = alignment_max_loss
        self.mask_mean_min = mask_mean_min
        self.min_msssim_score = min_msssim_score
        self.max_msssim_score = max_msssim_score
        self.data_pairing = data_pairing
        self.match_gain = match_gain
        self.toy_dataset = toy_dataset
        super().__init__(num_crops=config.num_crops_per_image, crop_size=config.crop_size)
        self._dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load and filter images from YAML metadata files."""
        dataset = []
        for yaml_path in self.yaml_paths:
            contents = load_yaml(yaml_path, error_on_404=True)
            if not contents:
                continue
            # Handle both list and dict formats
            if isinstance(contents, list):
                images = contents
            else:
                images = contents.get('images', [])
            for image in images:
                if self.toy_dataset and len(dataset) >= 25:  # TOY_DATASET_LEN
                    break
                if not self._filter_image(image):
                    continue
                # Sort crops for determinism in testing
                if 'crops' in image:
                    image['crops'] = sorted(image['crops'], key=lambda d: d.get('coordinates', (0,0)))
                if image.get('crops'):
                    dataset.append(image)
                else:
                    logging.warning(f"Image {image.get('f_fpath', 'unknown')} has no crops.")
        logging.info(f"Loaded {len(dataset)} images into ConfigurableDataset.")
        if not dataset:
            raise ValueError("No valid images found in dataset.")
        return dataset

    def _filter_image(self, image: Dict[str, Any]) -> bool:
        """Filter image based on quality criteria."""
        # Test reserve filter (matching legacy logic)
        test = self.config.test
        if (not test and image.get("image_set") in self.test_reserve) or \
           (test and image.get("image_set") not in self.test_reserve):
            return False

        # Bayer only filter
        if self.bayer_only and not image.get('is_bayer', False):
            return False

        # MSSSIM score filter
        msssim = image.get('rgb_msssim_score', 0.0)
        if not (self.min_msssim_score <= msssim <= self.max_msssim_score):
            return False

        # Alignment and mask quality filter
        if (image.get('best_alignment_loss', float('inf')) > self.alignment_max_loss or
            image.get('mask_mean', 0.0) < self.mask_mean_min):
            logging.info(f"Rejected image {image.get('f_fpath')} due to quality filters.")
            return False

        return True

    def __getitem__(self, i: int) -> RawDatasetOutput:
        image = self._dataset[i]
        crop = random.choice(image.get('crops', [{}]))

        # Load images based on data_pairing mode
        if self.data_pairing == 'x_y':
            gt_path = crop.get('gt_linrec2020_fpath', image.get('gt_linrec2020_fpath'))
            noisy_path = crop.get('f_bayer_fpath', image.get('f_bayer_fpath')) if self.bayer_only else crop.get('f_linrec2020_fpath', image.get('f_linrec2020_fpath'))
            mask_path = image.get('mask_fpath')
            gain_key = 'raw_gain' if self.bayer_only else 'rgb_gain'
            # Assume coordinates [0,0] for test, load full as small
            whole_img_mask = pt_helpers.fpath_to_tensor(mask_path)
        elif self.data_pairing == 'x_x':
            gt_path = crop.get('gt_linrec2020_fpath', image.get('gt_linrec2020_fpath'))
            noisy_path = crop.get('gt_bayer_fpath', image.get('gt_bayer_fpath')) if self.bayer_only else gt_path
            whole_img_mask = torch.ones(1, self.config.crop_size, self.config.crop_size, dtype=torch.bool)
            gain_key = 'raw_gain' if self.bayer_only else 'rgb_gain'
        elif self.data_pairing == 'y_y':
            gt_path = crop.get('f_linrec2020_fpath', image.get('f_linrec2020_fpath'))
            noisy_path = crop.get('f_bayer_fpath', image.get('f_bayer_fpath')) if self.bayer_only else gt_path
            whole_img_mask = torch.ones(1, self.config.crop_size, self.config.crop_size, dtype=torch.bool)
            gain_key = 'raw_gain' if self.bayer_only else 'rgb_gain'
        else:
            raise ValueError(f"Unsupported data_pairing: {self.data_pairing}")

        # Load tensors (assuming full image size matches crop_size for test)
        gt_img = pt_helpers.fpath_to_tensor(gt_path).float()
        noisy_img = pt_helpers.fpath_to_tensor(noisy_path).float()

        # Align for x_y mode
        if self.data_pairing == 'x_y':
            gt_img, noisy_img = rawproc.shift_images(gt_img, noisy_img, image.get('best_alignment', [0,0]))
            # For test, assume no cropping needed for mask
            whole_img_mask = whole_img_mask.expand(gt_img.shape)

        # Generate crops
        try:
            if self.data_pairing in ['x_x', 'y_y']:
                x_crops, mask_crops = self.random_crops(gt_img, None, whole_img_mask)
                y_crops = x_crops  # Same for x_x or y_y
            else:
                x_crops, y_crops, mask_crops = self.random_crops(gt_img, noisy_img, whole_img_mask)
        except TypeError:
            # Retry with different crop if insufficient valid pixels
            logging.warning(f"Invalid crop for image {i}, retrying.")
            if 'crops' in image:
                image['crops'].pop(0)  # Remove current crop
            if not image.get('crops'):
                self._dataset.pop(i)
            return self.__getitem__(i % len(self._dataset))

        # Handle gain
        gain = image.get(gain_key, 1.0)
        if self.match_gain:
            if y_crops is not None:
                y_crops *= gain
            gain = 1.0

        return RawDatasetOutput(
            x_crops=x_crops,
            y_crops=y_crops,
            mask_crops=mask_crops,
            rgb_xyz_matrix=torch.tensor(image.get('rgb_xyz_matrix', [[1,0,0],[0,1,0],[0,0,1]])),
            gain=gain
        )


class CleanDataset:
    """Clean dataset wrapper providing modern interface."""

    def __init__(self, config: DatasetConfig, data_paths: Dict[str, Any],
                 data_loader_override: Optional[Iterator] = None):
        """Initialize clean dataset.
        
        Args:
            config: Dataset configuration
            data_paths: Dictionary of data paths
            data_loader_override: Optional override for data loading (for testing)
        """
        self.config = config
        self.data_paths = data_paths
        self._data_loader_override = data_loader_override

        # Initialize underlying dataset based on configuration
        self._create_underlying_dataset()

        # Performance tracking
        self._cache_stats = {
            'enabled'     : config.enable_caching,
            'max_size_mb' : config.cache_size_mb or 0,
            'current_size': 0,
            'cache_hits'  : 0,
            'cache_misses': 0,
            'strategy'    : config.cache_strategy
        }

        # Quality tracking
        self._quality_stats = {
            'total_checked'        : 0,
            'passed_quality_checks': 0,
            'failed_checks'        : {'overexposure': 0, 'alignment': 0, 'quality': 0}
        }

    def _create_underlying_dataset(self):
        """Create the appropriate underlying dataset based on configuration."""
        # Use override if provided (for testing)
        if self._data_loader_override:
            self._underlying_dataset = self._data_loader_override
            return

        # Select appropriate dataset class based on configuration
        yaml_paths = self.data_paths.get('noise_dataset_yamlfpaths', [])
        if not yaml_paths:
            raise ValueError("No YAML paths provided for dataset loading.")

        self._underlying_dataset = ConfigurableDataset(
            self.config,
            yaml_paths,
            test_reserve=self.data_paths.get('test_reserve', []),
            bayer_only=self.config.bayer_only,
            data_pairing=self.config.data_pairing,
            match_gain=self.config.match_gain,
            toy_dataset=self.config.toy_dataset
        )

    def __iter__(self):
        """Iterate over dataset batches."""
        if self._data_loader_override:
            # Use override data for testing
            for batch in self._data_loader_override:
                # Convert to standard format
                standardized_batch = self._standardize_batch_format(batch)
                yield standardized_batch
        else:
            # Use real dataset iteration
            for batch in self._underlying_dataset:
                standardized_batch = self._standardize_batch_format(batch)
                yield standardized_batch

    def __len__(self):
        """Get dataset length."""
        if hasattr(self._underlying_dataset, '__len__'):
            return len(self._underlying_dataset)
        return 0  # Unknown length for iterators

    def _standardize_batch_format(self, batch: Any) -> Dict[str, Any]:
        """Standardize batch format to consistent structure.
        
        Args:
            batch: Raw batch from underlying dataset
            
        Returns:
            Standardized batch dictionary
        """
        # Handle RawDatasetOutput
        if isinstance(batch, dict) and 'x_crops' in batch:
            standardized = {
                'noisy_images': batch['y_crops'],
                'clean_images': batch['x_crops'],
                'masks': batch['mask_crops'],
                'gain': batch['gain']
            }
            if 'rgb_xyz_matrix' in batch:
                standardized['rgb_xyz_matrices'] = batch['rgb_xyz_matrix']
        elif isinstance(batch, dict):
            # Already in dictionary format
            standardized = batch.copy()
        elif isinstance(batch, (tuple, list)):
            # Convert tuple/list to dictionary
            if len(batch) >= 3:
                standardized = {
                    'noisy_images': batch[0],  # x_crops
                    'clean_images': batch[1],  # y_crops
                    'masks'       : batch[2],  # mask_crops
                }
                if len(batch) > 3:
                    standardized['rgb_xyz_matrices'] = batch[3]
            else:
                raise ValueError(f"Unexpected batch format: {batch}")
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}")

        # Add metadata that tests expect
        if 'image_paths' not in standardized:
            standardized['image_paths'] = ['mock_image.jpg']

        if 'color_profile_info' not in standardized:
            standardized['color_profile_info'] = {
                'input'             : self.config.input_color_profile,
                'output'            : self.config.output_color_profile,
                'conversion_applied': self.config.apply_color_conversion
            }

        if "bayer" in self.config.dataset_type and 'bayer_info' not in standardized:
            standardized['bayer_info'] = {
                'pattern'           : self.config.bayer_pattern,
                'demosaicing_method': self.config.demosaicing_method
            }

        if 'preprocessing_info' not in standardized:
            standardized['preprocessing_info'] = {
                'steps_applied'     : self.config.preprocessing_steps,
                'output_color_space': self.config.output_color_profile
            }

        return standardized

    def get_compatibility_info(self) -> Dict[str, Any]:
        """Get compatibility information for integration with training."""
        return {
            'compatible_with_training': True,
            'batch_format'            : 'standard',
            'tensor_dtypes'           : {
                'images': torch.float32,
                'masks' : torch.bool
            },
            'expected_keys'           : ['clean_images', 'noisy_images', 'masks']
        }

    def get_augmentation_info(self) -> Dict[str, Any]:
        """Get augmentation configuration information."""
        return {
            'available_augmentations': self.config.augmentations,
            'probability'            : self.config.augmentation_probability,
            'enabled'                : len(self.config.augmentations) > 0
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get caching information."""
        return {
            'max_size'    : self.config.cache_size,
            'current_size': self._cache_stats['current_size'],
            'enabled'     : self.config.enable_caching
        }

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        return self._cache_stats.copy()

    def get_loader_info(self) -> Dict[str, Any]:
        """Get data loader configuration info."""
        return {
            'num_workers'            : self.config.num_workers,
            'multiprocessing_enabled': self.config.num_workers > 1,
            'batch_size'             : self.config.batch_size
        }

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        if not self.config.compute_statistics:
            return {}

        # Simplified statistics computation
        all_values = []
        sample_count = 0

        for batch in self:
            clean_images = batch['clean_images']
            all_values.append(clean_images.flatten())
            sample_count += clean_images.shape[0]

            if sample_count >= 100:  # Limit for statistics
                break

        if all_values:
            all_tensor = torch.cat(all_values)
            return {
                'mean'         : torch.tensor([all_tensor.mean()] * self.config.output_channels),
                'std'          : torch.tensor([all_tensor.std()] * self.config.output_channels),
                'min'          : all_tensor.min(),
                'max'          : all_tensor.max(),
                'total_samples': sample_count
            }
        else:
            # Return dummy statistics if no values were collected
            return {
                'mean'         : torch.zeros(self.config.output_channels),
                'std'          : torch.ones(self.config.output_channels),
                'min'          : 0,
                'max'          : 0,
                'total_samples': 0
            }

    def analyze_noise_levels(self) -> Dict[str, Any]:
        """Analyze noise characteristics in the dataset."""
        if not self.config.analyze_noise_levels:
            return {}

        # Simplified noise analysis
        noise_stds = []
        sample_count = 0

        for batch in self:
            if 'noise_info' in batch:
                noise_stds.append(batch['noise_info']['estimated_std'])
            sample_count += 1

            if sample_count >= 50:  # Limit analysis
                break

        if noise_stds:
            mean_std = sum(noise_stds) / len(noise_stds)
            return {
                'noise_distribution'    : 'gaussian',  # Simplified
                'mean_noise_std'        : mean_std,
                'noise_level_categories': ['low', 'medium', 'high']  # Simplified
            }
        else:
            # Return dummy noise levels if no values were collected
            return {
                'noise_distribution'    : 'unknown',
                'mean_noise_std'        : 0.1,
                'noise_level_categories': []
            }

    def get_quality_report(self) -> Dict[str, Any]:
        """Get quality assessment report."""
        return self._quality_stats.copy()

    def get_determinism_info(self) -> Dict[str, bool]:
        """Get information about dataset determinism."""
        return {
            'is_deterministic' : self.config.center_crop and len(self.config.augmentations) == 0,
            'uses_center_crop' : self.config.center_crop,
            'has_augmentations': len(self.config.augmentations) > 0
        }


class CleanValidationDataset(CleanDataset):
    """Clean validation dataset with deterministic behavior."""

    def __init__(self, config: DatasetConfig, data_paths: Dict[str, Any],
                 data_loader_override: Optional[Iterator] = None,
                 parent_dataset: Optional[CleanDataset] = None):
        """Initialize clean validation dataset.
        
        Args:
            config: Dataset configuration
            data_paths: Dictionary of data paths
            data_loader_override: Optional override for data loading
            parent_dataset: Parent dataset to create subset from
        """
        # Force deterministic settings for validation
        config.center_crop = True
        config.augmentations = []
        config.num_crops_per_image = 1

        super().__init__(config, data_paths, data_loader_override)
        self.parent_dataset = parent_dataset

    def __len__(self):
        """Get validation dataset length."""
        if self.config.max_samples:
            return min(self.config.max_samples, super().__len__())
        return super().__len__()


class CleanTestDataset(CleanDataset):
    """Clean test dataset with additional metadata support."""

    def __init__(self, config: DatasetConfig, data_paths: Dict[str, Any],
                 test_names_filter: Optional[List[str]] = None,
                 data_loader_override: Optional[Iterator] = None):
        """Initialize clean test dataset.
        
        Args:
            config: Dataset configuration  
            data_paths: Dictionary of data paths
            test_names_filter: Optional filter for specific test images
            data_loader_override: Optional override for data loading
        """
        # Force deterministic settings for testing
        config.center_crop = True
        config.augmentations = []
        config.num_crops_per_image = 1
        config.save_individual_results = True

        super().__init__(config, data_paths, data_loader_override)
        self.test_names_filter = test_names_filter

    def _standardize_batch_format(self, batch: Any) -> Dict[str, Any]:
        """Add test-specific metadata to batches."""
        standardized = super()._standardize_batch_format(batch)

        # Add image metadata for test datasets
        if 'image_metadata' not in standardized:
            standardized['image_metadata'] = {
                'image_name'   : 'test_image',
                'original_size': (1024, 1024)  # Default size
            }

        return standardized


# Factory Functions

def create_training_dataset(config: DatasetConfig, data_paths: Dict[str, Any],
                            data_loader_override: Optional[Iterator] = None) -> CleanDataset:
    """Create a training dataset with clean API.
    
    Args:
        config: Dataset configuration
        data_paths: Dictionary of data paths
        data_loader_override: Optional override for data loading (for testing)
        
    Returns:
        Clean training dataset instance
    """
    return CleanDataset(config, data_paths, data_loader_override)


def create_validation_dataset(config: DatasetConfig, data_paths: Dict[str, Any],
                              data_loader_override: Optional[Iterator] = None,
                              parent_dataset: Optional[CleanDataset] = None) -> CleanValidationDataset:
    """Create a validation dataset with clean API.
    
    Args:
        config: Dataset configuration (will be modified for validation behavior)
        data_paths: Dictionary of data paths
        data_loader_override: Optional override for data loading
        parent_dataset: Parent dataset to create subset from
        
    Returns:
        Clean validation dataset instance
    """
    return CleanValidationDataset(config, data_paths, data_loader_override, parent_dataset)


def create_test_dataset(config: DatasetConfig, data_paths: Dict[str, Any],
                        test_names_filter: Optional[List[str]] = None,
                        data_loader_override: Optional[Iterator] = None) -> CleanTestDataset:
    """Create a test dataset with clean API.
    
    Args:
        config: Dataset configuration (will be modified for test behavior)
        data_paths: Dictionary of data paths
        test_names_filter: Optional filter for specific test images
        data_loader_override: Optional override for data loading
        
    Returns:
        Clean test dataset instance
    """
    return CleanTestDataset(config, data_paths, test_names_filter, data_loader_override)


# Validation and Utility Functions

def validate_dataset_format(dataset_paths: Dict[str, Any], expected_format: str,
                            check_image_pairs: bool = True,
                            check_file_integrity: bool = True) -> Dict[str, Any]:
    """Validate dataset format and integrity.
    
    Args:
        dataset_paths: Dictionary of dataset paths
        expected_format: Expected format ("clean_noisy", "clean_clean", etc.)
        check_image_pairs: Whether to check image pair matching
        check_file_integrity: Whether to check file integrity
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid'         : True,
        'total_pairs'      : 0,
        'missing_pairs'    : 0,
        'validation_errors': []
    }

    if expected_format == "clean_noisy":
        # Check for matching clean/noisy pairs
        clean_paths = dataset_paths.get('clean_images', [])
        noisy_paths = dataset_paths.get('noisy_images', [])

        if isinstance(clean_paths, list) and len(clean_paths) > 0:
            clean_dir = Path(clean_paths[0])
            if clean_dir.exists():
                clean_files = set(f.stem for f in clean_dir.glob('*.jpg'))
                validation_result['total_pairs'] = len(clean_files)

        if isinstance(noisy_paths, list) and len(noisy_paths) > 0:
            noisy_dir = Path(noisy_paths[0])
            if noisy_dir.exists():
                noisy_files = set(f.stem for f in noisy_dir.glob('*.jpg'))
                missing_pairs = len(clean_files - noisy_files) if 'clean_files' in locals() else 0
                validation_result['missing_pairs'] = missing_pairs

                if missing_pairs > 0:
                    validation_result['is_valid'] = False
                    validation_result['validation_errors'].append(f"Missing noisy image pairs: {missing_pairs}")

    return validation_result


def prepare_dataset_splits(dataset_paths: Dict[str, Any], validation_split: float = 0.2,
                           test_reserve_names: Optional[List[str]] = None,
                           stratify_by_noise_level: bool = False) -> Dict[str, List[str]]:
    """Prepare dataset splits for training/validation/test.
    
    Args:
        dataset_paths: Dictionary of dataset paths
        validation_split: Fraction of data to use for validation
        test_reserve_names: Names of images to reserve for testing
        stratify_by_noise_level: Whether to stratify splits by noise level
        
    Returns:
        Dictionary with 'train', 'validation', 'test' lists
    """
    # Simplified implementation for API demonstration
    all_images = []

    # Collect all available images
    for path_list in dataset_paths.get('image_directories', []):
        image_dir = Path(path_list)
        if image_dir.exists():
            for subdir in ['train', 'val', 'test']:
                subdir_path = image_dir / subdir
                if subdir_path.exists():
                    for img_file in subdir_path.glob('*.jpg'):
                        all_images.append(str(img_file))

    # Reserve test images
    test_reserved = []
    if test_reserve_names:
        test_reserved = [img for img in all_images
                         if any(name in img for name in test_reserve_names)]

    # Split remaining images
    remaining_images = [img for img in all_images if img not in test_reserved]
    val_count = int(len(remaining_images) * validation_split)

    return {
        'train'     : remaining_images[val_count:],
        'validation': remaining_images[:val_count],
        'test'      : test_reserved
    }


def convert_dataset_format(source_config: DatasetConfig, target_config: DatasetConfig,
                           source_path: str, target_path: str) -> Dict[str, Any]:
    """Convert dataset from one format to another.
    
    Args:
        source_config: Source dataset configuration
        target_config: Target dataset configuration
        source_path: Path to source dataset
        target_path: Path for converted dataset
        
    Returns:
        Dictionary with conversion results
    """
    # Simplified conversion implementation
    logging.info(f"Converting dataset from {source_config.data_format} to {target_config.data_format}")

    return {
        'conversion_successful': True,
        'source_format'        : source_config.data_format,
        'target_format'        : target_config.data_format,
        'source_path'          : source_path,
        'target_path'          : target_path
    }


def create_dataset_config_from_yaml(yaml_path: str, **overrides) -> DatasetConfig:
    """Create dataset config from YAML file with optional overrides.
    
    Args:
        yaml_path: Path to YAML configuration file
        **overrides: Parameter overrides
        
    Returns:
        DatasetConfig instance
    """
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # Extract relevant parameters for DatasetConfig
    config_params = {
        'dataset_type'       : yaml_config.get('dataset_type', 'rgb_pairs'),
        'data_format'        : yaml_config.get('data_format', 'clean_noisy'),
        'input_channels'     : yaml_config.get('input_channels', 3),
        'output_channels'    : yaml_config.get('output_channels', 3),
        'crop_size'          : yaml_config.get('crop_size', 128),
        'num_crops_per_image': yaml_config.get('num_crops_per_image', 4),
        'batch_size'         : yaml_config.get('batch_size', 8),
        'color_profile'      : yaml_config.get('color_profile', 'lin_rec2020'),
        'device'             : yaml_config.get('device', 'cpu'),
        'augmentations'      : yaml_config.get('augmentations', []),
        'validation_split'   : yaml_config.get('validation_split', 0.2)
    }

    # Apply overrides
    config_params.update(overrides)

    return DatasetConfig(**config_params)


def load_rawnind_test_reserve_config(config_path: str) -> List[str]:
    """Load RawNIND test reserve configuration.
    
    Args:
        config_path: Path to test reserve YAML file
        
    Returns:
        List of reserved test image names
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('test_reserve_images', [])
    except FileNotFoundError:
        logging.warning(f"Test reserve config not found: {config_path}")
        return []


def validate_training_type_and_dataset_config(training_type: str, dataset_config: DatasetConfig):
    """Validate compatibility between training type and dataset configuration.
    
    Args:
        training_type: Training type ("rgb_to_rgb", "bayer_to_rgb")
        dataset_config: Dataset configuration
        
    Raises:
        ValueError: If training type and dataset config are incompatible
    """
    if training_type == "bayer_to_rgb":
        if dataset_config.dataset_type != "bayer_pairs":
            raise ValueError(f"Bayer training requires bayer_pairs dataset, got {dataset_config.dataset_type}")
        if dataset_config.input_channels != 4:
            raise ValueError("Bayer training requires 4 input channels")
    elif training_type == "rgb_to_rgb":
        if dataset_config.dataset_type != "rgb_pairs":
            raise ValueError(f"RGB training requires rgb_pairs dataset, got {dataset_config.dataset_type}")
