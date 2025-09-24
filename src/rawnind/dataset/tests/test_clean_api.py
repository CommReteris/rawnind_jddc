import logging
import pytest

pytestmark = pytest.mark.dataset
import torch
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.rawnind.dataset.clean_api import (
    DatasetConfig,
    CleanDataset,
    CleanValidationDataset,
    CleanTestDataset,
    create_training_dataset,
    create_validation_dataset,
    create_test_dataset,
)
from src.rawnind.dataset.base_dataset import RawDatasetOutput



# Fixtures for common configurations
@pytest.fixture
def base_dataset_config():
    """A minimal valid DatasetConfig."""
    return DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=1
    )

@pytest.fixture
def bayer_dataset_config():
    """A minimal valid Bayer DatasetConfig."""
    return DatasetConfig(
        dataset_type="bayer_pairs",
        data_format="clean_noisy",
        input_channels=4,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=1
    )

@pytest.fixture
def mock_clean_dataset_yaml(tmp_path):
    """Create a mock YAML file for clean dataset paths."""
    yaml_content = {
        'image_paths': [
            str(tmp_path / 'image1.exr'),
            str(tmp_path / 'image2.exr')
        ]
    }
    yaml_path = tmp_path / 'clean_dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files that the RawImageDataset would expect
    (tmp_path / 'image1.exr').touch()
    (tmp_path / 'image2.exr').touch()
    return [str(yaml_path)]

@pytest.fixture
def mock_noisy_dataset_yaml(tmp_path):
    """Create a mock YAML file for noisy dataset paths."""
    yaml_content = {
        'image_paths': [
            str(tmp_path / 'noisy_image1.exr'),
            str(tmp_path / 'noisy_image2.exr')
        ]
    }
    yaml_path = tmp_path / 'noisy_dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files
    (tmp_path / 'noisy_image1.exr').touch()
    (tmp_path / 'noisy_image2.exr').touch()
    return [str(yaml_path)]

@pytest.fixture
def moc_data_paths(mock_clean_dataset_yaml, mock_noisy_dataset_yaml):
    """Mock dictionary of data paths."""
    return {
        'clean_dataset_yamlfpaths': mock_clean_dataset_yaml,
        'noise_dataset_yamlfpaths': mock_noisy_dataset_yaml,
        'rawnind_path': mock_noisy_dataset_yaml # for rawnind_academic test
    }

@pytest.fixture
def mock_batch_tuple():
    """Mock batch in tuple format."""
    return (
        torch.randn(1, 3, 128, 128),  # noisy_images
        torch.randn(1, 3, 128, 128),  # clean_images
        torch.ones(1, 1, 128, 128, dtype=torch.bool) # masks
    )

@pytest.fixture
def mock_batch_dict():
    """Mock batch in dictionary format."""
    return {
        "x_crops": torch.randn(1, 3, 128, 128),
        "y_crops": torch.randn(1, 3, 128, 128),
        "mask": torch.ones(1, 1, 128, 128, dtype=torch.bool)
    }

@pytest.fixture
def mock_data_loader_override(mock_batch_tuple):
    """A mock iterator for data_loader_override."""
    return iter([mock_batch_tuple])

# --- Test Factory Functions ---


def test_create_training_dataset_rgb_noisy(base_dataset_config, moc_data_paths):
    """Test create_training_dataset for RGB clean_noisy."""
    config = base_dataset_config
    dataset = create_training_dataset(config, moc_data_paths)
    assert isinstance(dataset, CleanDataset)


def test_create_training_dataset_rgb_clean(base_dataset_config, moc_data_paths):
    """Test create_training_dataset for RGB clean_clean."""
    config = base_dataset_config
    config.data_format = "clean_clean"
    dataset = create_training_dataset(config, moc_data_paths)
    assert isinstance(dataset, CleanDataset)


def test_create_training_dataset_bayer_noisy(bayer_dataset_config, moc_data_paths):
    """Test create_training_dataset for Bayer clean_noisy."""
    config = bayer_dataset_config
    dataset = create_training_dataset(config, moc_data_paths)
    assert isinstance(dataset, CleanDataset)


def test_create_training_dataset_bayer_clean(bayer_dataset_config, moc_data_paths):
    """Test create_training_dataset for Bayer clean_clean."""
    config = bayer_dataset_config
    config.data_format = "clean_clean"
    dataset = create_training_dataset(config, moc_data_paths)
    assert isinstance(dataset, CleanDataset)

def test_create_training_dataset_unsupported_type(base_dataset_config, moc_data_paths):
    """Test create_training_dataset with unsupported dataset_type."""
    config = base_dataset_config
    config.dataset_type = "unsupported"
    with pytest.raises(ValueError, match="Unsupported dataset type"):
        create_training_dataset(config, moc_data_paths)

def test_create_validation_dataset(base_dataset_config, moc_data_paths):
    """Test create_validation_dataset."""
    dataset = create_validation_dataset(base_dataset_config, moc_data_paths)
    assert isinstance(dataset, CleanValidationDataset)
    assert dataset.config.center_crop is True
    assert dataset.config.augmentations == []
    assert dataset.config.num_crops_per_image == 1

def test_create_test_dataset(base_dataset_config, moc_data_paths):
    """Test create_test_dataset."""
    dataset = create_test_dataset(base_dataset_config, moc_data_paths)
    assert isinstance(dataset, CleanTestDataset)
    assert dataset.config.center_crop is True
    assert dataset.config.augmentations == []
    assert dataset.config.num_crops_per_image == 1
    assert dataset.config.save_individual_results is True

# --- Test CleanDataset iteration and batch standardization ---

def test_cleandataset_iteration_override(mock_data_loader_override, mock_batch_tuple, base_dataset_config, moc_data_paths):
    """Test CleanDataset iteration using an override."""
    dataset = CleanDataset(base_dataset_config, moc_data_paths, mock_data_loader_override)
    for i, batch in enumerate(dataset):
        assert isinstance(batch, dict)
        assert 'noisy_images' in batch
        assert 'clean_images' in batch
        assert 'masks' in batch
        assert batch['noisy_images'].shape == mock_batch_tuple[0].shape
        assert batch['clean_images'].shape == mock_batch_tuple[1].shape
        assert batch['masks'].shape == mock_batch_tuple[2].shape
        assert 'image_paths' in batch
        assert 'color_profile_info' in batch
        assert 'preprocessing_info' in batch
    assert i == 0 # Only one batch in override

def test_cleandataset_standardize_batch_format_tuple(base_dataset_config, mock_batch_tuple):
    """Test _standardize_batch_format with tuple input."""
    dataset = CleanDataset(base_dataset_config, {}, None)
    standardized_batch = dataset._standardize_batch_format(mock_batch_tuple)
    assert 'noisy_images' in standardized_batch
    assert standardized_batch['clean_images'].shape == mock_batch_tuple[1].shape
    assert 'masks' in standardized_batch
    assert 'image_paths' in standardized_batch
    assert 'color_profile_info' in standardized_batch

def test_cleandataset_standardize_batch_format_dict(base_dataset_config, mock_batch_dict):
    """Test _standardize_batch_format with dict input."""
    dataset = CleanDataset(base_dataset_config, {}, None)
    standardized_batch = dataset._standardize_batch_format(mock_batch_dict)
    assert 'x_crops' in standardized_batch # Original keys are kept
    assert 'y_crops' in standardized_batch
    assert 'mask' in standardized_batch
    assert 'image_paths' in standardized_batch # New keys added
    assert 'color_profile_info' in standardized_batch

def test_cleandataset_standardize_batch_format_unsupported(base_dataset_config):
    """Test _standardize_batch_format with unsupported input type."""
    dataset = CleanDataset(base_dataset_config, {}, None)
    with pytest.raises(ValueError, match="Unknown batch type"):
        dataset._standardize_batch_format("not a batch")
    with pytest.raises(ValueError, match="Unexpected batch format"):
        dataset._standardize_batch_format((torch.randn(1,3,64,64),)) # too short tuple

def test_cleandataset_bayer_info_in_standardized_batch(bayer_dataset_config, mock_batch_tuple):
    """Test that bayer_info is added for bayer_pairs dataset type."""
    dataset = CleanDataset(bayer_dataset_config, {}, mock_data_loader_override)
    standardized_batch = dataset._standardize_batch_format(mock_batch_tuple)
    assert 'bayer_info' in standardized_batch
    assert standardized_batch['bayer_info']['pattern'] == bayer_dataset_config.bayer_pattern

# --- Test CleanDataset utility methods ---

def test_cleandataset_get_compatibility_info(base_dataset_config, moc_data_paths):
    """Test get_compatibility_info."""
    dataset = CleanDataset(base_dataset_config, moc_data_paths)
    info = dataset.get_compatibility_info()
    assert info['compatible_with_training'] is True
    assert info['batch_format'] == 'standard'

def test_cleandataset_get_augmentation_info(base_dataset_config, moc_data_paths):
    """Test get_augmentation_info."""
    config = base_dataset_config
    config.augmentations = ['flip', 'rotate']
    dataset = CleanDataset(config, moc_data_paths)
    info = dataset.get_augmentation_info()
    assert info['enabled'] is True
    assert info['available_augmentations'] == ['flip', 'rotate']

def test_cleandataset_get_cache_info(base_dataset_config, moc_data_paths):
    """Test get_cache_info."""
    config = base_dataset_config
    config.enable_caching = True
    config.cache_size = 10
    dataset = CleanDataset(config, moc_data_paths)
    info = dataset.get_cache_info()
    assert info['enabled'] is True
    assert info['max_size'] == 10

def test_cleandataset_get_loader_info(base_dataset_config, moc_data_paths):
    """Test get_loader_info."""
    config = base_dataset_config
    config.num_workers = 4
    config.batch_size = 8
    dataset = CleanDataset(config, moc_data_paths)
    info = dataset.get_loader_info()
    assert info['num_workers'] == 4
    assert info['batch_size'] == 8

# Test compute_statistics and analyze_noise_levels with mocks to avoid full dataset iteration
def test_cleandataset_compute_statistics(base_dataset_config, mock_batch_tuple):
    """Test compute_statistics."""
    config = base_dataset_config
    config.compute_statistics = True
    # Mock the underlying dataset to return a fixed number of batches
    mock_underlying_dataset = MagicMock()
    mock_underlying_dataset.__iter__.return_value = [
        {'clean_images': torch.randn(1, 3, 128, 128) * (i + 1)} for i in range(10)
    ]
    mock_underlying_dataset.__len__.return_value = 100 # Not used for 'if count >= 100' loop break condition
    
    dataset = CleanDataset(config, {"noise_dataset_yamlfpaths": ["dummy.yaml"]})
    stats = dataset.compute_statistics()
    assert 'mean' in stats
    assert 'std' in stats

def test_cleandataset_analyze_noise_levels(base_dataset_config, mock_batch_tuple):
    """Test analyze_noise_levels."""
    config = base_dataset_config
    config.analyze_noise_levels = True
    mock_underlying_dataset = MagicMock()
    mock_underlying_dataset.__iter__.return_value = [
        {'clean_images': mock_batch_tuple[1], 'noise_info': {'estimated_std': 0.1*(i+1)}} for i in range(5)
    ]
    dataset = CleanDataset(config, {"noise_dataset_yamlfpaths": ["dummy.yaml"]})
    stats = dataset.analyze_noise_levels()
    assert 'mean_noise_std' in stats
    assert stats['mean_noise_std'] > 0

def test_cleandataset_get_determinism_info(base_dataset_config, moc_data_paths):
    """Test get_determinism_info."""
    config = base_dataset_config
    config.center_crop = True
    config.augmentations = []
    dataset = CleanDataset(config, moc_data_paths)
    info = dataset.get_determinism_info()
    assert info['is_deterministic'] is True
    assert info['uses_center_crop'] is True
    assert info['has_augmentations'] is False

    config.center_crop = False
    info = dataset.get_determinism_info()
    assert info['is_deterministic'] is False

    config.augmentations = ['flip']
    info = dataset.get_determinism_info()
    assert info['is_deterministic'] is False

# --- Test CleanValidationDataset and CleanTestDataset specifics ---

def test_cleanvalidationdataset_init_forces_deterministic(base_dataset_config, moc_data_paths):
    """Test that CleanValidationDataset forces deterministic settings."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=10, # Will be overridden
        batch_size=4,
        augmentations=['flip'] # Will be overridden
    )
    val_dataset = CleanValidationDataset(config, moc_data_paths)
    assert val_dataset.config.center_crop is True
    assert val_dataset.config.augmentations == []
    assert val_dataset.config.num_crops_per_image == 1

def test_cleantestdataset_init_forces_deterministic(base_dataset_config, moc_data_paths):
    """Test that CleanTestDataset forces deterministic settings."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=10, # Will be overridden
        batch_size=4,
        augmentations=['flip'] # Will be overridden
    )
    test_dataset = CleanTestDataset(config, moc_data_paths)
    assert test_dataset.config.center_crop is True
    assert test_dataset.config.augmentations == []
    assert test_dataset.config.num_crops_per_image == 1
    assert test_dataset.config.save_individual_results is True

def test_cleantestdataset_standardize_batch_format_metadata(base_dataset_config, mock_batch_tuple):
    """Test that CleanTestDataset adds test-specific metadata."""
    dataset = CleanTestDataset(base_dataset_config, {}, None)
    standardized = dataset._standardize_batch_format(mock_batch_tuple)
    assert 'image_metadata' in standardized
    assert 'image_name' in standardized['image_metadata']
    assert 'original_size' in standardized['image_metadata']
