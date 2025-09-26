import pytest

pytestmark = pytest.mark.dataset
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from src.rawnind.dataset.clean_api import DatasetConfig

def test_dataset_config_valid_minimal():
    """Test a minimal valid DatasetConfig."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=4
    )
    assert config.dataset_type == "rgb_pairs"
    assert config.input_channels == 3
    assert config.crop_size == 128

def test_dataset_config_valid_bayer():
    """Test a valid Bayer DatasetConfig."""
    config = DatasetConfig(
        dataset_type="bayer_pairs",
        data_format="clean_noisy",
        input_channels=4,
        output_channels=3,
        crop_size=256,
        num_crops_per_image=2,
        batch_size=2
    )
    assert config.dataset_type == "bayer_pairs"
    assert config.input_channels == 4
    assert config.crop_size == 256

@pytest.mark.parametrize("crop_size", [0, -1, 7, 9])
def test_dataset_config_invalid_crop_size(crop_size):
    """Test DatasetConfig with invalid crop_size."""
    with pytest.raises(ValueError, match="Crop size must be positive and even"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=crop_size,
            num_crops_per_image=1,
            batch_size=4
        )

@pytest.mark.parametrize("num_crops_per_image", [0, -1])
def test_dataset_config_invalid_num_crops_per_image(num_crops_per_image):
    """Test DatasetConfig with invalid num_crops_per_image."""
    with pytest.raises(ValueError, match="Number of crops per image must be positive"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=num_crops_per_image,
            batch_size=4
        )

@pytest.mark.parametrize("batch_size", [0, -1])
def test_dataset_config_invalid_batch_size(batch_size):
    """Test DatasetConfig with invalid batch_size."""
    with pytest.raises(ValueError, match="Batch size must be positive"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=1,
            batch_size=batch_size
        )

@pytest.mark.parametrize("input_channels", [0, -1])
def test_dataset_config_invalid_input_channels(input_channels):
    """Test DatasetConfig with invalid input_channels."""
    with pytest.raises(ValueError, match="Input channels must be positive"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=input_channels,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=1,
            batch_size=4
        )

@pytest.mark.parametrize("output_channels", [0, -1])
def test_dataset_config_invalid_output_channels(output_channels):
    """Test DatasetConfig with invalid output_channels."""
    with pytest.raises(ValueError, match="Output channels must be positive"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=output_channels,
            crop_size=128,
            num_crops_per_image=1,
            batch_size=4
        )

def test_dataset_config_bayer_rgb_channel_mismatch():
    """Test Bayer dataset type with incorrect input channels."""
    with pytest.raises(ValueError, match="Bayer datasets require 4 input channels"):
        DatasetConfig(
            dataset_type="bayer_pairs",
            data_format="clean_noisy",
            input_channels=3,  # Incorrect for bayer_pairs
            output_channels=3,
            crop_size=128,
            num_crops_per_image=1,
            batch_size=4
        )

def test_dataset_config_rgb_bayer_channel_mismatch():
    """Test RGB dataset type with incorrect input channels."""
    with pytest.raises(ValueError, match="RGB datasets require 3 input channels"):
        DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=4,  # Incorrect for rgb_pairs
            output_channels=3,
            crop_size=128,
            num_crops_per_image=1,
            batch_size=4
        )

def test_dataset_config_quality_thresholds_default():
    """Test if quality_thresholds are set to default if not provided."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=4
    )
    assert config.quality_thresholds == {
        "max_alignment_error": 0.035,
        "max_overexposure_ratio": 0.01,
        "min_image_quality_score": 0.7
    }

def test_dataset_config_quality_thresholds_custom():
    """Test DatasetConfig with custom quality_thresholds."""
    custom_thresholds = {"custom_metric": 0.9}
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=4,
        quality_thresholds=custom_thresholds
    )
    assert config.quality_thresholds == custom_thresholds

def test_dataset_config_other_fields_preserved():
    """Test if other fields are correctly initialized."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=4,
        lazy_loading=False,
        cache_size=50,
        device="cuda"
    )
    assert not config.lazy_loading
    assert config.cache_size == 50
    assert config.device == "cuda"