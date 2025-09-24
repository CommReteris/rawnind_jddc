import pytest
import torch
from torch.utils.data import DataLoader
from unittest.mock import MagicMock, patch

from src.rawnind.dataset.clean_api import DatasetConfig, CleanDataset
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
def mock_raw_batch_tuple():
    """Mock a raw batch (tuple) from the underlying dataset."""
    return (
        torch.randn(3, 128, 128),  # x_crops (single image)
        torch.randn(3, 128, 128),  # y_crops (single image)
        torch.ones(1, 128, 128, dtype=torch.bool) # mask_crops (single image, 1 channel)
    )

@pytest.fixture
def mock_underlying_dataset(mock_raw_batch_tuple):
    """A mock underlying dataset that yields a single mock batch."""
    mock = MagicMock(spec=CleanProfiledRGBNoisyProfiledRGBImageCropsDataset)
    mock.__len__.return_value = 10 # Arbitrary length
    # When __getitem__ is called, it should return a single sample
    mock.__getitem__.side_effect = lambda idx: {
        "x_crops": mock_raw_batch_tuple[0],
        "y_crops": mock_raw_batch_tuple[1],
        "mask": mock_raw_batch_tuple[2] # RawDatasetOutput expects 'mask' key
    }
    return mock

@pytest.fixture
def mock_data_paths():
    """Mock dictionary of data paths."""
    return {
        'noise_dataset_yamlfpaths': ['dummy.yaml']
    }

# Mocking the underlying dataset instantiation to control its behavior
@pytest.fixture(autouse=True)
def patch_underlying_dataset_creation(monkeypatch, mock_underlying_dataset):
    """Patch the creation of the underlying dataset in CleanDataset to return our mock."""
    monkeypatch.setattr(
        'src.rawnind.dataset.clean_api.CleanProfiledRGBNoisyProfiledRGBImageCropsDataset',
        MagicMock(return_value=mock_underlying_dataset)
    )


@pytest.mark.parametrize("batch_size, num_workers", [
    (1, 0),
    (2, 0),
    (4, 1), # Test with num_workers > 0
    (1, 2)
])
@patch('src.rawnind.dataset.clean_api.DatasetConfig', autospec=True) # Patch config to avoid validation
def test_dataloader_batching_and_iteration(
    MockDatasetConfig, mock_underlying_dataset, mock_raw_batch_tuple, batch_size, num_workers, mock_data_paths
):
    """
    Test that DataLoader correctly batches and iterates over CleanDataset,
    and that batch items have expected shapes.
    """
    # Create a mock config for CleanDataset, then update batch_size
    mock_config_instance = MockDatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=batch_size, # This will be used by CleanDataset
        num_workers=num_workers # This will be used by DataLoader
    )
    # Patching __post_init__ to prevent validation errors on initial config creation
    mock_config_instance.__post_init__.return_value = None 
    mock_config_instance.lazy_loading = True # Ensure this is mocked to prevent errors

    # Instantiate CleanDataset with the mocked config, ensuring num_crops_per_image is 1
    # because the mock_underlying_dataset yields single crops
    clean_dataset = CleanDataset(mock_config_instance, mock_data_paths)
    
    # Manually ensure CleanDataset len is based on mock_underlying_dataset
    clean_dataset._underlying_dataset = mock_underlying_dataset # Ensure mock is used
    
    # Create DataLoader
    dataloader = DataLoader(
        clean_dataset,
        batch_size=batch_size,
        shuffle=False, # For predictable iteration
        num_workers=num_workers
    )

    # Expected raw shapes from mock_underlying_dataset __getitem__ (single sample)
    expected_x_shape_per_sample = mock_raw_batch_tuple[0].shape # (3, 128, 128)
    expected_y_shape_per_sample = mock_raw_batch_tuple[1].shape # (3, 128, 128)
    expected_mask_shape_per_sample = mock_raw_batch_tuple[2].shape # (1, 128, 128)

    # Iterate and assert batch shapes
    num_batches = 0
    for i, batch in enumerate(dataloader):
        num_batches += 1
        assert isinstance(batch, dict)
        assert 'noisy_images' in batch # standardized by CleanDataset._standardize_batch_format
        assert 'clean_images' in batch
        assert 'masks' in batch
        
        # Check batch shapes
        assert batch['noisy_images'].shape == (batch_size, *expected_x_shape_per_sample)
        assert batch['clean_images'].shape == (batch_size, *expected_y_shape_per_sample)
        # Note: mask_crops is standardized to have 3 channels in CleanDataset._standardize_batch_format
        assert batch['masks'].shape == (batch_size, 3, *expected_mask_shape_per_sample[1:])

        # Check other metadata added by _standardize_batch_format
        assert 'image_paths' in batch
        assert 'color_profile_info' in batch
        assert 'preprocessing_info' in batch
        
        if num_batches * batch_size >= mock_underlying_dataset.__len__.return_value:
            break # Stop if we've processed all mock samples

    # Ensure a reasonable number of batches were yielded based on mock dataset length
    assert num_batches > 0

def test_dataloader_len(base_dataset_config, mock_underlying_dataset, mock_data_paths):
    """Test len(dataloader) returns expected number of batches."""
    config = base_dataset_config
    config.batch_size = 2
    clean_dataset = CleanDataset(config, mock_data_paths)
    clean_dataset._underlying_dataset = mock_underlying_dataset # Ensure mock is used

    dataloader = DataLoader(clean_dataset, batch_size=config.batch_size)
    
    # mock_underlying_dataset.__len__.return_value is 10
    # Expected dataloader length = ceil(10 / 2) = 5
    assert len(dataloader) == (mock_underlying_dataset.__len__.return_value + config.batch_size - 1) // config.batch_size
    assert len(dataloader) == 5

# Test case for num_workers > 0 interaction (basic check for no immediate errors)
@pytest.mark.parametrize("num_workers", [1, 2])
@patch('src.rawnind.dataset.clean_api.DatasetConfig', autospec=True)
def test_dataloader_with_multiple_workers(MockDatasetConfig, mock_underlying_dataset, num_workers, mock_data_paths):
    """Test DataLoader with multiple workers for basic functionality."""
    mock_config_instance = MockDatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=2,
        num_workers=num_workers
    )
    mock_config_instance.__post_init__.return_value = None
    mock_config_instance.lazy_loading = True

    clean_dataset = CleanDataset(mock_config_instance, mock_data_paths)
    clean_dataset._underlying_dataset = mock_underlying_dataset

    dataloader = DataLoader(
        clean_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers # Set num_workers here
    )

    # Simply iterate a few times to ensure no immediate crashes with multiple workers
    for i, batch in enumerate(dataloader):
        assert isinstance(batch, dict)
        assert 'noisy_images' in batch # Just a quick check
        if i >= 2: # Limit iteration for speed
            break

def test_dataloader_error_propagation(base_dataset_config, mock_data_paths):
    """Test that errors from __getitem__ in underlying dataset propagate through DataLoader."""
    config = base_dataset_config
    
    # Mock the underlying dataset's __getitem__ to raise an error
    mock_item_failure_dataset = MagicMock()
    mock_item_failure_dataset.__len__.return_value = 1
    mock_item_failure_dataset.__getitem__.side_effect = RuntimeError("Simulated dataset item error")

    with patch('src.rawnind.dataset.clean_api.CleanProfiledRGBNoisyProfiledRGBImageCropsDataset', return_value=mock_item_failure_dataset):
        clean_dataset = CleanDataset(config, mock_data_paths)
        dataloader = DataLoader(clean_dataset, batch_size=1)

        with pytest.raises(RuntimeError, match="Simulated dataset item error"):
            for _ in dataloader:
                pass # Attempt to iterate, expecting error