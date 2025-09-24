import pytest
import torch
from unittest.mock import patch, MagicMock

from src.rawnind.dataset.clean_api import create_training_dataset, DatasetConfig

# A mock YAML content that satisfies the dataset's expected structure
MOCK_YAML_CONTENT = [
    {
        "is_bayer": True,
        "image_set": "test_set_1",
        "best_alignment_loss": 0.01,
        "mask_mean": 0.9,
        "rgb_msssim_score": 0.95,
        "rgb_xyz_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        "raw_gain": 1.5,
        "overexposure_lb": 0.99,
        "crops": [
            {
                "coordinates": [0, 0],
                "gt_linrec2020_fpath": "dummy/gt.exr",
                "f_bayer_fpath": "dummy/noisy.exr",
                "mask_fpath": "dummy/mask.exr"
            }
        ]
    }
]

@pytest.fixture
def mock_bayer_config():
    """Provides a basic config for a Bayer dataset."""
    return DatasetConfig(
        dataset_type="bayer_pairs",
        data_format="clean_noisy",
        content_fpaths=["dummy.yaml"],
        input_channels=4,
        output_channels=3,
        crop_size=256,
        num_crops_per_image=4,
        batch_size=1
    )

@patch('src.rawnind.dataset.clean_api.load_yaml', return_value=MOCK_YAML_CONTENT)

@patch('src.rawnind.dependencies.pytorch_helpers.fpath_to_tensor')
@patch('src.rawnind.dependencies.pytorch_helpers.fpath_to_tensor')
def test_integration_noisy_bayer_dataset(mock_fpath_to_tensor_dep, mock_fpath_to_tensor_base, mock_load_yaml, mock_bayer_config):
    """
    Integration test adapted from legacy `test_CleanProfiledRGBNoisyBayerImageCropsDataset`.
    This test validates the end-to-end pipeline for the most complex dataset type.
    """
    # Mock file loading to return tensors of the correct, expected shapes
    def mock_loader(fpath):
        if "gt" in fpath:
            return torch.ones(3, 300, 300)  # Clean RGB
        elif "noisy" in fpath:
            return torch.ones(4, 150, 150) # Noisy Bayer
        elif "mask" in fpath:
            return torch.ones(1, 300, 300, dtype=torch.bool)
        return torch.empty(0)
    mock_fpath_to_tensor.side_effect = mock_loader

    # 1. Create the dataset using the new public API
    dataset = create_training_dataset(mock_bayer_config, {})

    # 2. Get a single item to trigger the full processing pipeline
    # This is currently expected to FAIL until the logic is implemented
    output = dataset[0]

    # 3. Assertions (preserved from legacy test)
    assert output["x_crops"].shape == (4, 3, 256, 256)
    assert output["y_crops"].shape == (4, 4, 128, 128)
    assert output["mask_crops"].shape == (4, 3, 256, 256)
    assert output["rgb_xyz_matrix"].shape == (4, 3)
    assert output["gain"] != 1.0
