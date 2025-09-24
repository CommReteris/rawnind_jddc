import pytest

pytestmark = pytest.mark.dataset
import torch
from src.rawnind.dataset.clean_api import ConfigurableDataset, DatasetConfig


def test_configurable_dataset_initialization():
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=128,
        num_crops_per_image=1,
        batch_size=1
    )
    dataset = ConfigurableDataset(config, {})
    assert isinstance(dataset, torch.utils.data.Dataset)
