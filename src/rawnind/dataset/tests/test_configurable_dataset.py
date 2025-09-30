import pytest

pytestmark = pytest.mark.dataset

import random
from types import SimpleNamespace

import pytest
import torch

from src.rawnind.dataset.clean_api import CleanDataset, ConfigurableDataset, DatasetConfig


@pytest.fixture(autouse=True)
def deterministic_random():
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


def _make_fake_tensor(path: str) -> torch.Tensor:
    if "mask" in path:
        return torch.ones(1, 4, 4, dtype=torch.bool)
    if "bayer" in path:
        return torch.ones(4, 4, 4, dtype=torch.float32)
    return torch.ones(3, 4, 4, dtype=torch.float32)


def _make_rgb_xyz():
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def test_configurable_dataset_clean_noisy_bayer(monkeypatch, tmp_path):
    """ConfigurableDataset should load and return correct structures for clean-noisy Bayer data."""
    fake_yaml_path = tmp_path / "bayer.yaml"

    contents = [
        {
            "image_set": "train",
            "is_bayer": True,
            "rgb_msssim_score": 0.95,
            "best_alignment_loss": 0.01,
            "mask_mean": 0.8,
            "best_alignment": [0, 0],
            "mask_fpath": "mask_tensor",
            "rgb_xyz_matrix": _make_rgb_xyz(),
            "raw_gain": 2.0,
            "crops": [
                {
                    "coordinates": [0, 0],
                    "gt_linrec2020_fpath": "gt_tensor",
                    "f_bayer_fpath": "noisy_bayer_tensor",
                    "f_linrec2020_fpath": "noisy_rgb_tensor",
                    "gt_bayer_fpath": "gt_bayer_tensor",
                }
            ],
        }
    ]

    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.load_yaml",
        lambda path, error_on_404=True: contents if str(path) == str(fake_yaml_path) else [],
    )
    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.pt_helpers.fpath_to_tensor",
        lambda path: _make_fake_tensor(path),
    )
    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.rawproc.shift_images",
        lambda gt, noisy, alignment: (gt, noisy),
    )
    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.rawproc.shape_is_compatible",
        lambda *_: True,
    )

    config = DatasetConfig(
        dataset_type="bayer_pairs",
        data_format="clean_noisy",
        input_channels=4,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
        config=SimpleNamespace(bayer_only=True, data_pairing="x_y"),
        match_gain=True,
    )

    dataset = ConfigurableDataset(
        config,
        {"noise_dataset_yamlfpaths": [str(fake_yaml_path)]},
    )

    sample = dataset[0]
    assert set(sample.keys()) == {
        "x_crops",
        "y_crops",
        "mask_crops",
        "rgb_xyz_matrix",
        "gain",
    }
    assert sample["x_crops"].shape == (1, 3, 2, 2)
    assert sample["y_crops"].shape == (1, 4, 1, 1)
    assert sample["mask_crops"].dtype == torch.bool
    assert torch.allclose(sample["rgb_xyz_matrix"], torch.eye(3))
    assert sample["gain"] == pytest.approx(1.0)


def test_configurable_dataset_clean_clean_rgb(monkeypatch, tmp_path):
    """Clean-clean RGB branch should emit only x_crops, mask_crops, and gain."""
    fake_yaml_path = tmp_path / "rgb.yaml"

    contents = [
        {
            "image_set": "train",
            "rgb_msssim_score": 0.92,
            "best_alignment_loss": 0.0,
            "mask_mean": 1.0,
            "best_alignment": [0, 0],
            "rgb_xyz_matrix": _make_rgb_xyz(),
            "crops": [
                {
                    "coordinates": [0, 0],
                    "gt_linrec2020_fpath": "gt_tensor",
                    "gt_bayer_fpath": "gt_bayer_tensor",
                }
            ],
        }
    ]

    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.load_yaml",
        lambda path, error_on_404=True: contents if str(path) == str(fake_yaml_path) else [],
    )
    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.pt_helpers.fpath_to_tensor",
        lambda path: _make_fake_tensor(path),
    )
    monkeypatch.setattr(
        "src.rawnind.dataset.clean_api.rawproc.shape_is_compatible",
        lambda *_: True,
    )

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_clean",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
        config=SimpleNamespace(bayer_only=False),
        match_gain=False,
    )

    dataset = ConfigurableDataset(
        config,
        {"noise_dataset_yamlfpaths": [str(fake_yaml_path)]},
    )

    sample = dataset[0]
    assert set(sample.keys()) == {"x_crops", "mask_crops", "gain"}
    assert sample["x_crops"].shape == (1, 3, 2, 2)
    assert sample["mask_crops"].dtype == torch.bool
    assert sample["gain"] == pytest.approx(1.0)


def test_clean_dataset_standardizes_dict_batches():
    """CleanDataset._standardize_batch_format should normalize legacy keys."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
    )
    dataset = CleanDataset(config, data_paths={}, data_loader_override=[])

    batch = {
        "x_crops": torch.ones(1, 3, 2, 2),
        "y_crops": torch.ones(1, 3, 2, 2),
        "mask_crops": torch.ones(1, 3, 2, 2, dtype=torch.bool),
    }

    standardized = dataset._standardize_batch_format(batch)

    assert "x_crops" in standardized
    assert torch.equal(standardized["noisy_images"], batch["x_crops"])
    assert torch.equal(standardized["clean_images"], batch["y_crops"])
    assert torch.equal(standardized["masks"], batch["mask_crops"])
    assert "image_paths" in standardized
