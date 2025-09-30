import pytest

pytestmark = pytest.mark.dataset

import random
from types import SimpleNamespace

import pytest
import torch

from src.rawnind.dataset.clean_api import CleanDataset, CleanTestDataset, ConfigurableDataset, DatasetConfig


@pytest.fixture(autouse=True)
def deterministic_random():
    state = random.getstate()
    random.seed(0)
    yield
    random.setstate(state)


import src.rawnind.dataset.clean_api as clean_api

def _make_fake_tensor(path: str) -> torch.Tensor:
    if "mask" in path:
        return torch.ones(1, 16, 16, dtype=torch.bool)
    if "bayer" in path:
        return torch.ones(4, 16, 16, dtype=torch.float32)
    return torch.ones(3, 16, 16, dtype=torch.float32)


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
            "overexposure_lb": 2.0,
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
    """CleanDataset._standardize_batch_format should normalize legacy dict keys."""
    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
    )
    batch = {
        "x_crops": torch.ones(1, 3, 2, 2),
        "y_crops": torch.full((1, 3, 2, 2), 2.0),
        "mask_crops": torch.ones(1, 3, 2, 2, dtype=torch.bool),
        "gain": torch.tensor([1.0]),
    }

    dataset = CleanDataset(config, data_paths={}, data_loader_override=iter([batch]))

    standardized = dataset._standardize_batch_format(batch)

    assert set(standardized.keys()).issuperset(
        {"clean_images", "noisy_images", "masks", "gain", "image_paths"}
    )
    assert "x_crops" not in standardized
    assert "y_crops" not in standardized
    assert "mask_crops" not in standardized
    assert torch.equal(standardized["clean_images"], batch["x_crops"])
    assert torch.equal(standardized["noisy_images"], batch["y_crops"])
    assert torch.equal(standardized["masks"], batch["mask_crops"])


def test_clean_dataset_initializes_configurable_dataset(monkeypatch):
    """CleanDataset should instantiate ConfigurableDataset when no override is supplied."""
    created: Dict[str, Any] = {}

    class DummyDataset:
        def __init__(self, config_arg, data_paths_arg):
            created["config"] = config_arg
            created["data_paths"] = data_paths_arg

        def __len__(self):
            return 1

    monkeypatch.setattr(clean_api, "ConfigurableDataset", DummyDataset)

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
        test_reserve_images=[],
    )
    data_paths = {"noise_dataset_yamlfpaths": ["train.yaml"]}

    dataset = CleanDataset(config, data_paths)

    assert isinstance(dataset._underlying_dataset, DummyDataset)
    assert created["config"] is config
    assert created["data_paths"] is data_paths


def test_clean_dataset_respects_override(monkeypatch):
    """Data loader override should bypass ConfigurableDataset creation."""

    def _fail(*_args, **_kwargs):
        raise AssertionError("ConfigurableDataset should not be constructed when override is provided")

    monkeypatch.setattr(clean_api, "ConfigurableDataset", _fail)

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_noisy",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
        test_reserve_images=[],
    )
    override_batches = [{"dummy": torch.tensor(1.0)}]

    dataset = CleanDataset(config, data_paths={}, data_loader_override=override_batches)

    assert dataset._underlying_dataset is override_batches


def test_clean_dataset_standardizes_missing_noisy_images(monkeypatch):
    """Standardization should duplicate clean_images when noisy data is absent."""

    class DummyDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

    monkeypatch.setattr(clean_api, "ConfigurableDataset", DummyDataset)

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_clean",
        input_channels=3,
        output_channels=3,
        crop_size=2,
        num_crops_per_image=1,
        batch_size=1,
        test_reserve_images=[],
    )

    dataset = CleanDataset(config, data_paths={"noise_dataset_yamlfpaths": ["train.yaml"]})

    clean_tensor = torch.randn(1, 3, 2, 2)
    mask_tensor = torch.ones(1, 3, 2, 2, dtype=torch.bool)

    batch = {
        "x_crops": clean_tensor,
        "mask_crops": mask_tensor,
    }

    standardized = dataset._standardize_batch_format(batch)

    assert torch.equal(standardized["clean_images"], clean_tensor)
    assert torch.equal(standardized["noisy_images"], clean_tensor)
    assert standardized["noisy_images"] is standardized["clean_images"]


def test_clean_test_dataset_single_image(monkeypatch):
    """CleanTestDataset should iterate single-image batches from ConfigurableDataset."""

    class DummyDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                "x_crops": torch.randn(1, 3, 4, 4),
                "mask_crops": torch.ones(1, 3, 4, 4, dtype=torch.bool),
                "gain": torch.tensor([1.0]),
            }

    monkeypatch.setattr(clean_api, "ConfigurableDataset", DummyDataset)

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_clean",
        input_channels=3,
        output_channels=3,
        crop_size=4,
        num_crops_per_image=1,
        batch_size=1,
        test_reserve_images=[],
    )

    dataset = CleanTestDataset(config, data_paths={"clean_dataset_yamlfpaths": ["single.yaml"]})

    batch = next(iter(dataset))

    assert batch["clean_images"].shape == (1, 3, 4, 4)
    assert batch["noisy_images"].shape == (1, 3, 4, 4)
    assert torch.equal(batch["clean_images"], batch["noisy_images"])


def test_clean_test_dataset_len(monkeypatch):
    """Length of CleanTestDataset should respect underlying dataset size."""

    class DummyDataset:
        def __init__(self, *_args, **_kwargs):
            pass

        def __len__(self):
            return 1

        def __iter__(self):
            yield {
                "x_crops": torch.randn(1, 3, 4, 4),
                "mask_crops": torch.ones(1, 3, 4, 4, dtype=torch.bool),
                "gain": torch.tensor([1.0]),
            }

    monkeypatch.setattr(clean_api, "ConfigurableDataset", DummyDataset)

    config = DatasetConfig(
        dataset_type="rgb_pairs",
        data_format="clean_clean",
        input_channels=3,
        output_channels=3,
        crop_size=4,
        num_crops_per_image=1,
        batch_size=1,
        test_reserve_images=[],
    )

    dataset = CleanTestDataset(config, data_paths={"clean_dataset_yamlfpaths": ["single.yaml"]})

    assert len(dataset) == 1


@pytest.mark.skip(reason="TODO: enforce batch-friendly conversion semantics")
def test_clean_dataset_rejects_scalar_input(monkeypatch):
    """Placeholder for enforcing single-image inputs to be batched before inference."""
    pass
