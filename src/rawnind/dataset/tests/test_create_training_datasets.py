"""Unit tests for the create_training_datasets bridge function."""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import RandomSampler, SequentialSampler

import src.rawnind.dataset.clean_api as clean_api
from src.rawnind.dataset.clean_api import create_training_datasets


class _DummyDataset(torch.utils.data.Dataset):
    """Simple dataset returning fixed tensor batches for DataLoader tests."""

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        clean = torch.ones(3, 8, 8, dtype=torch.float32)
        noisy = torch.ones(3, 8, 8, dtype=torch.float32)
        masks = torch.ones(3, 8, 8, dtype=torch.bool)
        return {
            "clean_images": clean,
            "noisy_images": noisy,
            "masks": masks,
            "image_paths": [f"sample_{index}.exr"],
        }


def test_create_training_datasets_constructs_expected_dataloaders(monkeypatch):
    """Factory helpers should be invoked and dataloaders configured correctly."""

    captured: Dict[str, List] = {"train": [], "val": [], "test": []}

    def _make_factory(key: str):
        def _factory(config, data_paths, *args, **kwargs):
            captured[key].append((config, data_paths, args, kwargs))
            return _DummyDataset()

        return _factory

    monkeypatch.setattr(clean_api, "create_training_dataset", _make_factory("train"))
    monkeypatch.setattr(clean_api, "create_validation_dataset", _make_factory("val"))
    monkeypatch.setattr(clean_api, "create_test_dataset", _make_factory("test"))

    result = create_training_datasets(
        input_channels=3,
        output_channels=3,
        crop_size=64,
        batch_size=2,
        clean_dataset_yamlfpaths=["clean.yaml"],
        noise_dataset_yamlfpaths=["noise.yaml"],
        test_reserve=["reserved"],
        num_crops_per_image=4,
        match_gain=True,
    )

    # Ensure each factory was invoked exactly once
    assert len(captured["train"]) == 1
    assert len(captured["val"]) == 1
    assert len(captured["test"]) == 1

    train_config, train_paths, _, _ = captured["train"][0]
    assert train_config.dataset_type == "rgb_pairs"
    assert train_config.data_format == "clean_noisy"
    assert train_config.num_crops_per_image == 4
    assert train_config.batch_size == 2
    assert train_config.match_gain is True
    assert train_config.test_reserve_images == ["reserved"]
    assert train_paths == {"noise_dataset_yamlfpaths": ["noise.yaml"]}

    val_config, _, _, _ = captured["val"][0]
    assert val_config.center_crop is True
    assert val_config.num_crops_per_image == 1
    assert val_config.save_individual_results is False

    test_config, _, _, _ = captured["test"][0]
    assert test_config.center_crop is True
    assert test_config.num_crops_per_image == 1
    assert test_config.save_individual_results is True

    train_loader = result["train_dataloader"]
    val_loader = result["validation_dataloader"]
    test_loader = result["test_dataloader"]

    assert isinstance(train_loader.dataset, _DummyDataset)
    assert isinstance(train_loader.sampler, RandomSampler)
    assert isinstance(val_loader.sampler, SequentialSampler)
    assert isinstance(test_loader.sampler, SequentialSampler)
    assert train_loader.batch_size == 2


def test_create_training_datasets_returns_toy_dataset_when_no_yaml():
    """When no YAML descriptors are provided, synthetic toy loaders are returned."""

    result = create_training_datasets(
        input_channels=4,
        output_channels=3,
        crop_size=32,
        batch_size=1,
        clean_dataset_yamlfpaths=[],
        noise_dataset_yamlfpaths=[],
        test_reserve=[],
    )

    train_loader = result["train_dataloader"]
    val_loader = result["validation_dataloader"]
    test_loader = result["test_dataloader"]

    toy_dataset = train_loader.dataset
    assert len(toy_dataset) == 1
    assert val_loader.dataset is toy_dataset
    assert test_loader.dataset is toy_dataset

    batch = next(iter(train_loader))
    assert batch["clean_images"].shape == (1, 3, 32, 32)
    assert batch["noisy_images"].shape == (1, 4, 32, 32)
    assert batch["masks"].dtype == torch.bool
    image_path_entry = batch["image_paths"][0]
    if isinstance(image_path_entry, tuple):
        image_path_entry = image_path_entry[0]
    assert image_path_entry == "synthetic_image.exr"
