import os
import pytest
from pathlib import Path
from unittest.mock import patch
from rawnind.dependencies import config_manager
from rawnind.dependencies import json_saver

def test_load_config(tmp_path: Path):
    """
    Test load_config method for loading YAML configuration files.

    Objective: Verify that configurations can be loaded from YAML files.
    Test criteria: Returns correct dictionary structure from saved YAML.
    How testing fulfills purpose: Ensures config loading works for basic use cases.
    Components mocked: None - uses tmp_path fixture for file operations.
    Reason for hermeticity: File operations are isolated to temporary directory.
    """
    config_path = tmp_path / "config.yaml"
    data = {"a": 1, "b": {"c": 2}}
    json_saver.dict_to_yaml(data, config_path)

    manager = config_manager.ConfigManager()
    config = manager.load_config(config_path)
    assert config["a"] == 1
    assert config["b"]["c"] == 2

def test_load_config_nonexistent_file():
    """
    Test load_config method with nonexistent file.

    Objective: Verify proper error handling for missing config files.
    Test criteria: Raises exception when file doesn't exist.
    How testing fulfills purpose: Ensures robust error handling in config loading.
    Components mocked: None - tests actual error behavior.
    Reason for hermeticity: Error is raised before any external dependencies.
    """
    manager = config_manager.ConfigManager()
    with pytest.raises(FileNotFoundError):
        manager.load_config("nonexistent.yaml")

def test_get_training_config():
    """
    Test get_training_config method for loading training configurations.

    Objective: Verify training config loading with proper path construction.
    Test criteria: Returns config dict with correct structure.
    How testing fulfills purpose: Ensures training config access works.
    Components mocked: load_yaml to return test data.
    Reason for hermeticity: Avoids actual file system dependencies.
    """
    test_config = {"learning_rate": 1e-4, "batch_size": 8}

    with patch('rawnind.dependencies.config_manager.load_yaml') as mock_load:
        mock_load.return_value = test_config

        manager = config_manager.ConfigManager()
        config = manager.get_training_config("denoiser", "prgb2prgb")

        assert config == test_config
        mock_load.assert_called_once_with(os.path.join("config", "train_denoiser_prgb2prgb.yaml"), error_on_404=True)

def test_get_test_config():
    """
    Test get_test_config method for loading test configurations.

    Objective: Verify test config loading with proper path construction.
    Test criteria: Returns config dict with correct structure.
    How testing fulfills purpose: Ensures test config access works.
    Components mocked: load_yaml to return test data.
    Reason for hermeticity: Avoids actual file system dependencies.
    """
    test_config = {"test_images": ["image1.png"], "metrics": ["psnr"]}

    with patch('rawnind.dependencies.config_manager.load_yaml') as mock_load:
        mock_load.return_value = test_config

        manager = config_manager.ConfigManager()
        config = manager.get_test_config("minimal_dataset")

        assert config == test_config
        mock_load.assert_called_once_with(os.path.join("config", "test_minimal_dataset.yaml"), error_on_404=True)

def test_get_model_config():
    """
    Test get_model_config method for loading model configurations.

    Objective: Verify model config loading with proper path construction.
    Test criteria: Returns config dict with correct structure.
    How testing fulfills purpose: Ensures model config access works.
    Components mocked: load_yaml to return test data.
    Reason for hermeticity: Avoids actual file system dependencies.
    """
    model_config = {"architecture": "unet", "channels": 3}

    with patch('rawnind.dependencies.config_manager.load_yaml') as mock_load:
        mock_load.return_value = model_config

        manager = config_manager.ConfigManager()
        config = manager.get_model_config("denoise", "definitions")

        assert config == model_config
        mock_load.assert_called_once_with(os.path.join("config", "denoise_models_definitions.yaml"), error_on_404=True)

def test_save_config(tmp_path: Path):
    """
    Test save_config method for saving configurations to YAML.

    Objective: Verify configurations can be saved to YAML files.
    Test criteria: File is created with correct content.
    How testing fulfills purpose: Ensures config saving works for persistence.
    Components mocked: None - uses tmp_path fixture for file operations.
    Reason for hermeticity: File operations are isolated to temporary directory.
    """
    config_path = tmp_path / "saved_config.yaml"
    config_data = {"model": "unet", "lr": 1e-3}

    manager = config_manager.ConfigManager()
    manager.save_config(config_data, str(config_path))

    # Verify file was created and content is correct
    loaded_config = json_saver.load_yaml(str(config_path))
    assert loaded_config == config_data

def test_merge_configs(tmp_path: Path):
    """
    Test merge_configs method for merging configuration dictionaries.

    Objective: Verify configurations can be merged with proper override behavior.
    Test criteria: Override values replace base values, nested dicts are merged.
    How testing fulfills purpose: Ensures config merging works for layered configurations.
    Components mocked: None - pure dictionary operations.
    Reason for hermeticity: No external dependencies, pure computation.
    """
    base_config_path = tmp_path / "base.yaml"
    override_config_path = tmp_path / "override.yaml"

    base_data = {"a": 1, "b": {"c": 2}}
    override_data = {"b": {"c": 3, "d": 4}}

    json_saver.dict_to_yaml(base_data, base_config_path)
    json_saver.dict_to_yaml(override_data, override_config_path)

    manager = config_manager.ConfigManager()
    base_config = manager.load_config(base_config_path)
    override_config = manager.load_config(override_config_path)

    merged_config = manager.merge_configs(base_config, override_config)
    assert merged_config["a"] == 1
    assert merged_config["b"]["c"] == 3
    assert merged_config["b"]["d"] == 4

def test_merge_configs_deep_nesting():
    """
    Test merge_configs method with deeply nested dictionaries.

    Objective: Verify deep merging works correctly.
    Test criteria: Nested overrides are applied at all levels.
    How testing fulfills purpose: Ensures complex config merging is robust.
    Components mocked: None - pure dictionary operations.
    Reason for hermeticity: No external dependencies, pure computation.
    """
    base = {"level1": {"level2": {"value": 1, "keep": "base"}}}
    override = {"level1": {"level2": {"value": 2, "add": "override"}}}

    manager = config_manager.ConfigManager()
    merged = manager.merge_configs(base, override)

    assert merged["level1"]["level2"]["value"] == 2
    assert merged["level1"]["level2"]["keep"] == "base"
    assert merged["level1"]["level2"]["add"] == "override"

def test_merge_configs_non_dict_override():
    """
    Test merge_configs method when override contains non-dict values.

    Objective: Verify non-dict values completely replace dict values.
    Test criteria: Non-dict overrides replace entire sections.
    How testing fulfills purpose: Ensures correct behavior for config type changes.
    Components mocked: None - pure dictionary operations.
    Reason for hermeticity: No external dependencies, pure computation.
    """
    base = {"section": {"nested": "value"}}
    override = {"section": "replacement_string"}

    manager = config_manager.ConfigManager()
    merged = manager.merge_configs(base, override)

    assert merged["section"] == "replacement_string"
