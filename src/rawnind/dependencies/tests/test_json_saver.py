"""Tests for JSON and YAML saver utilities.

This module tests the functionality of the json_saver module, including basic
save/load operations and the JSONSaver/YAMLSaver classes for tracking training
metrics and best values.
"""

import json
import yaml
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open
from rawnind.dependencies import json_saver


def test_load_yaml_file_exists(tmp_path: Path):
    """Test loading YAML file when it exists.

    Objective: Verify that load_yaml correctly reads and parses YAML files.
    Test criteria: Function returns parsed dictionary matching expected data.
    How testing for this criteria fulfills purpose: Ensures YAML loading works
      for configuration files and saved results.
    What components are mocked: None - uses real file I/O.
    Reasons for hermeticity: Tests real file operations without mocking to ensure
      end-to-end functionality works.
    """
    data = {"test": "data", "number": 1, "nested": {"key": "value"}}
    file_path = tmp_path / "test.yaml"
    json_saver.dict_to_yaml(data, file_path)

    loaded_data = json_saver.load_yaml(str(file_path))
    assert loaded_data == data


def test_load_yaml_file_not_found_error():
    """Test load_yaml raises error when file not found and error_on_404=True.

    Objective: Verify error handling for missing YAML files.
    Test criteria: FileNotFoundError is raised with appropriate message.
    How testing for this criteria fulfills purpose: Ensures robust error handling
      in configuration loading scenarios.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file system behavior without mocking.
    """
    with pytest.raises(FileNotFoundError, match="not found"):
        json_saver.load_yaml("nonexistent.yaml", error_on_404=True)


def test_load_yaml_file_not_found_no_error():
    """Test load_yaml returns None when file not found and error_on_404=False.

    Objective: Verify graceful handling of missing YAML files.
    Test criteria: Function returns None instead of raising error.
    How testing for this criteria fulfills purpose: Supports optional configuration
      loading scenarios.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file system behavior.
    """
    result = json_saver.load_yaml("nonexistent.yaml", error_on_404=False)
    assert result is None


def test_dict_to_yaml_creates_directory(tmp_path: Path):
    """Test dict_to_yaml creates parent directories if they don't exist.

    Objective: Verify directory creation during YAML saving.
    Test criteria: File is created in newly created subdirectory.
    How testing for this criteria fulfills purpose: Ensures save operations work
      even when output directories don't exist.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file system operations.
    """
    data = {"test": "data"}
    subdir = tmp_path / "subdir"
    file_path = subdir / "test.yaml"

    json_saver.dict_to_yaml(data, str(file_path))

    assert file_path.exists()
    loaded_data = json_saver.load_yaml(str(file_path))
    assert loaded_data == data


def test_dict_to_json_basic(tmp_path: Path):
    """Test basic JSON dictionary saving and loading.

    Objective: Verify dict_to_json saves data correctly.
    Test criteria: Saved file contains expected JSON content.
    How testing for this criteria fulfills purpose: Ensures basic save functionality.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file operations.
    """
    data = {"test": "data", "number": 1}
    file_path = tmp_path / "test.json"

    json_saver.dict_to_json(data, str(file_path))

    with open(file_path, 'r') as f:
        loaded_data = json.load(f)
    assert loaded_data == data


def test_jsonfpath_load_file_exists(tmp_path: Path):
    """Test jsonfpath_load when file exists.

    Objective: Verify JSON loading functionality.
    Test criteria: Function returns correct dictionary data.
    How testing for this criteria fulfills purpose: Ensures JSON result loading works.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file operations.
    """
    data = {"test": "data", "number": 42}
    file_path = tmp_path / "test.json"
    json_saver.dict_to_json(data, str(file_path))

    loaded_data = json_saver.jsonfpath_load(str(file_path))
    assert loaded_data == data


def test_jsonfpath_load_file_not_found():
    """Test jsonfpath_load returns default when file not found.

    Objective: Verify default value handling for missing JSON files.
    Test criteria: Function returns provided default value.
    How testing for this criteria fulfills purpose: Supports optional result loading.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file system behavior.
    """
    default = {"default": "value"}
    result = json_saver.jsonfpath_load("nonexistent.json", default=default)
    assert result == default


def test_jsonfpath_load_file_not_found_no_default():
    """Test jsonfpath_load returns None when file not found and no default.

    Objective: Verify None return for missing files without default.
    Test criteria: Function returns None.
    How testing for this criteria fulfills purpose: Handles optional file loading.
    What components are mocked: None.
    Reasons for hermeticity: Tests real file system behavior.
    """
    result = json_saver.jsonfpath_load("nonexistent.json")
    assert result is None


class TestJSONSaver:
    """Test suite for JSONSaver class functionality."""

    def test_init_basic(self, tmp_path):
        """Test JSONSaver initialization with basic parameters.

        Objective: Verify JSONSaver initializes correctly.
        Test criteria: Object created with expected attributes.
        How testing for this criteria fulfills purpose: Ensures saver setup works.
        What components are mocked: None.
        Reasons for hermeticity: Tests real object creation.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        assert saver.jsonfpath == str(file_path)
        assert saver.best_key_str == "best_step"
        assert saver.warmup_nsteps == 0
        assert "best_val" in saver.results
        assert "best_step" in saver.results

    def test_init_with_epoch_type(self, tmp_path):
        """Test JSONSaver initialization with epoch type.

        Objective: Verify step_type parameter affects best_key_str.
        Test criteria: best_key_str uses epoch instead of step.
        How testing for this criteria fulfills purpose: Tests configuration options.
        What components are mocked: None.
        Reasons for hermeticity: Tests real initialization.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path), step_type="epoch")

        assert saver.best_key_str == "best_epoch"

    def test_init_with_custom_default(self, tmp_path):
        """Test JSONSaver initialization with custom default data.

        Objective: Verify custom default data is used.
        Test criteria: Results contain custom default structure.
        How testing for this criteria fulfills purpose: Tests initialization flexibility.
        What components are mocked: None.
        Reasons for hermeticity: Tests real initialization.
        """
        file_path = tmp_path / "test.json"
        custom_default = {"custom": "data", "best_val": {"metric": 1.0}}
        saver = json_saver.JSONSaver(str(file_path), default=custom_default)

        assert saver.results["custom"] == "data"
        assert saver.results["best_val"]["metric"] == 1.0

    def test_init_directory_path_raises_error(self, tmp_path):
        """Test JSONSaver raises error when path is directory.

        Objective: Verify path validation in initialization.
        Test criteria: AssertionError is raised for directory paths.
        How testing for this criteria fulfills purpose: Ensures proper error handling.
        What components are mocked: None.
        Reasons for hermeticity: Tests real validation logic.
        """
        dir_path = tmp_path / "directory"
        dir_path.mkdir()

        with pytest.raises(AssertionError, match="JSON path must be a file"):
            json_saver.JSONSaver(str(dir_path))

    def test_add_res_basic_minimize(self, tmp_path):
        """Test add_res with basic minimization metric.

        Objective: Verify result addition and best value tracking for minimization.
        Test criteria: Best values and steps are correctly updated.
        How testing for this criteria fulfills purpose: Tests core saver functionality.
        What components are mocked: None.
        Reasons for hermeticity: Tests real metric tracking.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        # Add first result
        saver.add_res(step=100, res={"loss": 0.5}, minimize=True, write=False)
        assert saver.results[100]["loss"] == 0.5
        assert saver.results["best_val"]["loss"] == 0.5
        assert saver.results["best_step"]["loss"] == 100

        # Add better result (lower loss)
        saver.add_res(step=200, res={"loss": 0.3}, minimize=True, write=False)
        assert saver.results["best_val"]["loss"] == 0.3
        assert saver.results["best_step"]["loss"] == 200

        # Add worse result (higher loss)
        saver.add_res(step=300, res={"loss": 0.4}, minimize=True, write=False)
        assert saver.results["best_val"]["loss"] == 0.3  # Should remain best

    def test_add_res_maximize_metric(self, tmp_path):
        """Test add_res with maximization metric.

        Objective: Verify best value tracking for maximization.
        Test criteria: Higher values are correctly identified as better.
        How testing for this criteria fulfills purpose: Tests maximization logic.
        What components are mocked: None.
        Reasons for hermeticity: Tests real metric tracking.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"accuracy": 0.8}, minimize=False, write=False)
        assert saver.results["best_val"]["accuracy"] == 0.8

        saver.add_res(step=200, res={"accuracy": 0.9}, minimize=False, write=False)
        assert saver.results["best_val"]["accuracy"] == 0.9  # Higher is better

    def test_add_res_mixed_minimize_dict(self, tmp_path):
        """Test add_res with mixed minimize configuration.

        Objective: Verify per-metric minimize configuration.
        Test criteria: Different metrics use different minimize settings.
        How testing for this criteria fulfills purpose: Tests flexible configuration.
        What components are mocked: None.
        Reasons for hermeticity: Tests real configuration handling.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        minimize_config = {"loss": True, "accuracy": False}
        saver.add_res(step=100, res={"loss": 0.5, "accuracy": 0.8},
                     minimize=minimize_config, write=False)

        assert saver.results["best_val"]["loss"] == 0.5
        assert saver.results["best_val"]["accuracy"] == 0.8

    def test_add_res_with_warmup(self, tmp_path):
        """Test add_res with warmup period ignores early steps.

        Objective: Verify warmup_nsteps prevents early steps from being best.
        Test criteria: Steps before warmup are not considered for best tracking.
        How testing for this criteria fulfills purpose: Tests training stabilization.
        What components are mocked: None.
        Reasons for hermeticity: Tests real warmup logic.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path), warmup_nsteps=50)

        # Early steps should not affect best values
        saver.add_res(step=10, res={"loss": 10.0}, minimize=True, write=False)
        assert "loss" not in saver.results["best_val"]

        # Steps after warmup should affect best values
        saver.add_res(step=60, res={"loss": 0.5}, minimize=True, write=False)
        assert saver.results["best_val"]["loss"] == 0.5

    def test_add_res_with_key_prefix(self, tmp_path):
        """Test add_res with key prefix for metric names.

        Objective: Verify key_prefix adds prefixes to metric names.
        Test criteria: Metric names include prefix in results.
        How testing for this criteria fulfills purpose: Tests metric organization.
        What components are mocked: None.
        Reasons for hermeticity: Tests real prefix logic.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5}, key_prefix="train_", write=False)
        assert "train_loss" in saver.results[100]
        assert saver.results["best_val"]["train_loss"] == 0.5

    def test_add_res_skips_none_values(self, tmp_path, capsys):
        """Test add_res skips None values with warning.

        Objective: Verify None value handling.
        Test criteria: None values are skipped and warning is printed.
        How testing for this criteria fulfills purpose: Tests error handling.
        What components are mocked: capsys for stdout capture.
        Reasons for hermeticity: capsys allows testing print output without side effects.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5, "invalid": None}, write=False)

        assert "loss" in saver.results[100]
        assert "invalid" not in saver.results[100]
        captured = capsys.readouterr()
        assert "missing value for invalid" in captured.out

    def test_add_res_with_epoch_parameter(self, tmp_path):
        """Test add_res using epoch parameter instead of step.

        Objective: Verify epoch parameter works as step alias.
        Test criteria: Epoch value is used as step number.
        How testing for this criteria fulfills purpose: Tests backward compatibility.
        What components are mocked: None.
        Reasons for hermeticity: Tests real parameter handling.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(epoch=100, res={"loss": 0.5}, write=False)
        assert 100 in saver.results
        assert saver.results[100]["loss"] == 0.5

    def test_add_res_missing_step_epoch_raises_error(self, tmp_path):
        """Test add_res raises error when neither step nor epoch provided.

        Objective: Verify parameter validation.
        Test criteria: ValueError is raised for missing step/epoch.
        How testing for this criteria fulfills purpose: Tests input validation.
        What components are mocked: None.
        Reasons for hermeticity: Tests real validation logic.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        with pytest.raises(ValueError, match="Must specify either step or epoch"):
            saver.add_res(res={"loss": 0.5})

    def test_get_best_step(self, tmp_path):
        """Test get_best_step returns correct step for metric.

        Objective: Verify best step retrieval.
        Test criteria: Returns step with best value for metric.
        How testing for this criteria fulfills purpose: Tests result querying.
        What components are mocked: None.
        Reasons for hermeticity: Tests real best step logic.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5}, write=False)
        saver.add_res(step=200, res={"loss": 0.3}, write=False)

        assert saver.get_best_step("loss") == 200

    def test_get_best_step_missing_metric_raises_error(self, tmp_path):
        """Test get_best_step raises error for unknown metric.

        Objective: Verify error handling for unknown metrics.
        Test criteria: KeyError is raised for non-existent metric.
        How testing for this criteria fulfills purpose: Tests error handling.
        What components are mocked: None.
        Reasons for hermeticity: Tests real error conditions.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        with pytest.raises(KeyError):
            saver.get_best_step("unknown_metric")

    def test_get_best_step_results(self, tmp_path):
        """Test get_best_step_results returns all metrics from best step.

        Objective: Verify complete result retrieval from best step.
        Test criteria: Returns all metrics from step with best value.
        How testing for this criteria fulfills purpose: Tests comprehensive querying.
        What components are mocked: None.
        Reasons for hermeticity: Tests real result retrieval.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5, "accuracy": 0.8}, write=False)
        saver.add_res(step=200, res={"loss": 0.3, "accuracy": 0.9}, write=False)

        best_results = saver.get_best_step_results("loss")
        assert best_results["loss"] == 0.3
        assert best_results["accuracy"] == 0.9

    def test_get_best_steps(self, tmp_path):
        """Test get_best_steps returns set of all best steps.

        Objective: Verify collection of all best-performing steps.
        Test criteria: Returns set containing steps that are best for any metric.
        How testing for this criteria fulfills purpose: Tests step collection logic.
        What components are mocked: None.
        Reasons for hermeticity: Tests real step aggregation.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))
        minimize_config = {"loss": True, "accuracy": False}

        saver.add_res(res={"loss": 0.5, "accuracy": 0.8}, step=100, minimize=minimize_config, write=False)
        saver.add_res(res={"loss": 0.3, "accuracy": 0.7}, step=200, minimize=minimize_config, write=False)
        saver.add_res(res={"loss": 0.4, "accuracy": 0.9}, step=300, minimize=minimize_config, write=False)


        best_steps = saver.get_best_steps()
        assert 200 in best_steps  # Best for loss
        assert 300 in best_steps  # Best for accuracy
        assert 100 not in best_steps

    def test_is_empty(self, tmp_path):
        """Test is_empty returns True when no results saved.

        Objective: Verify empty state detection.
        Test criteria: Returns True for new saver with no results.
        How testing for this criteria fulfills purpose: Tests state checking.
        What components are mocked: None.
        Reasons for hermeticity: Tests real empty state logic.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        assert saver.is_empty() is True

        saver.add_res(step=100, res={"loss": 0.5}, write=False)
        assert saver.is_empty() is False

    def test_write_saves_to_file(self, tmp_path):
        """Test write saves results to JSON file.

        Objective: Verify file persistence of results.
        Test criteria: File contains expected JSON structure after write.
        How testing for this criteria fulfills purpose: Tests data persistence.
        What components are mocked: None.
        Reasons for hermeticity: Tests real file writing.
        """
        file_path = tmp_path / "test.json"
        saver = json_saver.JSONSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5}, write=True)

        with open(file_path, 'r') as f:
            saved_data = json.load(f)

        assert saved_data["100"]["loss"] == 0.5
        assert "best_val" in saved_data
        assert "best_step" in saved_data


class TestYAMLSaver:
    """Test suite for YAMLSaver class functionality."""

    def test_init_inherits_from_json_saver(self, tmp_path):
        """Test YAMLSaver inherits JSONSaver functionality.

        Objective: Verify inheritance and basic initialization.
        Test criteria: YAMLSaver has JSONSaver attributes and methods.
        How testing for this criteria fulfills purpose: Tests class hierarchy.
        What components are mocked: None.
        Reasons for hermeticity: Tests real inheritance.
        """
        file_path = tmp_path / "test.yaml"
        saver = json_saver.YAMLSaver(str(file_path))

        assert saver.jsonfpath == str(file_path)
        assert saver.best_key_str == "best_step"
        assert hasattr(saver, 'add_res')
        assert hasattr(saver, 'get_best_step')

    def test_write_saves_to_yaml_file(self, tmp_path):
        """Test write saves results to YAML file.

        Objective: Verify YAML file persistence.
        Test criteria: File contains expected YAML structure after write.
        How testing for this criteria fulfills purpose: Tests YAML serialization.
        What components are mocked: None.
        Reasons for hermeticity: Tests real YAML writing.
        """
        file_path = tmp_path / "test.yaml"
        saver = json_saver.YAMLSaver(str(file_path))

        saver.add_res(step=100, res={"loss": 0.5, "accuracy": 0.8}, write=False)
        saver.write()

        loaded_data = json_saver.load_yaml(str(file_path))
        assert loaded_data[100]["loss"] == 0.5
        assert loaded_data[100]["accuracy"] == 0.8

    def test_load_existing_yaml_file(self, tmp_path):
        """Test loading existing YAML file with results.

        Objective: Verify YAML file loading on initialization.
        Test criteria: Existing data is loaded correctly.
        How testing for this criteria fulfills purpose: Tests YAML deserialization.
        What components are mocked: None.
        Reasons for hermeticity: Tests real YAML loading.
        """
        file_path = tmp_path / "existing.yaml"
        initial_data = {
            50: {"loss": 0.8},
            "best_val": {"loss": 0.8},
            "best_step": {"loss": 50}
        }
        json_saver.dict_to_yaml(initial_data, str(file_path))

        saver = json_saver.YAMLSaver(str(file_path))
        assert saver.results[50]["loss"] == 0.8
        assert saver.get_best_step("loss") == 50

    def test_load_nonexistent_yaml_file_uses_default(self, tmp_path):
        """Test loading nonexistent YAML file uses default data.

        Objective: Verify default handling for missing YAML files.
        Test criteria: Default data structure is used when file doesn't exist.
        How testing for this criteria fulfills purpose: Tests graceful initialization.
        What components are mocked: None.
        Reasons for hermeticity: Tests real file handling.
        """
        file_path = tmp_path / "nonexistent.yaml"
        saver = json_saver.YAMLSaver(str(file_path))

        assert "best_val" in saver.results
        assert "best_step" in saver.results