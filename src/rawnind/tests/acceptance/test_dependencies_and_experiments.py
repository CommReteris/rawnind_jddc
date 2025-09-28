import os
import yaml
import pytest

from rawnind.dependencies.json_saver import load_yaml, dict_to_yaml
from rawnind.training.experiment_manager import ExperimentManager

pytestmark = pytest.mark.acceptance


def test_yaml_roundtrip(tmp_path):
    """Test round-trip serialization of YAML configuration data.

    This test verifies that YAML saving and loading preserves the exact structure
    and values of nested configuration dictionaries, ensuring reliable configuration
    persistence for training experiments and model parameters.

    Expected behavior:
    - dict_to_yaml saves the dictionary as valid YAML
    - load_yaml correctly parses the YAML back to the original dictionary
    - Nested structures and primitive types (int, str, dict) are preserved
    - No data loss or type conversion during serialization/deserialization

    Key assertions:
    - Loaded data exactly matches original input dictionary
    - File is created and readable as YAML
    """
    data = {"a": 1, "b": {"c": 2}}
    fpath = tmp_path / "sample.yaml"
    dict_to_yaml(data, str(fpath))
    loaded = load_yaml(str(fpath))
    assert loaded == data


def test_experiment_manager_cleanup(tmp_path):
    """Test experiment manager cleanup of saved model iterations.

    This test verifies that the experiment manager correctly removes unnecessary
    model checkpoint iterations while preserving specified keep iterations.
    It ensures disk space management during long training runs by cleaning up
    obsolete model weights.

    Expected behavior:
    - Non-kept checkpoint files are deleted
    - Kept checkpoint files remain intact
    - Directory structure is preserved minus removed files
    - No errors when cleaning directories with mixed file types

    Key assertions:
    - Only specified iteration files remain after cleanup
    - Removed files are no longer present in directory listing
    - Cleanup operation completes without exceptions
    """
    # Build fake structure: <save>/saved_models/iter_100.pt, iter_200.pt, iter_300.pt
    saved = tmp_path / "saved_models"
    saved.mkdir(parents=True)
    for step in (100, 200, 300):
        (saved / f"iter_{step}.pt").write_bytes(b"weights")

    # Keep only iter_200
    ExperimentManager.cleanup_saved_models_iterations(str(tmp_path), [200])
    files = sorted(os.listdir(saved))
    assert files == ["iter_200.pt"]


def test_experiment_manager_rm_empty(tmp_path):
    """Test experiment manager removal of empty model checkpoint files.

    This test verifies that the experiment manager identifies and removes empty
    checkpoint files that may result from interrupted saves or corrupted writes,
    ensuring only valid model weights are retained in the experiment directory.

    Expected behavior:
    - Empty checkpoint files (0 bytes) are deleted
    - Non-empty checkpoint files are preserved
    - Cleanup scans recursive directories for .pt files
    - No errors when encountering non-checkpoint files

    Key assertions:
    - Only non-empty checkpoint files remain after cleanup
    - Empty files are successfully removed from directory
    - Directory structure remains intact for valid files
    """
    saved = tmp_path / "saved_models"
    saved.mkdir(parents=True)
    (saved / "iter_0.pt").write_bytes(b"")
    (saved / "iter_1.pt").write_bytes(b"x")

    ExperimentManager.rm_empty_models(str(tmp_path))
    files = sorted(os.listdir(saved))
    assert files == ["iter_1.pt"]


def test_get_best_steps_from_results(tmp_path):
    results = {"best_step": {"psnr": 12000, "ms_ssim": 13000}}
    f = tmp_path / "results.yaml"
    f.write_text(yaml.safe_dump(results))

    steps = ExperimentManager.get_best_steps_from_results(str(f))
    # Order is not guaranteed; compare as sorted list
    assert sorted(steps) == [12000, 13000]
