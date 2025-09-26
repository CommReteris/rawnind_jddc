import os
import yaml
import pytest

<<<<<<< HEAD
from rawnind.dependencies.utilities import load_yaml, dict_to_yaml
=======
from rawnind.dependencies.json_saver import load_yaml, dict_to_yaml
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
from rawnind.training.experiment_manager import ExperimentManager

pytestmark = pytest.mark.acceptance


def test_yaml_roundtrip(tmp_path):
    data = {"a": 1, "b": {"c": 2}}
    fpath = tmp_path / "sample.yaml"
    dict_to_yaml(data, str(fpath))
    loaded = load_yaml(str(fpath))
    assert loaded == data


def test_experiment_manager_cleanup(tmp_path):
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
