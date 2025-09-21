import importlib
import pytest

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize(
    "module_path, symbol_names",
    [
        (
                "rawnind.training.training_loops",
                [
                    "TrainingLoops",
                    "ImageToImageNNTraining",
                    "PRGBImageToImageNNTraining",
                    "BayerImageToImageNNTraining",
                    "DenoiseCompressTraining",
                    "DenoiserTraining",
                ],
        ),
        ("rawnind.training.experiment_manager", ["ExperimentManager"]),
        ("rawnind.dataset.base_dataset", ["RawImageDataset", "RawDatasetOutput"]),
        ("rawnind.inference.inference_engine", ["InferenceEngine"]),
        ("rawnind.dependencies.utilities", ["load_yaml", "dict_to_yaml"]),
    ],
)
def test_modules_and_symbols_present(module_path, symbol_names):
    mod = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(mod, name), f"{module_path} missing symbol {name}"
