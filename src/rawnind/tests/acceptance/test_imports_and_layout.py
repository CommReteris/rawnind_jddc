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
        ("rawnind.dependencies.json_saver", ["load_yaml", "dict_to_yaml"]),
    ],
)
def test_modules_and_symbols_present(module_path, symbol_names):
    """Test that core modules expose required symbols post-refactoring.

    This parametrized test verifies that key refactored modules make their essential
    classes, functions, and data structures available through standard imports.
    It serves as an acceptance criterion for the package refactoring, ensuring
    that the new structure maintains backward compatibility for critical components
    while removing legacy CLI dependencies.

    Expected behavior:
    - Each specified module imports successfully without errors
    - All required symbols (classes, functions) are attributes of the module
    - No ImportError or AttributeError during symbol access
    - Module structure supports the clean API design

    Key assertions:
    - hasattr(mod, name) is True for every required symbol per module
    - Specific error message if symbol is missing for debugging
    - Covers training, dataset, inference, and dependencies modules
    """
    mod = importlib.import_module(module_path)
    for name in symbol_names:
        assert hasattr(mod, name), f"{module_path} missing symbol {name}"
