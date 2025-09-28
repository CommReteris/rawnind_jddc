# test_imports_and_layout.py Passing Tests Documentation

## test_modules_and_symbols_present[rawnind.training.training_loops-symbol_names0]

This parametrized test verifies that the `rawnind.training.training_loops` module can be imported and contains all the expected training-related symbols: `TrainingLoops`, `ImageToImageNNTraining`, `PRGBImageToImageNNTraining`, `BayerImageToImageNNTraining`, `DenoiseCompressTraining`, and `DenoiserTraining`. This ensures that the training loops API is properly exposed and available for use.

## test_modules_and_symbols_present[rawnind.training.experiment_manager-symbol_names1]

This test verifies that the `rawnind.training.experiment_manager` module can be imported and contains the `ExperimentManager` class. This confirms that the experiment management functionality is properly accessible through the public API.

## test_modules_and_symbols_present[rawnind.dataset.base_dataset-symbol_names2]

This test verifies that the `rawnind.dataset.base_dataset` module can be imported and contains the `RawImageDataset` and `RawDatasetOutput` symbols. This ensures that the base dataset classes are available for implementing custom datasets.

## test_modules_and_symbols_present[rawnind.inference.inference_engine-symbol_names3]

This test verifies that the `rawnind.inference.inference_engine` module can be imported and contains the `InferenceEngine` class. This confirms that the inference functionality is properly exposed through the clean API.