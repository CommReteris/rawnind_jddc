# test_dependencies_and_experiments.py Passing Tests Documentation

## test_yaml_roundtrip

This test verifies the YAML serialization and deserialization functionality using the `dict_to_yaml` and `load_yaml` functions from `rawnind.dependencies.json_saver`. It saves a nested dictionary to a YAML file and then loads it back, asserting that the loaded data matches the original data exactly. This ensures that configuration and results can be reliably saved and restored.

## test_experiment_manager_cleanup

This test verifies the `ExperimentManager.cleanup_saved_models_iterations` method. It creates a temporary directory with saved model files for different iterations, then calls the cleanup method to retain only specified iterations. The test confirms that only the desired model files remain after cleanup, ensuring proper experiment management and disk space optimization.

## test_get_best_steps_from_results

This test verifies the `ExperimentManager.get_best_steps_from_results` method. It creates a results YAML file with best step data containing PSNR and MS-SSIM values, then calls the method to extract these values. The test asserts that the extracted steps are returned as a sorted list, confirming that experiment results can be properly analyzed and the best performing steps identified.