# Passing Tests Documentation

This document provides documentation for all tests that are currently passing when running `python -m pytest src/rawnind`. The tests were collected and verified on 2025-09-28. Only passing tests (287 total) are included here. Tests are grouped by their source module/file for clarity. Each test name is listed with a brief description inferred from its name and typical testing conventions in the RawNIND project context (e.g., dataset handling, configuration validation, PyTorch integrations, etc.). No failing, skipped, or errored tests are documented.

Descriptions focus on the core purpose: validating configurations, shapes, integrations, error handling, and domain-specific behaviors like Bayer processing, MS-SSIM constraints, and PyTorch operations.

## Dataset Module Tests (src/rawnind/dataset/tests/)

### test_configurable_dataset.py
- [`test_configurable_dataset_initialization()`](src/rawnind/dataset/tests/test_configurable_dataset.py): Validates the basic initialization of the ConfigurableDataset class, ensuring parameters like crop size, batch size, and channel configurations are accepted without errors.

### test_dataset_config.py
- [`test_dataset_config_valid_minimal()`](src/rawnind/dataset/tests/test_dataset_config.py): Tests creation of a minimal valid DatasetConfig with default values, confirming it passes validation.
- [`test_dataset_config_valid_bayer()`](src/rawnind/dataset/tests/test_dataset_config.py): Verifies DatasetConfig for Bayer (4-channel) input/output setups, ensuring channel counts and crop sizes align with domain requirements.
- [`test_dataset_config_invalid_crop_size[0]`](src/rawnind/dataset/tests/test_dataset_config.py): Checks validation raises errors for crop_size=0 (invalid non-positive value).
- [`test_dataset_config_invalid_crop_size[-1]`](src/rawnind/dataset/tests/test_dataset_config.py): Validates error for negative crop_size values.
- [`test_dataset_config_invalid_crop_size[7]`](src/rawnind/dataset/tests/test_dataset_config.py): Ensures crop_size=7 fails due to MS-SSIM size constraints (must allow 4 downsamplings, minimum ~160).
- [`test_dataset_config_invalid_crop_size[9]`](src/rawnind/dataset/tests/test_dataset_config.py): Confirms small crop sizes like 9 trigger validation errors for MS-SSIM compatibility.
- [`test_dataset_config_invalid_num_crops_per_image[0]`](src/rawnind/dataset/tests/test_dataset_config.py): Tests error for num_crops_per_image=0.
- [`test_dataset_config_invalid_num_crops_per_image[-1]`](src/rawnind/dataset/tests/test_dataset_config.py): Validates negative num_crops_per_image raises ValueError.
- [`test_dataset_config_invalid_batch_size[0]`](src/rawnind/dataset/tests/test_dataset_config.py): Ensures batch_size=0 is rejected.
- [`test_dataset_config_invalid_batch_size[-1]`](src/rawnind/dataset/tests/test_dataset_config.py): Confirms negative batch_size fails validation.
- [`test_dataset_config_invalid_input_channels[0]`](src/rawnind/dataset/tests/test_dataset_config.py): Tests input_channels=0 raises error.
- [`test_dataset_config_invalid_input_channels[-1]`](src/rawnind/dataset/tests/test_dataset_config.py): Validates negative input_channels is invalid.
- [`test_dataset_config_invalid_output_channels[0]`](src/rawnind/dataset/tests/test_dataset_config.py): Checks output_channels=0 error.
- [`test_dataset_config_invalid_output_channels[-1]`](src/rawnind/dataset/tests/test_dataset_config.py): Ensures negative output_channels fails.
- [`test_dataset_config_bayer_rgb_channel_mismatch()`](src/rawnind/dataset/tests/test_dataset_config.py): Verifies error when Bayer input (4 channels) mismatches RGB output (3 channels) without proper demosaicing config.
- [`test_dataset_config_rgb_bayer_channel_mismatch()`](src/rawnind/dataset/tests/test_dataset_config.py): Confirms mismatch error for RGB input to Bayer output.
- [`test_dataset_config_other_fields_preserved()`](src/rawnind/dataset/tests/test_dataset_config.py): Tests that non-validation fields (e.g., custom metrics) are preserved in config objects.

### test_integration.py
- [`test_create_training_dataset_shapes[clean-bayer-BayerDatasetConfig-mock_content_fpath_rawnind-extra_params0-expected_x_shape0-expected_y_shape0]`](src/rawnind/dataset/tests/test_integration.py): Validates tensor shapes for clean Bayer training dataset creation using mock paths.
- [`test_create_training_dataset_shapes[clean-rgb-RgbDatasetConfig-mock_content_fpath_extraraw-extra_params1-expected_x_shape1-expected_y_shape1]`](src/rawnind/dataset/tests/test_integration.py): Checks shapes for clean RGB dataset with extraraw mock data.
- [`test_create_training_dataset_shapes[noisy-bayer-BayerDatasetConfig-mock_content_fpath_rawnind-extra_params2-expected_x_shape2-expected_y_shape2]`](src/rawnind/dataset/tests/test_integration.py): Verifies noisy Bayer dataset shapes.
- [`test_create_training_dataset_shapes[noisy-rgb-RgbDatasetConfig-mock_content_fpath_extraraw-extra_params3-expected_x_shape3-expected_y_shape3]`](src/rawnind/dataset/tests/test_integration.py): Confirms noisy RGB dataset shapes.
- [`test_rawimagedataset_random_crops[128-4-True-expected_x_shape0-expected_y_shape0-expected_mask_shape0]`](src/rawnind/dataset/tests/test_integration.py): Tests random cropping (crop=128, num_crops=4, mask=True) yields expected shapes and masks.
- [`test_rawimagedataset_random_crops[128-4-False-expected_x_shape1-None-expected_mask_shape1]`](src/rawnind/dataset/tests/test_integration.py): Validates random crops without masking.
- [`test_rawimagedataset_random_crops[256-2-True-expected_x_shape2-expected_y_shape2-expected_mask_shape2]`](src/rawnind/dataset/tests/test_integration.py): Checks larger crop (256) with fewer crops (2) and masking.
- [`test_rawimagedataset_center_crop[128-True-True-expected_x_shape0-expected_y_shape0-expected_mask_shape0]`](src/rawnind/dataset/tests/test_integration.py): Tests center cropping with gain matching and masking.
- [`test_rawimagedataset_center_crop[128-True-False-expected_x_shape1-expected_y_shape1-expected_mask_shape1]`](src/rawnind/dataset/tests/test_integration.py): Center crop with gain but no mask.
- [`test_rawimagedataset_center_crop[256-False-True-expected_x_shape2-None-expected_mask_shape2]`](src/rawnind/dataset/tests/test_integration.py): Larger center crop without gain matching.
- [`test_create_training_dataset_gain[noisy-bayer-BayerDatasetConfig-mock_content_fpath_rawnind-extra_params0-1.5]`](src/rawnind/dataset/tests/test_integration.py): Verifies gain application (1.5) in noisy Bayer dataset.
- [`test_create_training_dataset_gain[noisy-rgb-RgbDatasetConfig-mock_content_fpath_extraraw-extra_params1-1.0]`](src/rawnind/dataset/tests/test_integration.py): Tests default gain (1.0) for noisy RGB.
- [`test_create_training_dataset_gain[clean-bayer-BayerDatasetConfig-mock_content_fpath_rawnind-extra_params2-1.0]`](src/rawnind/dataset/tests/test_integration.py): Confirms gain=1.0 for clean Bayer.
- [`test_create_training_dataset_msssim_filter[0.0-1.0-2]`](src/rawnind/dataset/tests/test_integration.py): Tests MS-SSIM filtering with min=0.0, max=1.0, expecting 2 images.
- [`test_create_training_dataset_msssim_filter[0.9-1.0-1]`](src/rawnind/dataset/tests/test_integration.py): Validates filtering for high-quality images (min=0.9), expecting 1 image.
- [`test_create_training_dataset_no_images()`](src/rawnind/dataset/tests/test_integration.py): Ensures graceful handling when no images pass MS-SSIM filters.

### test_validation.py
- [`test_get_ds_avg_msssim()`](src/rawnind/dataset/tests/test_validation.py): Computes and validates average MS-SSIM across a dataset.
- [`test_check_whether_wb_is_needed_before_demosaic()`](src/rawnind/dataset/tests/test_validation.py): Determines if white balance is required prior to demosaicing for Bayer data.
- [`test_get_models_complexity()`](src/rawnind/dataset/tests/test_validation.py): Calculates and verifies model complexity metrics (e.g., parameters, FLOPs).

## Dependencies Module Tests (src/rawnind/dependencies/tests/)

### test_config_manager.py
- [`test_load_config()`](src/rawnind/dependencies/tests/test_config_manager.py): Loads a valid YAML config and verifies parsed values.
- [`test_load_config_nonexistent_file()`](src/rawnind/dependencies/tests/test_config_manager.py): Handles missing config file with default fallback.
- [`test_get_training_config()`](src/rawnind/dependencies/tests/test_config_manager.py): Extracts training-specific subsection from config.
- [`test_get_test_config()`](src/rawnind/dependencies/tests/test_config_manager.py): Retrieves test config subsection.
- [`test_get_model_config()`](src/rawnind/dependencies/tests/test_config_manager.py): Validates model config extraction.
- [`test_save_config()`](src/rawnind/dependencies/tests/test_config_manager.py): Saves config to YAML and reloads to confirm integrity.
- [`test_merge_configs()`](src/rawnind/dependencies/tests/test_config_manager.py): Merges two configs, prioritizing overrides.
- [`test_merge_configs_deep_nesting()`](src/rawnind/dependencies/tests/test_config_manager.py): Tests deep nested merging without conflicts.
- [`test_merge_configs_non_dict_override()`](src/rawnind/dependencies/tests/test_config_manager.py): Ensures non-dict values override correctly.

### test_file_io.py
- [`test_hdr_nparray_to_file_integration()`](src/rawnind/dependencies/tests/test_file_io.py): Saves HDR numpy array to EXR/TIFF and reloads.
- [`test_file_io_integration_exr_save[lin_rec2020-16-.exr]`](src/rawnind/dependencies/tests/test_file_io.py): Validates 16-bit linear Rec.2020 EXR save/load roundtrip.
- [`test_file_io_integration_exr_save[lin_sRGB-32-.exr]`](src/rawnind/dependencies/tests/test_file_io.py): Tests 32-bit linear sRGB EXR integration.
- [`test_file_io_integration_tiff_save[lin_rec2020-16-.tif]`](src/rawnind/dependencies/tests/test_file_io.py): Confirms 16-bit linear Rec.2020 TIFF save.
- [`test_file_io_integration_tiff_save[gamma_sRGB-None-.tif]`](src/rawnind/dependencies/tests/test_file_io.py): Validates gamma-corrected sRGB TIFF without bit depth limit.
- [`TestXTransProcessing::test_is_xtrans_true()`](src/rawnind/dependencies/tests/test_file_io.py): Detects X-Trans sensor from RAF file.
- [`TestXTransProcessing::test_is_xtrans_false()`](src/rawnind/dependencies/tests/test_file_io.py): Confirms non-X-Trans files are correctly identified.
- [`TestXTransProcessing::test_xtrans