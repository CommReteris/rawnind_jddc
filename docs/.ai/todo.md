**Dataset Package Completion**
Extract ProfiledRGBBayerImageDataset class from rawds.py to dataset/bayer_datasets.py
Extract ProfiledRGBProfiledRGBImageDataset class from rawds.py to dataset/rgb_datasets.py
Extract TestDataLoader class from rawds.py to dataset/test_dataloaders.py
Extract CleanProfiledRGBCleanBayerImageCropsDataset from rawds.py to dataset/clean_datasets.py
Extract CleanProfiledRGBCleanProfiledRGBImageCropsDataset from rawds.py to dataset/clean_datasets.py
Extract CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset from rawds.py to dataset/validation_datasets.py
Extract CleanProfiledRGBNoisyBayerImageCropsValidationDataset from rawds.py to dataset/validation_datasets.py
Extract CleanProfiledRGBNoisyBayerImageCropsTestDataloader from rawds.py to dataset/test_dataloaders.py
Extract CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader from rawds.py to dataset/test_dataloaders.py
Extract DataLoadersUnitTests from rawds.py to dataset/tests/test_datasets.py
Update imports in dataset package files to use new structure
Remove extracted classes from rawds.py
**Inference Package Completion**
Extract ImageToImageNN base class from abstract_trainer.py to inference/base_inference.py
Extract BayerImageToImageNN class from abstract_trainer.py to inference/base_inference.py
Extract DenoiseCompress class from abstract_trainer.py to inference/model_factory.py
Extract Denoiser class from abstract_trainer.py to inference/model_factory.py
Extract BayerDenoiseCompress class from abstract_trainer.py to inference/model_factory.py
Extract BayerDenoiser class from abstract_trainer.py to inference/model_factory.py
Extract get_and_load_test_object function from abstract_trainer.py to inference/model_factory.py
Extract get_and_load_model function from abstract_trainer.py to inference/model_factory.py
Update inference package imports to use dependencies package
Remove extracted classes from abstract_trainer.py
**Dependencies Package Completion**
Move pt_ops.py from libs/ to dependencies/pytorch_operations.py
Move np_imgops.py from libs/ to dependencies/numpy_operations.py
Move raw.py from libs/ to dependencies/raw_processing.py
Move rawproc.py from libs/ to dependencies/raw_processing.py
Move icc.py from libs/ to dependencies/color_management.py
Move arbitrary_proc_fun.py from libs/ to dependencies/arbitrary_processing.py
Move stdcompression.py from libs/ to dependencies/compression.py
Move libimganalysis.py from libs/ to dependencies/image_analysis.py
Move locking.py from libs/ to dependencies/locking.py
Create dependencies/configs/ subdirectory
Move all YAML config files from config/ to dependencies/configs/
Move gdn.py from common/extlibs/ to dependencies/external_libraries.py
Update all import statements in dependencies package files
Update imports across codebase to use new dependencies package structure
**Test Suite Reorganization**
Move test_pytorch_integration.py from tests/ to inference/tests/
Move test_bm3d_denoiser.py from tests/ to inference/tests/
Move test_standard_compressor.py from tests/ to inference/tests/
Move test_ext_raw_denoise.py from tests/ to inference/tests/
Move test_datasets_load_time.py from tests/ to dataset/tests/
Move test_manproc.py from tests/ to dataset/tests/
Move test_validation.py from tests/ to dataset/tests/
Move test_alignment.py from tests/ to training/tests/
Move test_progressive_rawnind.py from tests/ to training/tests/
Move test_validate_and_test.py from tests/ to training/tests/
Move rawtestlib.py from tests/ to dependencies/testing_utils.py
Update test imports to use new package structure
Update test configurations and paths
**Training Package Updates**
Update training_loops.py imports to use dependencies package
Update denoise_compress_trainer.py imports to use dependencies package
Update denoiser_trainer.py imports to use dependencies package
Update experiment_manager.py imports to use dependencies package
**Root Package Updates**
Update train_*.py files in root to import from new training package
Update tools/ scripts to import from new packages
Update any remaining imports in libs/ files
**Cleanup Phase**
Remove empty or deprecated files from libs/
Remove old config/ directory after moving configs
Update documentation and README files
Run comprehensive tests to ensure functionality
Remove monolithic abstract_trainer.py and rawds.py files