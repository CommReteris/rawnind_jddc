# Documentation Plan

This document itemizes missing docstrings and documentation shortcomings in the src/ directory, organized by folder structure.

## common
### extlibs [100%]
#### gdn.py
- File [100%] - Comprehensive file-level docstring with explanation of GDN purpose and academic reference
- LowerBound [100%] - Detailed class docstring explaining the custom autograd function purpose and implementation
- LowerBound.forward [100%] - Thorough static method docstring with parameter and return descriptions
- LowerBound.backward [100%] - Comprehensive static method docstring explaining gradient computation
- GDN [100%] - Detailed class docstring with mathematical formula and thorough explanation
- GDN.__init__ [100%] - Constructor thoroughly documented with all parameter explanations
- GDN.build [100%] - Method fully documented with explanation of parameter initialization
- GDN.forward [100%] - Forward pass method comprehensively documented with input/output descriptions

### libs [50%]
#### icc.py [100%]
- File [100%] - Comprehensive file-level docstring explaining ICC color profile data, its purpose, usage in the project, and references
- Note: File contains only data arrays (rec2020, etc.) which are well commented - no functions or classes need documentation

#### json_saver.py [90%]
- File [100%] - Excellent file-level docstring with purpose, key features, and usage examples
- JSONSaver [100%] - Comprehensive class docstring explaining purpose and functionality
- JSONSaver.__init__ [100%] - Constructor fully documented with parameter explanations
- JSONSaver._load [100%] - Private method well documented with parameters and return values
- JSONSaver.add_res [100%] - Comprehensive docstring with detailed parameter explanations
- JSONSaver.write [100%] - Method well documented with clear purpose
- JSONSaver.get_best_steps [100%] - Method well documented with return type description
- JSONSaver.get_best_step [100%] - Method well documented with parameters and return type
- JSONSaver.get_best_step_results [100%] - Method well documented with parameters and return value
- JSONSaver.is_empty [100%] - Good docstring with clear description
- YAMLSaver [100%] - Child class well documented with explanation of differences from parent
- YAMLSaver.__init__ [100%] - Constructor well documented with parameter explanations
- YAMLSaver._load [100%] - Override method well documented with explanation of differences
- YAMLSaver.write [100%] - Override method well documented with explanation of differences

#### utilities.py [20%]
- File [100%] - Basic file-level docstring "Common utilities"
- checksum [0%] - Function missing docstring
- cp [0%] - Function missing docstring  
- get_date [0%] - Function missing docstring
- backup [100%] - Good docstring "Backup a given list of files per day"
- mt_runner [50%] - Partial docstring, explains some parameters but incomplete
- jsonfpath_load [0%] - Function missing docstring
- dict_to_json [0%] - Function missing docstring
- dict_to_yaml [0%] - Function missing docstring
- load_yaml [0%] - Function missing docstring
- Printer [0%] - Class missing docstring
- Printer.__init__ [0%] - Constructor missing docstring
- Printer.print [0%] - Method missing docstring
- Test_utilities [0%] - Test class missing docstring
- 20+ additional functions lacking documentation

#### libimganalysis.py [40%]
- File [100%] - Good file-level docstring "Image analysis on file paths"
- get_iso [25%] - No main docstring but nested functions have docstrings
- is_raw [0%] - Function missing docstring
- piqa_msssim [0%] - Function missing docstring
- ipx_psnr [100%] - Comprehensive docstring with usage details
- tf_psnr [75%] - Basic docstring but could be more detailed
- tf_msssim [75%] - Basic docstring but could be more detailed
- tf_psnr_cpu [25%] - Minimal docstring
- tf_msssim_cpu [25%] - Minimal docstring
- tf_compare [75%] - Good functional docstring
- pil_get_resolution [0%] - Function missing docstring
- is_valid_img [100%] - Comprehensive docstring with parameter details
- Test_libimganalysis [75%] - Has class docstring but test methods lack docstrings

#### locking.py [100%]
- File [100%] - Comprehensive file-level docstring explaining process-based resource locking purpose and features
- is_locked [100%] - Thoroughly documented function with clear parameter and return descriptions
- is_owned [100%] - Well-documented function with detailed explanation of ownership verification
- lock [100%] - Comprehensive docstring with detailed explanation of exponential backoff strategy
- unlock [100%] - Complete docstring with parameter, return value, and usage recommendation
- check_pause [100%] - Good docstring with clear usage instructions
- Test_locking [100%] - Well-documented test class with thorough explanation of test methodology

#### np_imgops.py [60%]
- File [100%] - Comprehensive file-level docstring with purpose, key features, backend details, and usage examples
- CropMethod enum [100%] - Well-documented enum with explanations of each value
- _oiio_img_fpath_to_np [100%] - Well-documented private function with parameters, return value, and notes
- _opencv_img_fpath_to_np [80%] - Documented but with less detail than the OIIO version
- img_fpath_to_np_flt [100%] - Comprehensive docstring with parameters and return type
- np_pad_img_pair [80%] - Good docstring with clear parameter descriptions
- np_crop_img_pair [75%] - Good docstring explaining functionality
- np_to_img [75%] - Documented with parameters and basic description
- TestImgOps [0%] - Test class and methods missing docstrings

#### pt_helpers.py [67%]
- File [100%] - Good file-level docstring "Helper functions operating on pytorch tensors"
- fpath_to_tensor [0%] - Complex function missing docstring despite importance
- to_smallest_type [100%] - Good docstring with clear purpose
- bits_per_value [100%] - Comprehensive docstring explaining functionality
- get_num_bits [100%] - Detailed docstring with parameters and TODO note
- get_device [100%] - Good docstring with parameter explanation
- sdr_pttensor_to_file [100%] - Clear docstring with format specifications
- get_lossclass [0%] - Function missing docstring
- freeze_model [0%] - Function missing docstring
- get_losses [0%] - Function missing docstring

#### pt_losses.py [100%]
- File [100%] - Comprehensive file-level docstring explaining module purpose, components, and usage
- MS_SSIM_loss [100%] - Thorough class docstring explaining purpose and relation to underlying metrics
- MS_SSIM_metric [100%] - Well-documented class with clear explanation of its evaluation usage
- losses/metrics dictionaries [100%] - Well-documented with explanatory comments
- findvaliddim [100%] - Function fully documented with parameter and return descriptions
- Note: All commented-out code has explanatory comments for context

#### pt_ops.py [100%]
- File [100%] - Comprehensive file-level docstring explaining module purpose and key features
- pt_crop_batch [100%] - Good docstring with clear usage description
- img_to_batch [100%] - Detailed docstring explaining conversion from full images to patches with parameter descriptions
- batch_to_img [100%] - Comprehensive docstring with notes on differentiability and dimensional requirements
- pixel_unshuffle [100%] - Comprehensive docstring with source reference, usage notes, and examples
- oneloss [100%] - Well-documented function with explanation of use cases and parameter descriptions
- fragile_checksum [100%] - Thorough docstring explaining purpose, inputs, outputs and limitations
- RoundNoGradient [100%] - Custom autograd function documented with STE gradient explanation
- crop_to_multiple [100%] - Complete docstring with parameter descriptions, return values, and example
- Test_PTOPS [100%] - Test class with comprehensive documentation of test methodology and purpose

#### stdcompression.py [70%]
- File [100%] - Comprehensive file-level docstring with purpose, key features, and usage examples
- StdCompression [100%] - Base class with detailed docstring explaining purpose and required attributes
- StdCompression.__init__ [100%] - Constructor documented with raises information
- StdCompression.make_tmp_fpath [100%] - Method thoroughly documented with parameters and behavioral notes
- StdCompression methods [50%] - Some methods like file_encdec, make_enc_cl, etc. have good documentation, others are missing
- JPG_Compression [80%] - Class has good documentation but some methods lack docstrings
- JPEGXL_Compression [80%] - Class has good documentation but some methods lack docstrings
- BPG_Compression [80%] - Class has good documentation but some methods lack docstrings
- JPEGXS_Compression [80%] - Class has good documentation but some methods lack docstrings
- Test_utilities [0%] - Test class and methods missing docstrings

### tools [5%]
#### save_src.py [75%]
- File [0%] - Missing file-level docstring
- save_src [100%] - Comprehensive function docstring with clear parameter descriptions
- Main section [0%] - Script execution section missing docstring

## rawnind
### config [13%]
- Directory contains YAML configuration files for training and testing
- train_dc.yaml [90%] - Comprehensive documentation throughout the file:
  - Detailed header explaining configuration purpose and usage
  - Training parameters section with explanations for each setting
  - Dataset configuration section with clear descriptions
  - Model architecture section with parameter explanations
  - Thoroughly commented parameters with value ranges and impacts
  - Consistent section headers for easy navigation
  - Well-organized parameter grouping by functional area
- train_dc_bayer2prgb.yaml [0%] - Missing documentation for configuration parameters
- train_denoiser_*.yaml [0%] - Missing documentation for configuration parameters
- graph_dc_model.yaml [0%] - Missing documentation for configuration parameters
- graph_denoiser.yaml [0%] - Missing documentation for configuration parameters
- test_reserve.yaml [0%] - Missing documentation for test image selection
- Note: The remaining configuration files would benefit from similar comprehensive documentation

### plot_cfg [32%]
- Directory contains plot configuration files for figure generation
- Picture1_32.yaml [95%] - Comprehensive documentation throughout the file:
  - Detailed header explaining visualization purpose and structure
  - Input category with thorough parameter descriptions
  - Denoising category with clear methodology explanations
  - U-Net method with proper Natural Image Noise Dataset (NIND) reference
  - Compression section with detailed key metrics explanation
  - All methods thoroughly documented with purpose and approach
  - Usage Guide section with instructions for modification and extension
- Picture2_32.yaml [0%] - Missing documentation for plot configuration
- Picture1_picture2.yaml [0%] - Missing documentation for plot configuration
- Note: The remaining plot configuration files would benefit from similar documentation

### libs [8%]
#### __init__.py [100%]
- File [100%] - Excellent module docstring explaining package purpose, design decisions, and usage examples
- __all__ [100%] - Well-documented module exports list
#### abstract_trainer.py [18%]
- File [100%] - Good file-level docstring with purpose and TODO notes
- error_handler [0%] - Function missing docstring
- ImageToImageNN [100%] - Excellent comprehensive class docstring explaining design
- ImageToImageNN.__init__ [0%] - Complex constructor missing docstring
- load_model [0%] - Static method missing docstring
- infer [100%] - Good method docstring with purpose and parameters
- get_best_step [100%] - Comprehensive static method docstring
- get_transfer_function [0%] - Static method missing docstring
- Other classes and methods [10%] - Mixed coverage across 8+ classes and 50+ methods
- Note: Large file (2171 lines) with inconsistent documentation coverage
#### arbitrary_proc_fun.py [85%]
- File [100%] - Excellent comprehensive file-level docstring explaining module purpose and design goals
- correct_white_balance [100%] - Good docstring with clear purpose
- apply_tone_mapping_reinhard [75%] - Basic docstring but could be more detailed
- apply_tone_mapping_drago [75%] - Basic docstring but could be more detailed  
- apply_gamma_correction_inplace [100%] - Good docstring with clear purpose
- adjust_contrast [100%] - Comprehensive docstring with parameter explanations
- sigmoid_contrast_enhancement [100%] - Clear docstring with functionality description
- Most functions have good to excellent documentation coverage
#### raw.py [35%]
- File [100%] - Good file-level docstring explaining raw handling library and HDR export
- raw_fpath_to_mono_img_and_metadata [100%] - Excellent comprehensive docstring with detailed parameters and return values
- mono_any_to_mono_rggb [100%] - Good nested function docstring explaining Bayer pattern conversion
- Large file (1178 lines) with 20+ functions, main functions well documented but some utility functions lack docstrings
#### rawds.py [35%]
- File [100%] - Good file-level docstring explaining dataset handlers purpose
- RawImageDataset [100%] - Excellent base class docstring with comprehensive design explanation
- random_crops [100%] - Outstanding method docstring with detailed parameters and return values
- Dataset classes have mixed coverage - some excellent class docstrings, methods mostly undocumented
- CleanProfiledRGBCleanProfiledRGBImageCropsDataset [100%] - Exceptional class documentation
#### rawds_manproc.py [75%]
- File [100%] - Excellent comprehensive file-level docstring explaining module purpose
- ManuallyProcessedImageTestDataHandler [100%] - Outstanding class docstring
- process_lin_rec2020_img [100%] - Exceptional static method documentation with detailed parameters
- High overall coverage with excellent class and method documentation
#### rawproc.py [90%]
- File [100%] - Outstanding comprehensive file-level docstring explaining utilities and design
- np_l1 [100%] - Excellent function documentation with parameters and return values
- gamma [100%] - Comprehensive docstring with detailed parameter explanations
- gamma_pt [100%] - Mirrors gamma documentation for torch tensors
- scenelin_to_pq [100%] - Exceptional documentation with references and implementation notes
- Consistently excellent documentation across sampled functions
#### rawtestlib.py [25%]
- File [100%] - Good file-level docstring explaining lightweight test helpers purpose
- DCTestCustomDataloaderBayerToProfiledRGB [0%] - Class missing docstring
- DenoiseTestCustomDataloaderBayerToProfiledRGB [0%] - Class missing docstring  
- DCTestCustomDataloaderProfiledRGBToProfiledRGB [0%] - Class missing docstring
- DenoiseTestCustomDataloaderProfiledRGBToProfiledRGB [0%] - Class missing docstring
- Methods [0%] - All get_dataloaders methods lack documentation

### models [15%]
#### __init__.py [0%]
- File essentially empty (2 lines with commented import) - no documentation
#### bitEstimator.py [30%]
- File [100%] - Good docstring explaining entropy model with copied ManyPriors classes
- MultiHeadBitEstimator [100%] - Class has docstring explaining CDF estimation
- MultiHeadBitparm [100%] - Component class has docstring
- Methods [0%] - __init__ and forward methods missing docstrings
#### bm3d_denoiser.py [15%]
- File [100%] - Good docstring explaining BM3D denoiser using external binary
- BM3D_Denoiser [0%] - Class missing docstring despite inheriting from Denoiser
- Methods [0%] - __init__ and forward methods missing docstrings despite complex implementations
#### compression_autoencoders.py [90%]
- File [100%] - Comprehensive file-level docstring explaining neural architectures, their purpose, and academic references
- AbstractRawImageCompressor [100%] - Thorough class docstring explaining the compression pipeline and architecture framework
- AbstractRawImageCompressor methods [100%] - All methods (__init__, forward, cpu, todev) fully documented with parameters and returns
- BalleEncoder [100%] - Detailed class docstring explaining architecture, features, and spatial transformations
- BalleEncoder methods [100%] - Constructor and forward method comprehensively documented with parameters and returns
- BalleDecoder [100%] - Thorough class docstring explaining decoder architecture and relationship to encoder
- BalleDecoder methods [100%] - Constructor and forward method fully documented with clear processing steps
- BayerPSDecoder [100%] - Comprehensive docstring explaining its specialized purpose for Bayer pattern processing with PixelShuffle
- BayerTCDecoder [100%] - Detailed docstring explaining its alternative approach using transposed convolutions
- Inline comments [90%] - Thorough inline documentation of complex operations and architecture components
#### denoise_then_compress.py [20%]
- File [100%] - Excellent file-level docstring explaining pipeline model purpose
- DenoiseThenCompress [0%] - Class missing docstring
- Methods [0%] - All methods (__init__, forward, parameters, load_state_dict, get_parameters) missing docstrings
#### manynets_compression.py [50%]
- File [100%] - Excellent comprehensive docstring explaining model basis and implementation
- ManyPriors_RawImageCompressor [25%] - Class missing docstring but has good __init__ and forward method docstrings
- get_parameters [0%] - Method missing docstring
- ManyPriorsEncoder [0%] - Empty class with pass implementation, missing docstring
- ManyPriorsDecoder [0%] - Empty class with pass implementation, missing docstring
- TestRawManyPriors [75%] - Test class has good method docstring explaining test purpose
#### raw_denoiser.py [100%]
- File [100%] - Comprehensive module-level docstring explaining model architectures and input/output formats
- Denoiser [100%] - Well-documented base class with clear abstraction purpose and parameter validation
- Passthrough [100%] - Thoroughly documented identity/baseline model with method explanations
- get_activation_class_params [100%] - Utility function fully documented with parameters, returns, and error handling
- UtNet2 [100%] - Main denoiser class extensively documented with architecture diagram and detailed explanation
- ResBlock [100%] - Clear documentation with ASCII diagram showing the residual connection flow
- UtNet3 [100%] - Thorough documentation explaining enhancements over base UtNet2 implementation
- Methods [100%] - All methods comprehensively documented with parameter descriptions and return values
#### standard_compressor.py [80%]
- File [100%] - Excellent file-level docstring explaining PyTorch interface for standard compression methods
- Std_ImageCompressor [75%] - Good class docstring but could be more detailed
- make_input_image_file [100%] - Good method docstring with clear purpose
- forward [100%] - Comprehensive method docstring with clear step-by-step explanation
- get_parameters [0%] - Method missing docstring
- JPEG_ImageCompressor [100%] - Clear class docstring
- BPG_ImageCompressor [100%] - Clear class docstring
- JPEGXS_ImageCompressor [100%] - Clear class docstring
- JPEGXL_ImageCompressor [100%] - Clear class docstring
- Passthrough_ImageCompressor [100%] - Clear class docstring

### tools [35%]
#### denoise_image.py [75%]
- File [100%] - Excellent comprehensive docstring explaining single image denoising with examples
- add_arguments [100%] - Good function docstring with clear parameter descriptions
- load_image [100%] - Good docstring with clear parameter and return descriptions
- process_image_base [100%] - Excellent docstring with detailed explanations
- apply_nonlinearity [100%] - Clear docstring explaining purpose and parameters
- compute_metrics [100%] - Good docstring with parameter explanations
- save_image [100%] - Clear docstring with parameter explanations
- save_metrics [100%] - Good concise docstring
- denoise_image_from_to_fpath [100%] - Clear docstring
- bayer_to_prgb [100%] - Excellent detailed docstring
- denoise_image_compute_metrics [100%] - Comprehensive docstring with detailed explanations
- denoise_image_from_fpath_compute_metrics_and_export [100%] - Good docstring with parameter descriptions
- Processing pipeline well documented with inline comments and overview sections
#### add_msssim_score_to_dataset_yaml_descriptor.py [20%]
- File [80%] - Good file-level docstring explaining purpose
- Functions [5%] - Most functions missing docstrings
- Main section [0%] - No documentation for execution flow
#### check_dataset.py [100%]
- File [100%] - Comprehensive docstring explaining script's purpose, functionality, usage, and parameters
- is_valid_img_mtrunner [100%] - Function fully documented with clear purpose, parameters, and return value
- Main block [100%] - Thorough step-by-step comments explaining workflow, parameters, and output with user-friendly status messages
#### cleanup_saved_models_iterations.py [15%]
- File [75%] - Basic file-level docstring
- Functions [0%] - All functions missing docstrings
- Main section [0%] - No documentation for execution flow
#### capture_image_set.py [20%]
- File [90%] - Good file-level docstring
- Functions [5%] - Most functions missing docstrings
- Main section [0%] - No documentation for execution flow
#### prep_image_dataset.py [35%]
- File [100%] - Good docstring explaining dataset preparation
- Functions [25%] - Some functions have basic docstrings
- Main section [15%] - Limited documentation for execution flow
#### Other tools [10-30%]
- 15+ additional utility scripts with varying documentation quality
- Most have basic file-level docstrings but lack function-level documentation
- Scripts would benefit from more comprehensive documentation of parameters and return values

### tests [5%]
#### check_whether_wb_is_needed_before_demosaic.py [5%]
- File [25%] - Very basic file-level comment but not a proper docstring
- Functions [0%] - No docstrings for any functions
- Main section [0%] - No documentation for execution flow
#### get_models_complexity.py [100%]
- File [100%] - Comprehensive file-level docstring explaining purpose, metrics, and model architectures
- No functions - script consists of main block only, but this is properly noted in docstring
- Main section [100%] - Thoroughly commented code with explanations of model configurations, input dimensions, and testing approach
#### get_ds_avg_msssim.py [5%]
- File [25%] - Minimal file-level docstring
- Functions [0%] - All functions missing docstrings
- Main section [0%] - No documentation for execution flow
#### get_RawNIND_test_quality_distribution.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, methodology, and usage
- Data loading [100%] - Well-documented YAML loading with clear comments
- Test image selection [100%] - Detailed explanation of each test image set with descriptive comments
- Data processing [100%] - Thorough documentation of MS-SSIM score extraction and sorting
- Visualization [95%] - Comprehensive documentation of histogram creation and parameters
- Alternative visualization [100%] - Detailed explanation of CDF plot with interpretation guidance
- Code organization [90%] - Well-structured with clear sections for each processing step
#### Other test files [3%]
- 50+ additional test files with minimal documentation
- Most lack file-level docstrings explaining test purpose
- Few or no function-level docstrings
- Minimal inline comments explaining test workflows
- Test files would benefit from clear docstrings explaining what is being tested and expected outcomes

### onetimescripts [5%]
#### create_bm3d_argsyaml.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, functionality, and outputs
- Script structure [100%] - Clear organization with section headers as comments
- YAML template [100%] - Thoroughly documented with section headers and parameter explanations
- Configuration matrix [100%] - Well-documented parameter space with explanatory comments
- Main execution block [90%] - Detailed step-by-step comments explaining file generation workflow
- Note: No functions to document as script consists of main block only
#### create_jpegxl_argsyaml.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, functionality, and outputs
- No functions - script consists of main block only
- Configuration matrix [100%] - Well-documented parameter space with explanatory comments
- Main execution block [90%] - Detailed step-by-step comments explaining file generation workflow
#### find_best_bm3d_models_for_given_pictures.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, methodology, and outputs
- Configuration section [100%] - Well-documented paths and model selection
- Test images section [100%] - Detailed explanation of image selection with descriptive comments
- Model evaluation section [100%] - Thorough explanation of result tracking methodology
- Results formatting section [100%] - Clear documentation of output generation
- Main execution flow [90%] - Well-structured with explanatory comments throughout
#### upload_dir_to_dataverse.py [10%]
- File [25%] - Basic file-level docstring explaining purpose
- Functions [5%] - Most functions missing docstrings
- Main section [0%] - No documentation for execution flow

### paper_scripts [20%]
#### mk_megafig.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, functionality, and outputs
- Data structures [100%] - Detailed comments explaining LITERATURE mapping, YAML_FILES, and color conversion matrix
- generate_unique_filename [100%] - Good docstring with purpose and parameter descriptions
- linear_to_srgb [100%] - Well-documented function with implementation explanation
- convert_rec2020_to_srgb [100%] - Comprehensive docstring with parameter and return descriptions
- create_placeholder_with_closeups [100%] - Extensive documentation of this complex function with section headers
- draw_dashed_rectangle [100%] - Thorough docstring explaining the dashed rectangle drawing algorithm
- plot_section [100%] - Comprehensive docstring with detailed parameter explanations and return value
- create_figure [100%] - Thorough documentation explaining the entire figure generation pipeline
- Main execution block [90%] - Well-organized with clear section headers and detailed workflow comments
- Nested loops [100%] - Clear explanation of the loop structure and data flow
#### mk_combined_mosaic.py [95%]
- File [100%] - Comprehensive file-level docstring explaining purpose, features, and usage
- Functions [95%] - All functions thoroughly documented with detailed parameter descriptions
- load_image [100%] - Well-documented with error handling explanation
- crop_to_width [100%] - Detailed explanation of center-aligned cropping methodology
- create_dotted_line [100%] - Thorough documentation of line drawing algorithm and parameters
- combine_images_with_dotted_line [100%] - Comprehensive documentation of image combination workflow
- Main section [95%] - Thoroughly documented command-line interface and execution flow
- Code organization [100%] - Clearly structured with section headers for easy navigation
#### mk_pipelinefig.py [35%]
- File [75%] - Good file-level docstring
- Functions [25%] - Some functions have basic docstrings
- Main section [15%] - Some inline comments explaining workflow
#### plot_dataset_msssim_distributionv2.py [30%]
- File [100%] - Comprehensive file-level docstring
- Functions [15%] - Most functions missing docstrings
- Main section [10%] - Limited documentation for execution flow

### scripts [10%]
#### mk_denoise_then_compress_models.py [10%]
- File [50%] - Basic file-level docstring
- Functions [5%] - Most functions missing docstrings
- Main section [0%] - No documentation for execution flow

### Root level training scripts [50%]
#### train_dc_bayer2prgb.py [60%]
- File [100%] - Excellent docstring explaining joint denoise+compress training with architecture details
- DCTrainingBayerToProfiledRGB [0%] - Class missing docstring but inherits from well-documented mixins
- Methods [25%] - Simple methods with clear implementation but missing docstrings
#### train_dc_prgb2prgb.py [50%]
- Training script pattern similar to train_dc_bayer2prgb.py - good file docstring
#### train_denoiser_bayer2prgb.py [50%]
- Denoising training script pattern similar to others - good file docstring
#### train_denoiser_prgb2prgb.py [50%]
- Denoising training script pattern similar to others - good file docstring
#### unittests.py [5%]
- Unit testing module - minimal documentation coverage

## src root
### __init__.py [0%]
- Root module file is completely empty - no documentation

---
*Analysis Status: Documentation assessment revised with verification of actual file content and correction of significant inaccuracies*
*Overall Project Documentation Coverage: ~45%* 
*Key Findings: The project has varying documentation quality across modules. Many key files have excellent documentation: gdn.py (100%), raw_denoiser.py (100%), get_models_complexity.py (100%), check_dataset.py (100%), pt_losses.py (100%), locking.py (100%), pt_ops.py (100%), compression_autoencoders.py (90%), icc.py (100%), stdcompression.py (70%), json_saver.py (90%), np_imgops.py (60%), create_bm3d_argsyaml.py (95%), create_jpegxl_argsyaml.py (95%), find_best_bm3d_models_for_given_pictures.py (95%), mk_megafig.py (95%), mk_combined_mosaic.py (95%), get_RawNIND_test_quality_distribution.py (95%), denoise_image.py (75%), rawproc.py (90%), and rawds_manproc.py (75%). However, there are significant gaps, particularly in common/libs/utilities.py (20%), many of the test files (3-5%), and various smaller utility scripts (10-30%). The largest source files generally have better documentation than smaller utility files.*
*Priority Areas: Improving documentation for utilities.py, test files, and numerous small utility scripts would significantly increase overall project documentation coverage. Adding missing class and method docstrings to well-used utility functions should be prioritized over less frequently used test files.*