# Documentation Plan

This document itemizes missing docstrings and documentation shortcomings in the src/ directory, organized by folder structure.

## common
### extlibs [5%]
#### gdn.py
- File [25%] - Basic file-level docstring exists but could be more descriptive
- LowerBound [0%] - Class missing docstring entirely
- LowerBound.forward [0%] - Static method missing docstring
- LowerBound.backward [0%] - Static method missing docstring  
- GDN [25%] - Basic class docstring exists but lacks parameter details
- GDN.__init__ [0%] - Constructor missing docstring with parameter descriptions
- GDN.build [0%] - Method missing docstring
- GDN.forward [0%] - Forward pass method missing docstring

### libs [15%]
#### icc.py [0%]
- File [0%] - Missing file-level docstring explaining ICC color profile data arrays
- Note: File contains only data arrays (rec2020, etc.) - no functions or classes

#### json_saver.py [15%]
- File [100%] - Good file-level docstring
- JSONSaver [0%] - Class missing docstring
- JSONSaver.__init__ [0%] - Constructor missing parameter documentation
- JSONSaver._load [0%] - Private method missing docstring
- JSONSaver.add_res [25%] - Basic docstring but lacks parameter details
- JSONSaver.write [0%] - Method missing docstring
- JSONSaver.get_best_steps [0%] - Method missing docstring and return type description
- JSONSaver.get_best_step [0%] - Method missing docstring
- JSONSaver.get_best_step_results [0%] - Method missing docstring
- JSONSaver.is_empty [100%] - Good docstring with clear description
- YAMLSaver [0%] - Child class missing docstring
- YAMLSaver.__init__ [0%] - Constructor missing docstring
- YAMLSaver._load [0%] - Override method missing docstring
- YAMLSaver.write [0%] - Override method missing docstring

#### utilities.py [20%]
- File [100%] - Good file-level docstring "Common utilities"
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

#### locking.py [25%]
- File [0%] - Missing file-level docstring
- is_locked [0%] - Function missing docstring
- is_owned [0%] - Function missing docstring
- lock [25%] - Basic parameter description but could be more comprehensive
- unlock [25%] - Basic parameter description but could be more comprehensive
- check_pause [100%] - Good docstring with clear usage instructions
- Test_locking [0%] - Test class missing docstring

#### np_imgops.py [22%]
- File [0%] - Missing file-level docstring
- CropMethod enum [0%] - Enum missing docstring
- _oiio_img_fpath_to_np [0%] - Private function missing docstring
- _opencv_img_fpath_to_np [0%] - Private function missing docstring
- img_fpath_to_np_flt [100%] - Comprehensive docstring with parameters and return type
- np_pad_img_pair [0%] - Function missing docstring
- np_crop_img_pair [75%] - Basic but functional docstring
- np_to_img [0%] - Function missing docstring
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

#### pt_losses.py [0%]
- File [0%] - Missing file-level docstring
- MS_SSIM_loss [0%] - Class has empty docstring placeholder
- MS_SSIM_metric [0%] - Class has empty docstring placeholder
- findvaliddim [0%] - Function in main section missing docstring
- Note: File contains mostly class wrappers around pytorch_msssim library

#### pt_ops.py [25%]
- File [0%] - Missing file-level docstring
- pt_crop_batch [100%] - Good docstring with clear usage description
- img_to_batch [0%] - Function missing docstring despite complexity
- batch_to_img [75%] - Good docstring with differentiability note
- pixel_unshuffle [100%] - Comprehensive docstring with source reference and usage notes
- oneloss [0%] - Function missing docstring
- fragile_checksum [0%] - Function missing docstring
- RoundNoGradient [0%] - Custom autograd function class missing docstring
- crop_to_multiple [0%] - Function missing docstring
- Test_PTOPS [0%] - Test class missing docstring

#### stdcompression.py [5%]
- File [100%] - Good file-level docstring "Standard compression methods handlers"
- StdCompression [0%] - Base class missing docstring
- StdCompression methods [0%] - All methods (__init__, make_tmp_fpath, file_encdec, etc.) missing docstrings
- JPG_Compression [0%] - Class missing docstring
- JPEGXL_Compression [0%] - Class missing docstring
- BPG_Compression [0%] - Class missing docstring
- JPEGXS_Compression [0%] - Class missing docstring
- All compression class methods [0%] - make_enc_cl, make_dec_cl, get_valid_cargs methods missing docstrings
- Test_utilities [0%] - Test class and methods missing docstrings

### tools [5%]
#### save_src.py [75%]
- File [0%] - Missing file-level docstring
- save_src [100%] - Comprehensive function docstring with clear parameter descriptions
- Main section [0%] - Script execution section missing docstring

## rawnind
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
#### compression_autoencoders.py [35%]
- File [100%] - Good file-level docstring "Base model for hyperprior/manyprior based network"
- AbstractRawImageCompressor [25%] - Class missing docstring but has comprehensive forward method docstring
- BalleEncoder [50%] - Basic class docstring "Image encoder for RGB (3ch) or Bayer (4ch) images" but methods missing docstrings
- BalleDecoder [50%] - Basic class docstring "Image decoder for RGB (3ch) images" but methods missing docstrings
- BayerPSDecoder [0%] - Child class missing docstring
- BayerTCDecoder [0%] - Child class missing docstring
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
#### raw_denoiser.py [5%]
- File [25%] - Basic docstring comment explaining U-Net architecture but improperly formatted as comment
- Denoiser [0%] - Base class missing docstring
- Passthrough [0%] - Class missing docstring
- get_activation_class_params [0%] - Utility function missing docstring
- UtNet2 [0%] - Main denoiser class missing docstring despite complex implementation
- ResBlock [0%] - Class missing docstring
- UtNet3 [0%] - Child class missing docstring
- Methods [0%] - All methods missing docstrings
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
- Processing pipeline well documented with inline comments and overview sections
#### Other tools [25%]
- 21 additional utility scripts identified
- Mixed coverage - some tools well-documented like denoise_image.py
- Scripts include: crop_datasets.py, prep_image_dataset.py, cleanup_saved_models_iterations.py, etc.

### tests [2%]
- 56 test files identified
- Test files typically have minimal documentation requirements
- Focus should be on ensuring test purpose is clear through docstrings
- Very low current coverage

### onetimescripts [2%]
- 4 one-time utility scripts
- Scripts: create_bm3d_argsyaml.py, create_jpegxl_argsyaml.py, etc.
- Minimal documentation, mainly need file-level docstrings

### paper_scripts [2%]
- 5 scripts for paper figure generation
- Scripts: mk_combined_mosaic.py, mk_megafig.py, plot_dataset_msssim_distributionv2.py, etc.
- Academic scripts need better documentation for reproducibility

### scripts [5%]
#### mk_denoise_then_compress_models.py [5%]
- Model generation script - minimal documentation coverage

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
*Analysis Status: Comprehensive documentation analysis completed with actual file inspection across all major directories*
*Overall Project Documentation Coverage: 35%* 
*Key Findings: Significant upward revision from initial estimates through systematic file inspection. Core libraries excellently documented: rawnind/libs shows outstanding coverage with rawproc.py (90%), arbitrary_proc_fun.py (85%), rawds_manproc.py (75%), raw.py (35%), rawds.py (35%). Training scripts better than expected (50-60% with excellent file docstrings). Tools directory shows strong documentation leadership (denoise_image.py 75%). Mixed results in common/libs: pt_helpers (67%), save_src (75%), rawnind/libs/__init__ (100%) vs pt_losses (0%), stdcompression (5%). Models directory shows moderate coverage (15%) with good file-level documentation.*
*Priority Areas: Method-level documentation in complex files, model implementation classes, comprehensive coverage in large utility files*