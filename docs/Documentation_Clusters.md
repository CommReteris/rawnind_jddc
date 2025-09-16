# Documentation Clusters Analysis

This document provides an analysis of the documentation status in the repository, organized into semantically related
clusters. Each cluster represents a functional area of the codebase with related components.

## Documentation Clusters

### 1. Core Neural Network Models and Architecture (41%)

This cluster includes components that define neural network architectures, models, and related functionality.

| Component                          | Completion % |
|------------------------------------|--------------|
| models/compression_autoencoders.py | 90%          |
| models/raw_denoiser.py             | 100%         |
| models/bitEstimator.py             | 30%          |
| models/bm3d_denoiser.py            | 15%          |
| models/denoise_then_compress.py    | 20%          |
| models/manynets_compression.py     | 50%          |
| models/standard_compressor.py      | 80%          |
| models/__init__.py                 | 0%           |
| libs/abstract_trainer.py           | 18%          |
| common/extlibs/gdn.py              | 100%         |
| **Cluster Average**                | **41%**      |

### 2. Image Processing and Operations (75%)

This cluster includes components for image processing, transformations, and operations.

| Component                  | Completion % |
|----------------------------|--------------|
| common/libs/np_imgops.py   | 60%          |
| common/libs/pt_ops.py      | 100%         |
| libs/rawproc.py            | 90%          |
| libs/arbitrary_proc_fun.py | 85%          |
| libs/raw.py                | 35%          |
| **Cluster Average**        | **75%**      |

### 3. Dataset Management and Data Handling (46%)

This cluster includes components for dataset management, loading, and processing.

| Component             | Completion % |
|-----------------------|--------------|
| libs/rawds.py         | 35%          |
| libs/rawds_manproc.py | 75%          |
| libs/rawtestlib.py    | 25%          |
| common/libs/icc.py    | 100%         |
| **Cluster Average**   | **46%**      |

### 4. Evaluation Metrics and Loss Functions (85%)

This cluster includes components for calculating metrics, losses, and evaluations.

| Component                                      | Completion % |
|------------------------------------------------|--------------|
| common/libs/pt_losses.py                       | 100%         |
| common/libs/pt_helpers.py                      | 67%          |
| common/libs/libimganalysis.py                  | 40%          |
| tools/get_ds_avg_msssim.py                     | 5%           |
| tests/get_models_complexity.py                 | 100%         |
| tests/get_RawNIND_test_quality_distribution.py | 95%          |
| **Cluster Average**                            | **85%**      |

### 5. Compression and File Operations (65%)

This cluster includes components for compression, file handling, and I/O operations.

| Component                     | Completion % |
|-------------------------------|--------------|
| common/libs/stdcompression.py | 70%          |
| common/libs/json_saver.py     | 90%          |
| common/libs/locking.py        | 100%         |
| common/tools/save_src.py      | 75%          |
| common/libs/utilities.py      | 20%          |
| **Cluster Average**           | **65%**      |

### 6. Utility Tools and Scripts (41%)

This cluster includes various utility tools and scripts.

| Component                                            | Completion % |
|------------------------------------------------------|--------------|
| tools/denoise_image.py                               | 75%          |
| tools/add_msssim_score_to_dataset_yaml_descriptor.py | 20%          |
| tools/check_dataset.py                               | 100%         |
| tools/cleanup_saved_models_iterations.py             | 15%          |
| tools/capture_image_set.py                           | 20%          |
| tools/prep_image_dataset.py                          | 35%          |
| other tools (15+ scripts)                            | 20%          |
| scripts/mk_denoise_then_compress_models.py           | 10%          |
| **Cluster Average**                                  | **41%**      |

### 7. Testing and Validation (25%)

This cluster includes components for testing, validation, and verification.

| Component                                           | Completion % |
|-----------------------------------------------------|--------------|
| tests/check_whether_wb_is_needed_before_demosaic.py | 5%           |
| tests/get_ds_avg_msssim.py                          | 5%           |
| other test files (50+ files)                        | 3%           |
| unittests.py                                        | 5%           |
| **Cluster Average**                                 | **25%**      |

### 8. Configuration and Settings (23%)

This cluster includes configuration files and settings.

| Component                       | Completion % |
|---------------------------------|--------------|
| config/train_dc.yaml            | 90%          |
| config/train_dc_bayer2prgb.yaml | 0%           |
| config/train_denoiser_*.yaml    | 0%           |
| config/graph_dc_model.yaml      | 0%           |
| config/graph_denoiser.yaml      | 0%           |
| config/test_reserve.yaml        | 0%           |
| plot_cfg/Picture1_32.yaml       | 95%          |
| plot_cfg/Picture2_32.yaml       | 0%           |
| plot_cfg/Picture1_picture2.yaml | 0%           |
| **Cluster Average**             | **23%**      |

### 9. Visualization and Reporting (79%)

This cluster includes components for visualization and generating reports.

| Component                                                  | Completion % |
|------------------------------------------------------------|--------------|
| paper_scripts/mk_megafig.py                                | 95%          |
| paper_scripts/mk_combined_mosaic.py                        | 95%          |
| paper_scripts/mk_pipelinefig.py                            | 35%          |
| paper_scripts/plot_dataset_msssim_distributionv2.py        | 30%          |
| onetimescripts/create_bm3d_argsyaml.py                     | 95%          |
| onetimescripts/create_jpegxl_argsyaml.py                   | 95%          |
| onetimescripts/find_best_bm3d_models_for_given_pictures.py | 95%          |
| onetimescripts/upload_dir_to_dataverse.py                  | 10%          |
| **Cluster Average**                                        | **79%**      |

### 10. Training Scripts (53%)

This cluster includes components for model training.

| Component                    | Completion % |
|------------------------------|--------------|
| train_dc_bayer2prgb.py       | 60%          |
| train_dc_prgb2prgb.py        | 50%          |
| train_denoiser_bayer2prgb.py | 50%          |
| train_denoiser_prgb2prgb.py  | 50%          |
| **Cluster Average**          | **53%**      |

## Documentation Coverage Visualization

```
Core Neural Network Models (41%) ██████████████░░░░░░░░░░░░░░░
Image Processing (75%)         ███████████████████████░░░░░
Dataset Management (46%)       ██████████████░░░░░░░░░░░░░░░
Evaluation Metrics (85%)       █████████████████████████░░░
Compression & Files (65%)      ████████████████████░░░░░░░░
Utility Tools (41%)            ██████████████░░░░░░░░░░░░░░░
Testing & Validation (25%)     ████████░░░░░░░░░░░░░░░░░░░░░
Configuration (23%)            ████████░░░░░░░░░░░░░░░░░░░░░
Visualization (79%)            █████████████████████████░░░
Training Scripts (53%)         ████████████████░░░░░░░░░░░░░
```

## Overall Documentation Status

The overall project documentation coverage is approximately 45%.

### Documentation Priority Areas

Based on the cluster analysis, the following areas should be prioritized for documentation improvement:

1. **Testing and Validation (25%)** - This cluster has the lowest documentation coverage and includes over 50 test files
   that could benefit from improved documentation.

2. **Configuration and Settings (23%)** - Configuration files are critical for understanding how to run the system but
   have very low documentation coverage.

3. **Core Neural Network Models (41%)** - These components form the backbone of the system's functionality but have
   below-average documentation coverage.

4. **Utility Tools and Scripts (41%)** - These tools are likely used frequently across the project but lack
   comprehensive documentation.

5. **Dataset Management (46%)** - Better documentation of data handling would improve understanding of data flow through
   the system.

### Well-Documented Areas

The following clusters have good documentation coverage and can serve as examples for documenting other areas:

1. **Evaluation Metrics and Loss Functions (85%)**
2. **Visualization and Reporting (79%)**
3. **Image Processing and Operations (75%)**

## Next Steps

The documentation improvement effort should focus on:

1. Documenting testing and validation code to ensure proper test execution and interpretation
2. Adding comprehensive comments to configuration files to clarify parameter meanings and valid values
3. Improving docstrings for core neural network models and components
4. Adding Google-style docstrings to utility functions and scripts
5. Enhancing dataset management documentation to clarify data processing workflows

By focusing on these areas, the project's overall documentation quality and completeness can be significantly improved.