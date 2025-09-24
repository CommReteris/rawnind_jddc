You are right. My previous FRS was still not detailed enough. I will now create a new, more comprehensive FRS that captures all of the nuances from the legacy implementation.

***

# Functional Requirements Specification (FRS)
## for the `rawnind.dataset` Package

**Version:** 2.0
**Date:** 2025-09-24

### 1. Introduction

#### 1.1 Purpose
This document provides a detailed description of the functions, features, and behaviors of the `rawnind.dataset` package. Its purpose is to serve as the definitive guide for the implementation, testing, and validation of the data loading and preparation pipeline for the RawNIND project. It ensures that all critical domain-specific logic from the legacy system (`legacy_rawds.py`) is retained in the refactored architecture.

#### 1.2 Scope
The scope of this FRS is strictly limited to the `rawnind.dataset` package. This includes the configuration, initialization, data filtering, augmentation, processing, and batching of all image datasets used by the `rawnind.training` and `rawnind.inference` packages. It also specifies the public API through which other packages will interact with the datasets.

#### 1.3 System Overview
The `rawnind.dataset` package is a core component responsible for loading raw and processed image data from disk, applying a series of transformations and augmentations, and serving it in a format suitable for consumption by PyTorch models. It is designed to be highly configurable, deterministic for evaluation, and robust in handling large-scale scientific image data.

---

### 2. Overall Description

#### 2.1 System Architecture
The system is built on a modular, configuration-driven architecture:
*   **`DatasetConfig` (Dataclass):** A centralized dataclass acting as the Single Source of Truth (SOT) for all dataset parameters, from file paths to augmentation strategies and quality thresholds.
*   **`ConfigurableDataset` (Core Class):** The main dataset class that interprets the `DatasetConfig` to perform data loading, filtering, processing, and augmentation.
*   **`RawImageDataset` (Base Class):** A lower-level utility class containing fundamental, stateless image operations like cropping.
*   **Specialized Datasets:** Subclasses (`CleanValidationDataset`, `PreCroppedDataset`) that inherit from or compose the core classes to provide specialized behavior for validation, testing, or handling pre-processed data.
*   **`clean_api.py` (Public Interface):** A single module providing factory functions as the exclusive public entry point for creating dataset instances.

#### 2.2 User Characteristics
The "users" of this package are other software components within the project, primarily:
*   The `rawnind.training` package, which requires batched, augmented data for model training.
*   The `rawnind.inference` package, which requires deterministically processed data for model evaluation.

#### 2.3 General Constraints
*   The system must be fully compatible with `torch.utils.data.Dataset` and `torch.utils.data.DataLoader`.
*   All dataset metadata shall be loaded from YAML files.
*   Image data will be loaded from standard file formats (e.g., `.exr`, `.png`, `.pt`).

---

### 3. Specific Functional Requirements

#### 3.1 FR-CONF: System Configuration (`DatasetConfig`)
The system's behavior shall be controlled entirely by the `DatasetConfig` dataclass, which must include, at a minimum, the following configurable parameters:

*   **`FR-CONF-01`:** `content_fpaths` (list[str]): A list of file paths to the YAML metadata files.
*   **`FR-CONF-02`:** `dataset_type` (str): Defines the dataset category (e.g., "bayer", "rgb").
*   **`FR-CONF-03`:** `data_format` (str): Specifies the input/output format (e.g., "Bayer", "ProfiledRGB").
*   **`FR-CONF-04`:** `crop_size` (int): The edge length of a square crop. Must be an even number.
*   **`FR-CONF-05`:** `num_crops` (int): The number of random crops to extract per image during training.
*   **`FR-CONF-06`:** `batch_size` (int): The number of items per batch.
*   **`FR-CONF-07`:** `data_pairing` (str): A string enum ('x_y', 'x_x', 'y_y') to control clean/noisy pairing. 'x_y' is the default.
*   **`FR-CONF-08`:** `match_gain` (bool): If true, normalizes exposure by applying the metadata gain value to the noisy image. Defaults to `False`.
*   **`FR-CONF-09`:** `augmentations` (list[str]): A list of augmentation names to apply (e.g., `['flip_vertical', 'rotate_90']`). Defaults to an empty list.
*   **`FR-CONF-10`:** `pre_cropped` (bool): If true, signals the use of a specialized dataset for pre-cropped data on disk. Defaults to `False`.
*   **`FR-CONF-11`:** `min_msssim_score` (float, optional): The minimum acceptable MS-SSIM score for an image pair to be included.
*   **`FR-CONF-12`:** `max_alignment_loss` (float, optional): The maximum acceptable alignment error for an image pair to be included.
*   **`FR-CONF-13`:** `mask_mean_min` (float, optional): The minimum acceptable mean of the validity mask for an image pair to be included.
*   **`FR-CONF-14`:** `bayer_only` (bool): If `True`, the dataset will filter out and use only images that have associated Bayer data (`is_bayer: true` in metadata). Defaults to `True`.
*   **`FR-CONF-15`:** `test_reserve` (list[str], optional): A list of `image_set` names to be exclusively used or excluded, depending on the dataset mode (training vs. testing).
*   **`FR-CONF-16`:** `test_mode` (bool): An internal flag, set by the factory functions, to indicate if the dataset is for training or testing, which controls the `test_reserve` logic. Defaults to `False`.
*   **`FR-CONF-17`:** `toy_dataset` (bool): If `True`, the dataset will truncate its length to a small, predefined number (e.g., 25) for rapid debugging. Defaults to `False`.

#### 3.2 FR-LOAD: Data Loading and Filtering (`ConfigurableDataset`)
*   **`FR-LOAD-01`:** The system shall load all image metadata from the YAML files specified in `content_fpaths`.
*   **`FR-LOAD-02`:** During initialization, the system shall iterate through all image entries from the loaded metadata and apply the following sequence of filters to determine inclusion in the final dataset:
    *   **FR-LOAD-02a (Bayer Filter):** If `bayer_only` is true, the system shall discard any image where the metadata field `is_bayer` is `False`.
    *   **FR-LOAD-02b (Test/Train Split Filter):** The system shall discard images based on the `test_mode` flag and the `test_reserve` list. If `test_mode` is `False` (training), discard any image whose `image_set` is in `test_reserve`. If `test_mode` is `True` (testing), discard any image whose `image_set` is *not* in `test_reserve`.
    *   **FR-LOAD-02c (Quality Filter):** The system shall discard any image that fails to meet any of the specified quality thresholds: `min_msssim_score`, `max_alignment_loss`, or `mask_mean_min`.
    *   **FR-LOAD-02d (Empty Crops Filter):** The system shall discard any image entry for which the `crops` list in the metadata is empty.
*   **`FR-LOAD-03 (Toy Mode):`** If `toy_dataset` is `True`, the system shall stop the loading and filtering process as soon as the dataset reaches a predefined length (`TOY_DATASET_LEN`).
*   **`FR-LOAD-04`:** The system must raise a critical error and halt execution if the final filtered dataset contains zero images.

#### 3.3 FR-PROC: Per-Item Data Processing (`ConfigurableDataset.__getitem__`)
For each item requested, the system shall perform the following sequence of operations:

*   **`FR-PROC-01`:** Randomly select one crop dictionary from the `crops` list of the requested image's metadata.
*   **`FR-PROC-02`:** Load the appropriate clean and/or noisy image files from disk based on the file paths in the selected crop dictionary and the `data_pairing` configuration.
*   **`FR-PROC-03`:** If `match_gain` is enabled, multiply the noisy image tensor by the `raw_gain` or `rgb_gain` value from the metadata.
*   **`FR-PROC-04`:** Apply pixel-level alignment to the image pair using the `best_alignment` vector from the metadata. This must occur before any other spatial transformations.
*   **`FR-PROC-05`:** Apply the sequence of augmentations (e.g., flips, rotations) as specified in the `augmentations` list in the config.
*   **`FR-PROC-06`:** Perform demosaicing and/or color space conversions as required to match the target `data_format` in the config.
*   **`FR-PROC-07`:** Generate a pixel-wise overexposure mask based on the `overexposure_lb` metadata value.
*   **`FR-PROC-08`:** Extract one or more crops from the processed image using either random or deterministic cropping methods (see FR-EVAL).
*   **`FR-PROC-09`:** For random cropping, the system must ensure that any selected crop contains a percentage of valid pixels greater than the `MAX_MASKED` threshold, retrying up to `MAX_RANDOM_CROP_ATTEMPS` times.
*   **`FR-PROC-10 (Runtime Self-Correction):`** If, after the maximum number of attempts, a valid crop cannot be found, the system shall:
    1.  Log a warning specifying the problematic crop.
    2.  Permanently remove that crop from the in-memory list (`self._dataset[i]['crops']`).
    3.  If the `crops` list for that image becomes empty, log a second warning and permanently remove the entire image entry from the in-memory dataset (`self._dataset`).
    4.  Recursively call `__getitem__` with the same index to fetch a new, valid item.
*   **`FR-PROC-11`:** The method shall return a dictionary containing, at minimum, `x_crops` (clean/source), `y_crops` (noisy/target), `mask_crops`, `gain`, and `rgb_xyz_matrix`.

#### 3.4 FR-EVAL: Specialized Evaluation Loading
*   **`FR-EVAL-01`:** The `CleanValidationDataset` and `CleanTestDataset` classes shall not use random cropping. All transformations must be deterministic.
*   **`FR-EVAL-02`:** For validation (`CleanValidationDataset`), the system shall deterministically select the middle crop from the `crops` list and perform a `center_crop` on it.
*   **`FR-EVAL-03`:** The test dataset (`CleanTestDataset`) shall implement a custom iterator (`__iter__`) that processes the entire source image by generating and yielding a sequence of tiled/overlapping crops in a sliding window fashion. The crop size shall be determined by `DatasetConfig.crop_size`. If `crop_size` is 0, the image should be padded to be divisible by 256 and returned as a single crop.

#### 3.5 FR-PRE: Pre-Cropped Data Handling
*   **`FR-PRE-01`:** If `DatasetConfig.pre_cropped` is `True`, the public API factory shall instantiate a `PreCroppedDataset`.
*   **`FR-PRE-02`:** The `PreCroppedDataset` shall bypass all dynamic processing steps (alignment, augmentation, cropping). Its `__getitem__` method will load a pre-existing crop file directly from disk.

#### 3.6 FR-API: Public API (`clean_api.py`)
*   **`FR-API-01`:** The package shall expose exactly three public factory functions: `create_training_dataset`, `create_validation_dataset`, and `create_test_dataset`.
*   **`FR-API-02`:** The `create_validation_dataset` and `create_test_dataset` factories must internally set the `test_mode` flag in the `DatasetConfig` to `True`.
*   **`FR-API-03`:** Each factory function shall accept a `DatasetConfig` object as its primary argument and return an appropriate, fully configured dataset instance that is ready for use with a `torch.utils.data.DataLoader`.

---

### 4. Non-Functional Requirements

*   **`NFR-01` Performance:** The data loading pipeline must be memory-efficient and leverage the multi-worker capabilities of PyTorch's `DataLoader` to prevent I/O from becoming a bottleneck during training.
*   **`NFR-02` Reliability:** The system must log warnings for non-critical issues (e.g., a single invalid crop) but fail gracefully with an informative error for critical configuration failures (e.g., empty dataset after filtering).
*   **`NFR-03` Maintainability:** All code, especially the processing logic within `ConfigurableDataset`, must be well-commented and logically organized. All public functions and classes must have clear docstrings.

***