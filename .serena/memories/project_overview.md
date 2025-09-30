# Project Overview
- **Purpose**: Implements research code for "Learning Joint Denoising, Demosaicing, and Compression from the Raw Natural Image Noise Dataset". Provides production-ready dataset loaders, training, and inference tooling for neural networks that transform RAW camera captures into compressed outputs.
- **Core Functionality**: Handles RAW Bayer image ingestion, quality filtering, alignment, crop sampling, and mask computation; trains deep denoise/compress models; supplies inference pipelines for processing new captures.
- **High-Level Architecture**:
  - `dependencies/`: Shared image/RAW processing utilities, PyTorch helpers, configuration assets.
  - `dataset/`: Clean API datasets (incl. unified `ConfigurableDataset`), base cropping/mask logic, validation/test loaders.
  - `training/`: Trainers, loops, experiment orchestration, configuration integration.
  - `inference/`: Model loading, denoiser/compressor execution, evaluation metrics.
- **Domain Assets**: Relies on RawNIND paired clean/noisy dataset with YAML descriptors, masks, and alignment metadata.
- **Key Features**: Supports multiple Bayer mosaics, joint denoise+compression, modular configuration for clean/noisy RGB & Bayer pipelines.