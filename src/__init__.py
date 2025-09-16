"""RawNIND: Source code for learning joint denoising, demosaicing, and compression from raw images.

This package contains implementations of neural network models and utilities for processing
raw camera images. The code supports the research presented in "Learning Joint Denoising,
Demosaicing, and Compression from the Raw Natural Image Noise Dataset".

The package has two main components:
    - common: Shared utilities and libraries used across the project
    - rawnind: Core functionality for raw image processing and neural networks

The implementation focuses on:
    - Processing both Bayer and X-Trans sensor data
    - Joint denoising, demosaicing, and compression of raw images
    - Training and evaluation of neural networks for image processing tasks
    - Dataset handling for paired (clean-noisy) and unpaired (clean-clean) image sets

For detailed usage instructions, refer to the README files in the repository.

"""
