"""Dataset loading performance benchmark for the RawNIND image processing pipeline.

This script measures and analyzes the loading time for different dataset types in the 
rawds module, which is critical for optimizing training and inference pipelines.
Performance bottlenecks in dataset loading can significantly impact overall training
efficiency, especially when working with large high-resolution raw image datasets.

The test evaluates four key dataset classes:
1. CleanProfiledRGBNoisyBayerImageDataset - Clean profiled RGB paired with noisy Bayer images
2. CleanProfiledRGBNoisyProfiledRGBImageDataset - Clean profiled RGB paired with noisy profiled RGB
3. CleanProfiledRGBCleanBayerImageDataset - Clean profiled RGB paired with clean Bayer images
4. CleanProfiledRGBCleanProfiledRGBImageDataset - Clean profiled RGB paired with clean profiled RGB

For each dataset class, the script:
- Creates an instance with standard training parameters
- Measures the time to load each sample
- Calculates and reports min/avg/max loading times

This benchmark helps identify:
- Which dataset types are most resource-intensive
- Whether certain image formats lead to loading bottlenecks
- How image resolution and crop size affect loading performance
- The impact of dataset parameters on throughput

The results can guide optimization efforts and infrastructure planning for
training deep learning models on large image datasets.

Usage:
    python test_datasets_load_time.py

Output:
    - Log file with detailed timing for each image
    - Summary statistics (min/avg/max) for each dataset type
"""

import logging
import os
import time
import statistics
import sys

from ..libs import rawds
from ..libs import rawproc
from ..libs import abstract_trainer

# Configure log file path
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
os.makedirs(os.path.dirname(LOG_FPATH), exist_ok=True)


def measure_train_images_load_time(dataset_class: rawds.RawImageDataset, **kwargs):
    """Measure the load time for each image in the dataset.
    
    Args:
        dataset_class: The dataset class to benchmark (subclass of RawImageDataset)
        **kwargs: Parameters to pass to the dataset constructor (crop_size, num_crops, etc.)
        
    Returns:
        dict: A dictionary with timing statistics including 'min', 'avg', and 'max' load times
    """
    # Initialize the dataset with the provided parameters
    dataset = dataset_class(**kwargs)
    start_time = time.time()
    timings = []

    # Iterate through each item in the dataset and measure loading time
    for i, image in enumerate(dataset):
        # Calculate how long it took to fetch this item
        fetch_time = time.time() - start_time
        timings.append(fetch_time)

        # Extract and log the file paths being loaded, which vary by dataset type
        fpaths = {}
        if isinstance(dataset, rawds.CleanProfiledRGBCleanBayerImageDataset):
            # This dataset contains paired clean RGB and clean Bayer data
            fpaths["x"], fpaths["y"] = dataset.ds_xy_fpaths[i]
        elif isinstance(dataset, rawds.CleanProfiledRGBCleanProfiledRGBImageDataset):
            # This dataset contains only clean profiled RGB data
            fpaths["x"] = dataset.ds_fpaths[i]
        elif isinstance(dataset, rawds.CleanProfiledRGBNoisyBayerImageDataset):
            # This dataset contains clean RGB paired with noisy Bayer data
            fpaths["x"] = dataset.dataset[i]["gt_linrec2020_fpath"]
            fpaths["y"] = dataset.dataset[i]["f_bayer_fpath"]
        elif isinstance(dataset, rawds.CleanProfiledRGBNoisyProfiledRGBImageDataset):
            # This dataset contains clean RGB paired with noisy RGB data
            fpaths["x"] = dataset.dataset[i]["gt_linrec2020_fpath"]
            fpaths["y"] = dataset.dataset[i]["f_linrec2020_fpath"]

        # Log the image paths and their fetch time
        logging.info(f"Image(s) {fpaths} fetch time: {fetch_time:.2f} seconds. ")

        # Reset timer for the next image
        start_time = time.time()

    # Return statistics summarizing the load times
    return {
        "min": min(timings),  # Fastest load time
        "avg": statistics.mean(timings),  # Average load time
        "max": max(timings)  # Slowest load time
    }


if __name__ == "__main__":
    # Configure logging to write to both file and console
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",  # Overwrite existing log file
    )

    # Add stdout handler to display logs in console as well
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"# python {' '.join(sys.argv)}")
    logging.info(f"Starting dataset load time benchmark...")

    # ----------------------------------------------------------
    # Test 1: Clean Profiled RGB vs Noisy Bayer Dataset
    # ----------------------------------------------------------
    # This dataset is used for training models that convert from noisy Bayer data
    # to clean profiled RGB images (typical raw photo denoising pipeline)
    logging.info(f"Testing CleanProfiledRGBNoisyBayerImageDataset runtimes...")
    timings = measure_train_images_load_time(
        rawds.CleanProfiledRGBNoisyBayerImageDataset,
        num_crops=16,  # Number of random crops per image
        content_fpath=rawproc.RAWNIND_CONTENT_FPATH,  # Path to RawNIND dataset descriptor
        crop_size=256,  # Size of image crops in pixels
        test_reserve=[],  # No images reserved for testing
    )
    logging.info(f"CleanProfiledRGBNoisyBayerImageDataset timing: {timings}")

    # ----------------------------------------------------------
    # Test 2: Clean Profiled RGB vs Noisy Profiled RGB Dataset
    # ----------------------------------------------------------
    # This dataset is used for training models that convert from noisy profiled RGB
    # to clean profiled RGB (denoising after demosaicing)
    logging.info(f"Testing CleanProfiledRGBNoisyProfiledRGBImageDataset runtimes...")
    timings = measure_train_images_load_time(
        rawds.CleanProfiledRGBNoisyProfiledRGBImageDataset,
        num_crops=8,  # Fewer crops for RGB data (3 channels vs 1 for Bayer)
        content_fpath=rawproc.RAWNIND_CONTENT_FPATH,  # Path to RawNIND dataset descriptor
        crop_size=256,  # Size of image crops in pixels
        test_reserve=[],  # No images reserved for testing
    )
    logging.info(f"CleanProfiledRGBNoisyProfiledRGBImageDataset timing: {timings}")

    # ----------------------------------------------------------
    # Test 3: Clean Profiled RGB vs Clean Bayer Dataset
    # ----------------------------------------------------------
    # This dataset is used for training models that convert from clean Bayer data
    # to clean profiled RGB (demosaicing without denoising)
    logging.info(f"Testing CleanProfiledRGBCleanBayerImageDataset runtimes...")
    timings = measure_train_images_load_time(
        rawds.CleanProfiledRGBCleanBayerImageDataset,
        num_crops=16,  # Number of random crops per image
        data_dpath=abstract_trainer.EXTRARAW_DATA_DPATH,  # Path to ExtraRaw dataset
        crop_size=256,  # Size of image crops in pixels
    )
    logging.info(f"CleanProfiledRGBCleanBayerImageDataset timing: {timings}")

    # ----------------------------------------------------------
    # Test 4: Clean Profiled RGB vs Clean Profiled RGB Dataset
    # ----------------------------------------------------------
    # This dataset is used for training models that convert between different
    # color profiles or compression methods in RGB space
    logging.info(f"Testing CleanProfiledRGBCleanProfiledRGBImageDataset runtimes...")
    timings = measure_train_images_load_time(
        rawds.CleanProfiledRGBCleanProfiledRGBImageDataset,
        num_crops=8,  # Fewer crops for RGB data
        data_dpath=abstract_trainer.EXTRARAW_DATA_DPATH,  # Path to ExtraRaw dataset
        crop_size=256,  # Size of image crops in pixels
    )
    logging.info(f"CleanProfiledRGBCleanProfiledRGBImageDataset timing: {timings}")

    logging.info("Dataset load time benchmark complete.")
