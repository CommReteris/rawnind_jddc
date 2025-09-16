"""Denoising model evaluation on manually processed Bayer pattern images.

This script evaluates a trained denoising model on manually processed images,
converting from Bayer pattern (raw sensor data) inputs to profiled RGB outputs.
The "manproc" (manually processed) dataset contains carefully curated images with
known characteristics, providing a controlled test environment for model evaluation.

The test uses manually processed images that have:
1. Clean target images with professional processing
2. Corresponding raw Bayer pattern versions
3. Consistent processing parameters

This script is particularly useful for:
- Evaluating denoising performance on carefully controlled images
- Comparing model results against professionally processed references
- Validating model behavior on diverse but controlled image content
- Establishing baselines for model comparison

Args:
    --config: Path to training configuration file (.yaml)
    --load_path: Path to trained model checkpoint (optional)
    --debug_options output_valtest_images: Flag to save output images for visual inspection

Returns:
    Evaluation metrics saved to the model output directory as "manproc" test results.
    If debug_options includes output_valtest_images, also saves the denoised images.
"""

import configargparse  # For command-line argument parsing
import sys
import torch

# Add parent directory to Python path for imports


# Import necessary modules for denoiser training and dataset handling
from rawnind import train_denoiser_bayer2prgb
from rawnind.libs import abstract_trainer
from rawnind.libs import rawds_manproc  # Module for manually processed image handling
from rawnind.tests import rawtestlib  # Test helper classes

if __name__ == "__main__":
    # Configure test-only mode for the denoiser training class
    # This ensures the model is only evaluated, not trained
    preset_args = {"test_only": True, "init_step": None}

    # Allow for model loading from command line or use best checkpoint
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None  # Will use best_val checkpoint by default

    # Initialize the denoiser test harness for Bayer-to-RGB processing
    # This test class extends the training class but overrides dataloaders
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )

    # Check if manproc test results already exist to avoid redundant testing
    # Tests for three possible result key formats for backward compatibility
    if (
            "manproc_msssim_loss.None" in denoiserTraining.json_saver.results["best_val"]
            or "manproc_msssim_loss" in denoiserTraining.json_saver.results["best_val"]
            or "manproc_msssim_loss.gamma22"
            in denoiserTraining.json_saver.results["best_val"]
    ):
        print(f"Skipping test, manproc_msssim_loss is known")
        sys.exit(0)  # Exit successfully if results exist

    # Create test dataset using manually processed images
    # The ManuallyProcessedImageTestDataHandler loads pairs of clean target
    # images and corresponding Bayer pattern inputs
    dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
        # Configure for Bayer pattern input (raw sensor data)
        net_input_type="bayer"
    )

    # Create a batched iterator for the dataset
    # This converts the dataset into batches suitable for model processing
    dataloader = dataset.batched_iterator()

    # Run the denoising test with the manually processed dataset
    denoiserTraining.offline_custom_test(
        # Use the manually processed image dataloader
        dataloader=dataloader,
        # Set a descriptive test name for results identification
        test_name="manproc",
        # Save processed images for visual inspection
        save_individual_images=True
    )

