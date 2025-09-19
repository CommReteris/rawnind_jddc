"""Test denoise+compress model on clean unpaired 'playraw' images, Bayer to profiled RGB.

This script evaluates a denoise+compress (DC) model on clean unpaired images
from the 'playraw' dataset, processing from Bayer pattern to profiled RGB output.
It uses the same procedure as in training but in test-only mode.

The test loads a model checkpoint and runs inference on a dataset of clean images,
evaluating the model's performance on demosaicing and compression without the
denoising component being significantly tested (as input images are already clean).

Required arguments:
    --config <path to training config file>.yaml: Configuration file for the model

Optional arguments:
    --load_path <path>: Path to specific model checkpoint (if not using best_val)
    --debug_options output_valtest_images: Flag to save output images for inspection

Note:
    This file has a code issue: it references 'rawds_cleancleantest' module which 
    appears to be missing or renamed in the current codebase. A potential fix would 
    be to use 'rawds_manproc.ManuallyProcessedImageTestDataHandler' instead, 
    as seen in the test_manproc_playraw_* files.
"""

import sys

# Import the manually processed image test handler
import .libs.rawds_manproc
# Import the test helper classes
from .tests import rawtestlib

if __name__ == "__main__":
    # Configure test-only mode for the denoise+compress model
    preset_args = {"test_only": True, "init_step": None}
    
    # Allow for model loading from command line or use best checkpoint
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None  # Will use best_val by default
    
    # Initialize the denoise+compress test harness for Bayer-to-RGB processing
    # This uses a lightweight test subclass that maintains core functionality
    denoiserTraining = rawtestlib.DCTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )
    
    # Check if test results already exist to avoid redundant testing
    # This improves efficiency for batch testing scenarios
    if any(
            akey in denoiserTraining.json_saver.results["best_val"]
            for akey in ["playraw_combined.None", "playraw_combined"]
    ):
        print(f"Skipping test, best_val is known")
        sys.exit(0)  # Exit successfully if results exist
    
    # Create test dataset using manually processed clean images
    # These are high-quality "playraw" images that test the model's
    # performance on ideal inputs without significant noise
    dataset = .libs.rawds_manproc.ManuallyProcessedImageTestDataHandler(
        # Configure for profiled RGB processing (demosaiced images)
        net_input_type="prgb",
        # Path to the test dataset descriptor
        test_descriptor_fpath="../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml",
    )
    
    # Create a batched iterator for the dataset
    dataloader = dataset.batched_iterator()

    # Run the denoise+compress test with the playraw dataset
    denoiserTraining.offline_custom_test(
        # Use the manually processed clean image dataloader
        dataloader=dataloader, 
        # Set a descriptive test name
        test_name="playraw", 
        # Save processed images for visual inspection
        save_individual_images=True
    )

