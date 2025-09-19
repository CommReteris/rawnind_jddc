"""Test pure denoising model on clean unpaired 'playraw' images, Bayer to profiled RGB.

This script evaluates a pure denoising model (without compression) on clean unpaired 
images from the 'playraw' dataset, processing from Bayer pattern to profiled RGB output.
It uses the same procedure as in training but in test-only mode.

The test loads a model checkpoint and runs inference on a dataset of clean images,
primarily testing the model's demosaicing capabilities since input images are already 
clean and don't require significant denoising.

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
    
    To fix this file:
    1. Add "from .libs import rawds_manproc"
    2. Replace the dataset initialization with rawds_manproc.ManuallyProcessedImageTestDataHandler
    3. Update the test_name parameter in offline_custom_test if necessary
"""

import sys

from .tests import rawtestlib

if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )
    if (
            "playraw_msssim_loss.None" in denoiserTraining.json_saver.results["best_val"]
            or "playraw_msssim_loss" in denoiserTraining.json_saver.results["best_val"]
    ):
        print(f"Skipping test, best_val is known")
        sys.exit(0)
    dataset = rawds_manproc.ManuallyProcessedImageTestDataHandler(
        net_input_type="prgb",
        test_descriptor_fpath="../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml",
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="playraw", save_individual_images=True
    )

