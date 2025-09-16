"""Test denoise+compress model on clean unpaired 'playraw' images, profiled RGB to profiled RGB.

This script evaluates a denoise+compress (DC) model on clean unpaired images
from the 'playraw' dataset, processing from profiled RGB to profiled RGB output.
It uses the same procedure as in training but in test-only mode.

The test loads a model checkpoint and runs inference on a dataset of clean images,
evaluating the model's performance on compression without the denoising component
being significantly tested (as input images are already clean).

Required arguments:
    --config <path to training config file>.yaml: Configuration file for the model

Optional arguments:
    --load_path <path>: Path to specific model checkpoint (if not using best_val)
    --debug_options output_valtest_images: Flag to save output images for inspection

Note:
    This file has several code issues:
    1. It has syntax errors in the dataset creation section (lines 28-35)
    2. It's missing the import for 'rawds_manproc'
    3. It imports 'rawtestlib' directly instead of from 'rawnind.libs'
    
    A corrected version would:
    - Import "from rawnind.libs import rawds_manproc"
    - Fix the dataset initialization syntax
    - Follow patterns from test_manproc_playraw_dc_prgb2prgb.py
"""

import sys

import rawnind.libs.rawds_manproc
from rawnind.tests import rawtestlib

if __name__ == "__main__":
    preset_args = {"test_only": True, "init_step": None}
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None
    denoiserTraining = rawtestlib.DCTestCustomDataloaderProfiledRGBToProfiledRGB(
        preset_args=preset_args
    )
    if any(
            akey in denoiserTraining.json_saver.results["best_val"]
            for akey in [
                "playraw_combined.None",
                "playraw_combined",
                "playraw_combined.bayer",
            ]
    ):
        print(f"Skipping test, best_val is known")
        sys.exit(0)
    dataset = (rawnind.libs.rawds_manproc.ManuallyProcessedImageTestDataHandler(
        net_input_type="bayer",
        test_descriptor_fpath="../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml",
    )
    )
    dataloader = dataset.batched_iterator()

    denoiserTraining.offline_custom_test(
        dataloader=dataloader, test_name="playraw", save_individual_images=True
    )

