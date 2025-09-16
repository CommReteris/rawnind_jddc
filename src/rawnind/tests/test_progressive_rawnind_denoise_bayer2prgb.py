"""Progressive denoising model evaluation on RawNIND Bayer pattern images.

This script evaluates a trained denoising model on RawNIND dataset images filtered
by MS-SSIM (Multi-Scale Structural Similarity) scores, allowing assessment of 
denoising performance across varying levels of image quality. The model converts 
Bayer pattern (raw sensor data) inputs to profiled RGB outputs.

The "progressive" testing approach evaluates models on image subsets with different
quality thresholds, providing insights into model performance on images with:
1. Easy cases (high MS-SSIM scores between clean/noisy pairs)
2. Difficult cases (low MS-SSIM scores between clean/noisy pairs)
3. Range of quality levels (using greater-than and less-than filters)

This script is particularly useful for:
- Analyzing denoising performance across quality levels
- Identifying model strengths and weaknesses on different image types
- Comparing progressive improvements in model versions

Args:
    --config: Path to training configuration file (.yaml)
    --load_path: Path to trained model checkpoint (optional)
    --debug_options output_valtest_images: Flag to save output images for visual inspection

Returns:
    Evaluation metrics for each MS-SSIM threshold, saved to the model output directory.
    If debug_options includes output_valtest_images, also saves the denoised images.
"""

import sys
import os

# Add parent directory to Python path for imports


from rawnind.libs import rawds
from rawnind.tests import rawtestlib

# Define MS-SSIM threshold values for filtering test images
# MS-SSIM (Multi-Scale Structural Similarity) measures image similarity
# between clean and noisy pairs, with 1.0 being identical
MS_SSSIM_VALUES = {
    # "le" (less than or equal) selects difficult cases with poor similarity
    "le": {0.85, 0.9, 0.97, 0.99},
    # "ge" (greater than or equal) selects progressively better quality pairs
    "ge": [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.00],
}

if __name__ == "__main__":
    # Configure test-only mode for the denoiser training class
    preset_args = {"test_only": True, "init_step": None}

    # Allow for model loading from command line or use None if not specified
    if "--load_path" not in sys.argv:
        preset_args["load_path"] = None

    # Initialize the denoiser test harness for Bayer-to-RGB processing
    # This uses the test subclass that overrides get_dataloaders but keeps other functionality
    denoiserTraining = rawtestlib.DenoiseTestCustomDataloaderBayerToProfiledRGB(
        preset_args=preset_args
    )

    # Iterate through both filtering operators (le/ge) and their threshold values
    for operator, msssim_values in MS_SSSIM_VALUES.items():
        for msssim_value in msssim_values:
            # Configure MS-SSIM filtering based on the current operator
            if operator == "le":
                # Filter for images with MS-SSIM <= threshold (more difficult cases)
                kwargs = {"max_msssim_score": msssim_value}
            elif operator == "ge":
                # Filter for images with MS-SSIM >= threshold (easier cases)
                kwargs = {"min_msssim_score": msssim_value}

            # Create a test dataloader with the current MS-SSIM filter
            dataloader = rawds.CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
                # Path to YAML file containing dataset image pairs
                content_fpaths=[
                    "../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml"
                ],  # denoiserTraining.noise_dataset_yamlfpaths,

                # Use the crop size from the training configuration
                crop_size=denoiserTraining.test_crop_size,

                # Use the test reservation setting from training configuration
                test_reserve=denoiserTraining.test_reserve,

                # Process only Bayer pattern inputs (not RGB)
                bayer_only=True,

                # Match gain based on input image
                match_gain="input",

                # Apply the MS-SSIM filtering parameters
                **kwargs,
            )

            # Commented code shows an alternative dataloader implementation
            # for external test data that could be used instead
            # dataset = (
            #     rawds_ext_paired_test.CleanProfiledRGBNoisyBayerImageCropsExtTestDataloader(
            #         content_fpaths=[
            #             os.path.join(
            #                 "..",
            #                 "..",
            #                 "datasets",
            #                 "ext_raw_denoise_test",
            #                 "ext_raw_denoise_test_masks_and_alignments.yaml",
            #             )
            #         ]
            #     )
            # )
            # dataloader = dataset.batched_iterator()

            # Run the denoising test with the current dataloader and settings
            denoiserTraining.offline_custom_test(
                # Use the MS-SSIM filtered dataloader
                dataloader=dataloader,

                # Create a descriptive test name that includes the MS-SSIM filter details
                test_name=f"progressive_test_msssim_{operator}_{msssim_value}",

                # Save processed images for visual inspection
                save_individual_images=True,
            )
