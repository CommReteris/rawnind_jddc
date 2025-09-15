"""
Find the best performing BM3D denoising models for specific test images.

This script evaluates multiple BM3D (Block Matching and 3D Filtering) denoising models 
with different configurations against specific test images to determine which 
configuration performs best for each image. It uses MS-SSIM (Multi-Scale Structural 
Similarity) as the quality metric, with higher values indicating better denoising results.

The script:
1. Scans a directory for BM3D model variations (different noise parameters)
2. Examines YAML result files containing performance metrics for each model
3. Compares MS-SSIM scores across different models for the same test images
4. Identifies and reports the best performing model for each test image

This is useful for:
- Selecting optimal denoising parameters for specific image types
- Understanding which noise levels work best for different real-world scenes
- Preparing optimal models for figure generation in academic papers
- Creating benchmarks for comparing with other denoising approaches

Usage:
    python find_best_bm3d_models_for_given_pictures.py
    
Output:
    Text report showing each test image with its best MS-SSIM score and the 
    corresponding model path.
"""

import os
import yaml

# -----------------------------------------------------------------------------
# Configuration: Directory paths and model selection
# -----------------------------------------------------------------------------

# Base directory containing all the BM3D model variants
base_dir = "/orb/benoit_phd/models/rawnind_denoise/"

# Select only sRGB BM3D models (filtering by name prefix)
# Each model represents a different noise parameter configuration
bm3d_models = [
    model for model in os.listdir(base_dir) if model.startswith("bm3d_sRGB_")
]

# -----------------------------------------------------------------------------
# Test images: Dictionary mapping test directories to specific image files
# -----------------------------------------------------------------------------

# Structure: {directory_name: [list_of_test_images]}
# - "manproc": Museum test images with various ISO values (noise levels)
# - "manproc_bostitch": Drawing test images with various exposure settings
pictures = {
    "manproc": [
        # Museum bird images with increasing ISO/noise (50, 16000, 64000, 204800)
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO50_capt0015.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO16000_capt0002.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO64000_capt0007.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
        "MuseeL-bluebirds-A7C_MuseeL-bluebirds-A7C_ISO204800_capt0010.arw.tif_aligned_to_ISO50_capt0015.arw.tif",
    ],
    "manproc_bostitch": [
        # Black and white drawing images with different exposure settings
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7692.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7688.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7689.CR2.tif_aligned_to_IMG_7692.CR2.tif",
        "LucieB_bw_drawing1_LucieB_bw_drawing1_IMG_7691.CR2.tif_aligned_to_IMG_7692.CR2.tif",
    ],
}

# -----------------------------------------------------------------------------
# Model evaluation: Track best results for each test image
# -----------------------------------------------------------------------------

# Dictionary to store best model results for each image
# Structure: {image_name: {"msssim_loss": value, "model": model_name, "dir": directory}}
best_results = {}

# Iterate through all models and evaluate each against the test images
for model in bm3d_models:
    for dir_name, pic_list in pictures.items():
        # Construct path to the model's results YAML file for this test directory
        yaml_path = os.path.join(base_dir, model, dir_name, "iter_1.yaml")
        
        # Skip if results file doesn't exist for this model/directory
        if not os.path.exists(yaml_path):
            continue

        # Load model results from YAML file
        # The file contains metrics for each test image processed by this model
        with open(yaml_path, "r") as file:
            data = yaml.safe_load(file)

        # Check each picture's results from this model
        for picture in pic_list:
            # Skip if this picture wasn't processed by this model
            if picture in data:
                # Extract MS-SSIM loss value (lower is better)
                # MS-SSIM loss = 1 - MS-SSIM, so minimizing loss means maximizing MS-SSIM
                msssim_loss = data[picture]["msssim_loss"]
                
                # Update best_results if:
                # 1. This is the first model we've seen for this picture, or
                # 2. This model has a lower MS-SSIM loss (better result) than previous best
                if (
                    picture not in best_results
                    or msssim_loss < best_results[picture]["msssim_loss"]
                ):
                    best_results[picture] = {
                        "msssim_loss": msssim_loss,  # Store the MS-SSIM loss value
                        "model": model,              # Store the model name
                        "dir": dir_name,             # Store the test directory
                    }

# -----------------------------------------------------------------------------
# Results formatting: Generate human-readable output
# -----------------------------------------------------------------------------

# Format results for each image into a list of strings
results = []
for picture, info in best_results.items():
    # Convert MS-SSIM loss to MS-SSIM score for clearer presentation
    # MS-SSIM ranges from 0 to 1, where 1 is perfect quality
    msssim = 1 - info["msssim_loss"]
    
    # Construct the model path for this result
    model_path = f"{info['model']}/{info['dir']}/iter_1/{picture}"
    
    # Format as "MS-SSIM: <score>\n<model_path>"
    results.append(f"MS-SSIM: {msssim:.6f}\n{model_path}")

# Join all results with double newlines for readability
output = "\n\n".join(results)

# Print the final report
print(output)
