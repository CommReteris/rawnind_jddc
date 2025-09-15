"""
Quality Score Distribution Analysis for Natural Image Noise Dataset (RawNIND).

This script analyzes the distribution of MS-SSIM (Multi-Scale Structural Similarity) 
scores across a selected subset of test images from the RawNIND dataset. MS-SSIM is 
a perceptual image quality metric that ranges from 0 to 1, where higher values 
indicate better image quality/similarity.

The analysis helps understand:
1. The quality distribution of the test dataset
2. The range of noise levels present in the test images
3. Whether the test set provides adequate coverage of different quality levels

The script:
- Loads image metadata from the RawNIND masks and alignments YAML file
- Filters for specific test image sets (standard benchmark images)
- Extracts MS-SSIM scores that measure similarity between noisy and clean images
- Generates a histogram showing the distribution of quality scores

Usage:
    python get_RawNIND_test_quality_distribution.py

Output:
    Displays a histogram showing the distribution of MS-SSIM scores across test images.
    Higher scores indicate images with less perceptible noise.
"""

import yaml
import matplotlib.pyplot as plt
import numpy as np

# Load the RawNIND dataset metadata from YAML file
file_path = "../../datasets/RawNIND/RawNIND_masks_and_alignments.yaml"
with open(file_path, "r") as file:
    data = yaml.safe_load(file)

# Select standard test image sets used for evaluation
# These specific image sets are consistently used across the project for:
# 1. Evaluating model performance (test reserves in training configs)
# 2. Generating figures for paper publications
# 3. Benchmarking different denoising and compression approaches
# They represent a diverse range of scenes, lighting conditions, and camera models
selected_images = [
    item
    for item in data
    if item.get("image_set")
    in [
        "7D-2",                       # Canon 7D outdoor nature scene
        "Vaxt-i-trad",                # Plant/garden scene with fine details
        "Pen-pile",                   # Close-up of pens (fine detail test)
        "MuseeL-vases-A7C",           # Museum artifacts (Sony A7C)
        "D60-1",                      # Nikon D60 indoor scene
        "MuseeL-Saint-Pierre-C500D",  # Museum statue (Canon 500D)
        "TitusToys",                  # Colorful toys (texture/color test)
        "boardgames_top",             # Board games (text and color patterns)
        "Laura_Lemons_platformer",    # Still life with fruit
        "MuseeL-bluebirds-A7C",       # Museum display (primary figure image)
    ]
]

# Extract MS-SSIM scores from selected test images
# Note: There are two possible metrics in the data:
# 1. best_alignment_loss: Lower values indicate better alignment (commented out)
# 2. rgb_msssim_score: Higher values indicate better image quality (0-1 range)
# 
# We use rgb_msssim_score as it directly measures image quality through the
# Multi-Scale Structural Similarity Index, which correlates well with human
# perception of image quality and is used throughout the project
loss_values = [item.get("rgb_msssim_score", 0.0) for item in selected_images]

# Sort the values in ascending order
# This sorting is needed for:
# 1. Properly visualizing the distribution in the histogram
# 2. Preparing data for the optional cumulative distribution plot (currently commented out)
# 3. Making it easier to identify min/max/median values for analysis
loss_values.sort()


# Create a histogram visualization of MS-SSIM score distribution
# This helps identify:
# - The range of quality levels in the test set
# - Whether quality scores are evenly distributed or clustered
# - If there are any gaps in quality coverage
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure with reasonable size

# Create histogram with 50 bins and normalized density
# - bins=50: Provides good granularity without being too noisy
# - density=True: Normalizes the histogram to represent a probability density
#   (area under the histogram sums to 1)
ax.hist(loss_values, bins=50, density=True, color='skyblue', alpha=0.7)

# Add descriptive labels and title
ax.set_xlabel("MS-SSIM score (higher = better quality)")
ax.set_ylabel("Probability density")
ax.set_title("Distribution of Image Quality Scores in RawNIND Test Set")

# Add grid for easier reading of values
ax.grid(True, linestyle='--', alpha=0.7)

# Uncomment the following lines to display min, median, and max values as vertical lines
# ax.axvline(min(loss_values), color='r', linestyle='--', label=f'Min: {min(loss_values):.3f}')
# ax.axvline(np.median(loss_values), color='g', linestyle='--', label=f'Median: {np.median(loss_values):.3f}')
# ax.axvline(max(loss_values), color='b', linestyle='--', label=f'Max: {max(loss_values):.3f}')
# ax.legend()

# Alternative visualization: Cumulative Distribution Function (CDF)
# This shows the probability that a randomly selected image has a quality score
# less than or equal to a given value.
# 
# A steep slope in the CDF indicates many images clustered around that quality level,
# while a flat section indicates few images in that quality range.
# 
# Uncomment to generate this alternative visualization:
# fig, ax = plt.subplots(figsize=(10, 6))
# counts, bin_edges = np.histogram(loss_values, bins=50, density=True)
# cdf = np.cumsum(counts)
# ax.plot(bin_edges[1:], cdf / cdf[-1], label="Cumulative Distribution", color='darkblue', linewidth=2)
# ax.set_xlabel("MS-SSIM Score")
# ax.set_ylabel("Cumulative Probability")
# ax.set_title("Cumulative Distribution of MS-SSIM Scores in RawNIND Test Set")
# ax.grid(True, linestyle='--', alpha=0.7)
# ax.legend()

# Display the visualization
# This will show an interactive matplotlib window with the plot
plt.tight_layout()  # Adjust layout for better spacing
plt.show()
