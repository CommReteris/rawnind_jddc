"""
Configuration generator for BM3D denoising model training.

This script automatically generates args.yaml configuration files for training 
BM3D denoising models across multiple color spaces and noise levels. It creates
a structured directory hierarchy and populates each directory with the appropriate
configuration file.

Purpose:
1. Create configuration files for a grid search across color spaces and noise levels
2. Maintain consistent configuration settings across all experiments
3. Simplify the process of setting up multiple related training runs

The script generates configurations for:
- Color spaces: linRGB and sRGB
- Noise levels: ranging from 5 to 99

Each configuration includes settings for:
- Model architecture (BM3D)
- Training parameters (batch sizes, learning rates, etc.)
- Dataset paths and processing
- Evaluation metrics
- Testing and validation settings

Usage:
    python create_bm3d_argsyaml.py

Output:
    Creates args.yaml files in directories structured as:
    /path/to/base_dir/bm3d_{color_space}_{noise_level}/args.yaml

Original reference: https://chatgpt.com/c/66f42139-5b74-800f-bbdf-b7a697e4f393
"""

import os

# Base directory where all model configurations will be stored
base_dir = "/orb/benoit_phd/models/rawnind_denoise"

# Configuration matrix parameters
# Two color spaces to test: linear RGB and standard RGB
color_spaces = ["linRGB", "sRGB"]

# Noise levels to test ranging from mild (5) to severe (99)
# Higher values represent higher noise levels for the BM3D algorithm
noise_levels = [
    "5",   # Very mild noise
    "10",  # Mild noise
    "20",  # Moderate noise
    "30", 
    "40", 
    "50",  # Medium noise
    "60", 
    "70", 
    "80",  # Strong noise
    "90",  # Very strong noise
    "93", 
    "95", 
    "97", 
    "99",  # Extreme noise
]

# YAML content template for args.yaml configuration files
# This template contains all parameters needed for training BM3D denoising models
yaml_content = """
# Model architecture and processing settings
arbitrary_proc_method: null  # No additional processing method
arch: bm3d                   # Using BM3D architecture
bayer_only: true             # Process only Bayer pattern images
in_channels: 3               # Number of input channels
funit: {noise_level}         # Noise level parameter

# Training dataset configuration
clean_dataset_yamlfpaths:    # Paths to clean reference images
- ../../datasets/extraraw/trougnouf/crops_metadata.yaml
- ../../datasets/extraraw/raw-pixls/crops_metadata.yaml
noise_dataset_yamlfpaths:    # Path to noisy images
- /scratch/brummer/39509156/RawNIND/RawNIND_masks_and_alignments.yaml
data_pairing: x_y            # Pairing strategy for clean/noisy images

# Training parameters
batch_size_clean: 1          # Batch size for clean images
batch_size_noisy: 4          # Batch size for noisy images
crop_size: 256               # Size of image crops for training
num_crops_per_image: 4       # Number of crops per training image
init_lr: 0.0003              # Initial learning rate
lr_multiplier: 0.85          # Learning rate decay multiplier
tot_steps: 6000000           # Total training steps
patience: 100000             # Patience for early stopping
warmup_nsteps: 0             # No warmup steps

# Checkpointing and model loading
continue_training_from_last_model_if_exists: true  # Resume training if possible
init_step: 1                 # Initial step count
fallback_load_path: null     # No fallback model path
load_path: ../../models/rawnind_denoise/bm3d_{color_space}_{noise_level}/saved_models/iter_1.pt
reset_lr: false              # Don't reset learning rate when loading
reset_optimizer_on_fallback_load_path: false  # Don't reset optimizer for fallback

# Loss function and evaluation metrics
loss: msssim_loss            # Using MS-SSIM loss function
match_gain: output           # Match gain on output images
metrics:                     # Metrics for evaluation
- msssim_loss
- mse
transfer_function: None      # No transfer function for training
transfer_function_valtest: None  # No transfer function for validation/testing

# Validation and testing settings
val_crop_size: 1024          # Size of validation crops
val_interval: 15000          # Validate every 15000 steps
test_crop_size: 1024         # Size of test crops
test_interval: 1500000       # Test every 1500000 steps

# Reserved test images (not used for training)
test_reserve:
- 7D-2
- Vaxt-i-trad
- Pen-pile
- MuseeL-vases-A7C
- D60-1
- MuseeL-Saint-Pierre-C500D
- TitusToys
- boardgames_top
- Laura_Lemons_platformer
- MuseeL-bluebirds-A7C
- LucieB_bw_drawing1
- LucieB_bw_drawing2
- LucieB_board
- LucieB_painted_wallpaper
- LucieB_painted_plants
- LucieB_groceries

# Output and experiment naming
comment: bm3d_{color_space}_{noise_level}  # Comment for logging
expname: bm3d_{color_space}_{noise_level}  # Experiment name
save_dpath: ../../models/rawnind_denoise/bm3d_{color_space}_{noise_level}  # Save directory

# Miscellaneous
config: null                 # No additional config file
device: null                 # Auto-select device (CPU/GPU)
debug_options:               # Debug settings
- minimize_threads           # Use fewer threads for debugging
"""

# Create args.yaml files in each directory by iterating through the configuration matrix
for color_space in color_spaces:
    for noise_level in noise_levels:
        # Define the target directory path for this configuration
        dir_path = f"{base_dir}/bm3d_{color_space}_{noise_level}"

        # Define the full file path for the args.yaml file
        file_path = os.path.join(dir_path, "args.yaml")

        # Replace placeholders in the template with specific values
        yaml_data = yaml_content.format(
            color_space=color_space, noise_level=noise_level
        )

        # Create the args.yaml file (will create directories if they don't exist)
        with open(file_path, "w") as file:
            file.write(yaml_data)

        # Log the creation for tracking
        print(f"Created: {file_path}")
