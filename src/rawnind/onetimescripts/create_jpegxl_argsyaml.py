"""Configuration generator for JPEG XL image compression model training.

This script automatically generates args.yaml configuration files for training 
neural compression models using JPEG XL across multiple color spaces and compression
levels. It creates a structured directory hierarchy and populates each directory 
with the appropriate configuration file.

Purpose:
1. Create configuration files for a grid search across color spaces and compression levels
2. Maintain consistent configuration settings across all experiments
3. Simplify the process of setting up multiple related training runs

The script generates configurations for:
- Color spaces: linRGB and sRGB
- Compression levels: 1, 5, 10, 20, 30, 40, 50, 60 (higher = higher quality)

Each configuration includes settings for:
- Model architecture (JPEG XL)
- Training parameters (batch sizes, learning rates, etc.)
- Dataset paths and processing
- Evaluation metrics
- Testing and validation settings

Usage:
    python create_jpegxl_argsyaml.py

Output:
    Creates args.yaml files in directories structured as:
    /path/to/base_dir/JPEGXL_{color_space}_{compression_level}/args.yaml

Original reference: https://chatgpt.com/c/66f42139-5b74-800f-bbdf-b7a697e4f393
"""

import os

# Base directory where all model configurations will be stored
base_dir = "/orb/benoit_phd/models/rawnind_dc"

# Configuration matrix parameters
# Two color spaces to test: linear RGB and standard RGB
color_spaces = ["linRGB", "sRGB"]

# Compression quality levels to test
# Lower values = more compression, higher values = higher quality
compression_levels = [
    "1",  # Very aggressive compression
    "5",  # High compression
    "10", # Medium-high compression
    "20", # Medium compression
    "30", # Medium-low compression
    "40", # Low compression
    "50", # Very low compression
    "60", # Minimal compression
]

# YAML content template for args.yaml configuration files
# This template contains all parameters needed for training JPEG XL compression models
yaml_content = """
# Processing method settings
arbitrary_proc_method: {arbitrary_proc_method}
arch: JPEGXL                   # Using JPEG XL architecture

# Training dataset configuration
batch_size_clean: 1            # Batch size for clean images
batch_size_noisy: 4            # Batch size for noisy images
bayer_only: true               # Process only Bayer pattern images
bitEstimator_lr_multiplier: 10.0
bitstream_out_channels: 320    # Number of channels in bitstream representation
clean_dataset_yamlfpaths:      # Paths to clean reference images
- ../../datasets/extraraw/trougnouf/crops_metadata.yaml
- ../../datasets/extraraw/raw-pixls/crops_metadata.yaml
comment: JPEGXL_{color_space}_{compression_level}  # Comment for logging
config: config/train_dc_{proc_method}.yaml         # Base configuration file
continue_training_from_last_model_if_exists: true  # Resume training if possible

# Image processing parameters
crop_size: 256                 # Size of image crops for training
data_pairing: x_y              # Pairing strategy for clean/noisy images
debug_options:
- minimize_threads             # Use fewer threads for debugging
device: null                   # Auto-select device (CPU/GPU)
expname: JPEGXL_{proc_method}_{compression_level}  # Experiment name
fallback_load_path: JPEGXL_{proc_method}_{compression_level}
funit: {compression_level}     # Compression level parameter

# Model architecture parameters
hidden_out_channels: 192       # Number of hidden channels
in_channels: 3                 # Number of input channels (RGB)
init_lr: 0.0001                # Initial learning rate
init_step: 1                   # Initial step count
load_path: ../../models/rawnind_dc/JPEGXL_{color_space}_{compression_level}/saved_models/iter_1.pt

# Loss function and evaluation metrics
loss: msssim_loss              # Using MS-SSIM loss function
lr_multiplier: 0.85            # Learning rate decay multiplier
match_gain: input              # Match gain on input images
metrics:                       # Metrics for evaluation
- msssim_loss
- mse

# Dataset configuration
noise_dataset_yamlfpaths:      # Path to noisy images
- /scratch/brummer/39509156/RawNIND/RawNIND_masks_and_alignments.yaml
num_crops_per_image: 2         # Number of crops per training image
patience: 100000               # Patience for early stopping
reset_lr: false                # Don't reset learning rate when loading
reset_optimizer_on_fallback_load_path: false  # Don't reset optimizer for fallback
save_dpath: ../../models/rawnind_dc/JPEGXL_{color_space}_{compression_level}  # Save directory

# Validation and testing settings
test_crop_size: 1024           # Size of test crops
test_interval: 1500000         # Test every 1500000 steps
test_reserve:                  # Reserved test images (not used for training)
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
tot_steps: 6000000             # Total training steps
train_lambda: 4.0              # Lambda parameter for rate-distortion tradeoff
transfer_function: None        # No transfer function for training
transfer_function_valtest: None  # No transfer function for validation/testing
val_crop_size: 1024            # Size of validation crops
val_interval: 15000            # Validate every 15000 steps
warmup_nsteps: 100000          # Warmup steps for learning rate scheduling
"""

# Create args.yaml files in each directory by iterating through the configuration matrix
for color_space in color_spaces:
    for compression_level in compression_levels:
        # Define the target directory path for this configuration
        dir_path = f"{base_dir}/JPEGXL_{color_space}_{compression_level}"
        os.makedirs(dir_path, exist_ok=True)

        # Define the full file path for the args.yaml file
        file_path = os.path.join(dir_path, "args.yaml")

        # Determine processing method based on color space
        # - proc2proc for sRGB
        # - prgb2prgb for linRGB
        proc_method = "proc2proc" if color_space == "sRGB" else "prgb2prgb"
        
        # Replace placeholders in the template with specific values
        yaml_data = yaml_content.format(
            color_space=color_space,
            compression_level=compression_level,
            proc_method=proc_method,
            # Use opencv processing for sRGB, null for linRGB
            arbitrary_proc_method="opencv" if color_space == "sRGB" else "null",
        )

        # Create the args.yaml file (will create directories if they don't exist)
        with open(file_path, "w") as file:
            file.write(yaml_data)

        # Log the creation for tracking
        print(f"Created: {file_path}")
