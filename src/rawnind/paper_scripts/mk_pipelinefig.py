"""
Pipeline Visualization Generator for Research Papers.

This script creates publication-ready figures that visualize the image processing pipeline
used in the Natural Image Noise Dataset (NIND) research. It generates diagrams showing the 
progression of image processing stages from raw sensor data to final output.

Key features:
1. Creates closeup views of specific image regions to highlight important details
2. Adds numbered labels to identify each pipeline stage
3. Arranges images in customizable grid layouts
4. Handles color space conversion (Rec.2020 to sRGB) for proper display
5. Generates two figure variants optimized for different publication formats:
   - Thesis format: 3x3 grid showing all pipeline stages
   - Journal paper format: 4x2 grid of selected stages

The script uses the bluebirds test image, processing it through 9 stages:
0. Raw sensor data
1. Bayer pattern with white balance points
2. Demosaiced linear Rec.2020 image
3. Denoised image
4. Lens correction, perspective correction, and cropping
5. Exposure adjustment
6. Color calibration
7. Diffuse/sharpen adjustment
8. Color balance and filmic tone mapping

Usage:
    python mk_pipelinefig.py

Output:
    Two JPEG files showing the pipeline stages arranged in different grid layouts
    for thesis and journal paper publications.
"""

import os
import hashlib
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import imageio.v3 as iio


# Rec. 2020 to Rec. 709 transformation matrix for color space conversion
REC2020_TO_REC709_MATRIX = np.array(
    [
        [1.6605, -0.5876, -0.0728],
        [-0.1246, 1.1329, -0.0083],
        [-0.0182, -0.1006, 1.1187],
    ]
)


def generate_unique_filename(filepath, prefix="closeup_"):
    """
    Generate a unique filename based on the file path using a checksum.

    Args:
        filepath (str): The original file path.
        prefix (str): A prefix for the new filename.

    Returns:
        str: A unique filename with the checksum.
    """
    # Generate a SHA-256 hash of the file path
    checksum = hashlib.sha256(filepath.encode()).hexdigest()[
        :8
    ]  # Use the first 8 characters
    filename = os.path.basename(filepath)
    unique_filename = f"{prefix}{checksum}_{filename}"
    return unique_filename


def linear_to_srgb(linear_rgb):
    """Apply gamma correction to convert linear RGB to sRGB."""
    # Ensure values are in the range [0, 1] (clip invalid values)
    linear_rgb = np.clip(linear_rgb, 0, 1)

    # Apply gamma correction
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * np.power(linear_rgb, 1 / 2.4) - 0.055,
    )
    return np.clip(srgb, 0, 1)


def convert_rec2020_to_srgb(image):
    """
    Convert an image from Linear Rec. 2020 RGB to sRGB.

    Args:
        image (PIL.Image.Image): Input image in Linear Rec. 2020 RGB.

    Returns:
        PIL.Image.Image: Output image in sRGB.
    """
    # Convert PIL image to NumPy array (normalize to [0, 1])
    img_array = np.array(image).astype(np.float32) / 255.0

    # Handle RGBA images by removing the alpha channel
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    # Apply Rec. 2020 to Rec. 709 transformation
    rec709_rgb = np.dot(img_array, REC2020_TO_REC709_MATRIX.T)

    # Apply gamma correction to get sRGB
    srgb = linear_to_srgb(rec709_rgb)

    # Convert back to uint8 and PIL.Image
    srgb_image = (srgb * 255).astype(np.uint8)
    return srgb_image


def create_placeholder_with_closeups(image_path, save_path):
    """
    Create a composite image with main image and closeup regions.
    
    This function takes an input image, creates a larger canvas with the original 
    image, and adds closeup views of specific regions highlighted with colored 
    dashed rectangles. The layout adapts to the image orientation:
    - For landscape images: closeups appear below the main image
    - For portrait images: closeups appear to the right of the main image
    
    The function handles various image formats (PNG, TIFF) and bit depths
    (8-bit, 16-bit, float), normalizing them for consistent display.
    
    Args:
        image_path (str): Path to the input image file
        save_path (str): Path where the composite image will be saved
        
    Returns:
        PIL.Image.Image: The final composite image with main view and closeups
    """
    # No additional whitespace around the image
    WHITESPACE = 0

    # ===== IMAGE LOADING AND PREPROCESSING =====
    # Load image with imageio (supports various formats including HDR)
    img_data = iio.imread(image_path)  # Reads as a NumPy array
    
    # Rotate specific images for better composition (bluebirds in portrait orientation)
    if (
        (image_path.endswith(".tif"))
        and "bluebirds" in image_path
        and ("lin_rec2020" in image_path or "faux_Bayer" in image_path)
    ):
        img_data = np.rot90(img_data, 1)
    
    # Normalize image data based on data type
    if img_data.dtype in [np.float32, np.float16]:
        # For float images: clip to [0,1] range and scale to [0,255]
        img_data = np.clip(img_data, 0.0, 1.0)
        img_data = img_data * 255
    elif img_data.dtype == np.uint16:
        # For 16-bit images: scale from [0,65535] to [0,255]
        img_data = img_data / 65535.0 * 255
    
    # Convert to 8-bit unsigned integer for PIL
    img = img_data.astype(np.uint8)
    
    # Create PIL Image from numpy array
    img = Image.fromarray(img)

    # Get image dimensions
    width, height = img.size

    # Determine image orientation (landscape vs portrait)
    # This affects how closeups will be arranged
    is_landscape = width > height

    if is_landscape:
        new_height = int(height * 1.5)  # Increase height by 50% for bottom closeups
        placeholder_width = width
        placeholder = Image.new("RGB", (placeholder_width, new_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = height // 2  # Height for all close-ups
        x_offset = 0
        paste_location = lambda x, y: (x, height)  # paste on the bottom
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (w / h))
    else:  # portrait
        new_width = int(width * 1.5)  # Increase width by 50% for right side closeups
        placeholder_height = height
        placeholder = Image.new("RGB", (new_width, placeholder_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = width // 2  # Width for all close-ups
        y_offset = 0
        paste_location = lambda x, y: (width, y)  # paste to the right
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (h / w))

    if is_landscape:
        closeup_regions = [
            # (2728, 2014, 2831, 2117),
            (2792, 1777, 2911, 1895),
            (1075, 1262, 1758, 1945),
            (1745, 23, 1976, 254),
        ]
    else:
        # Define close-up regions and properties
        closeup_regions = [
            (2755, 1873, 3172, 2232),  # Rectangle 1
            (3318, 4586, 3440, 4670),  # Rectangle 2
            (2035, 2677, 2265, 3017),  # Rectangle 3
        ]
        # if (
        #     image_path.endswith(".tif")
        #     and "bluebirds" in image_path
        #     and "lin_rec2020" in image_path
        # ):
        #     closeup_regions = [
        #         (3001, 2382, 3407, 2734),  # Rectangle 1
        #         (3687, 5067, 3809, 5151),  # Rectangle 2
        #         (2341, 3160, 2560, 3482),  # Rectangle 3
        #     ]
        # elif (
        #     (image_path.endswith(".png"))
        #     and "bluebirds" in image_path
        #     and "faux_Bayer" in image_path
        # ):
        if (
            "/0_" in image_path
            or "/1_" in image_path
            or "/2_" in image_path
            or "/3_" in image_path
        ):
            closeup_regions = [
                (3025, 2383, 3378, 2702),  # Rectangle 1
                (3681, 5061, 3807, 5143),  # Rectangle 2
                (2350, 3158, 2562, 3479),  # Rectangle 3 / claw
            ]
    colors = ["cyan", "magenta", "yellow"]  # Colors for each close-up

    def draw_dashed_rectangle(draw_obj, x1, y1, x2, y2, color, dash_length, width):
        """Draw a dashed rectangle by manually creating dashed lines."""
        # Top edge
        for i in range(x1, x2, dash_length * 2):
            draw_obj.line(
                [(i, y1), (min(i + dash_length, x2), y1)], fill=color, width=width
            )
        # Left edge
        for i in range(y1, y2, dash_length * 2):
            draw_obj.line(
                [(x1, i), (x1, min(i + dash_length, y2))], fill=color, width=width
            )
        # Bottom edge
        for i in range(x1, x2, dash_length * 2):
            draw_obj.line(
                [(i, y2), (min(i + dash_length, x2), y2)], fill=color, width=width
            )
        # Right edge
        for i in range(y1, y2, dash_length * 2):
            draw_obj.line(
                [(x2, i), (x2, min(i + dash_length, y2))], fill=color, width=width
            )

    # Make a copy of the main image to ensure close-ups are independent
    main_image_with_rectangles = img.copy()
    main_image_draw = ImageDraw.Draw(main_image_with_rectangles)

    # Draw rectangles on the main image and prepare close-ups
    for idx, (x1, y1, x2, y2) in enumerate(closeup_regions):
        # Draw rectangles on the main image
        draw_dashed_rectangle(
            main_image_draw, x1, y1, x2, y2, colors[idx], dash_length=20, width=25
        )

        # Prepare close-ups
        cropped = img.crop((x1, y1, x2, y2))

        if is_landscape:
            resized_width = resized_dimension_calculation(cropped.width, cropped.height)
            resized = cropped.resize((resized_width, closeup_dimension), Image.LANCZOS)
        else:
            resized_height = resized_dimension_calculation(
                cropped.width, cropped.height
            )
            resized = cropped.resize((closeup_dimension, resized_height), Image.LANCZOS)

        # Create a new image for the resized close-up to draw dashed lines
        if is_landscape:
            resized_with_border = Image.new(
                "RGB", (resized_width, closeup_dimension), "white"
            )
        else:
            resized_with_border = Image.new(
                "RGB", (closeup_dimension, resized_height), "white"
            )
        resized_with_border.paste(resized, (0, 0))

        # Draw a dashed rectangle on the resized close-up
        closeup_draw = ImageDraw.Draw(resized_with_border)
        if is_landscape:
            draw_dashed_rectangle(
                closeup_draw,
                0,
                0,
                resized_width - 1,
                closeup_dimension - 1,
                colors[idx],
                dash_length=40,
                width=50,
            )
        else:
            draw_dashed_rectangle(
                closeup_draw,
                0,
                0,
                closeup_dimension - 1,
                resized_height - 1,
                colors[idx],
                dash_length=40,
                width=50,
            )

        # Add the resized close-up with dashed borders to the placeholder
        if is_landscape:
            placeholder.paste(resized_with_border, paste_location(x_offset, 0))
            x_offset += resized_width  # next x offset
        else:
            placeholder.paste(resized_with_border, paste_location(0, y_offset))
            y_offset += resized_height  # next y offset

    # Update placeholder size to include whitespace on the side/bottom
    if is_landscape:
        total_width = max(x_offset, width) + WHITESPACE
        placeholder_with_whitespace = Image.new(
            "RGB", (total_width, new_height), "white"
        )
    else:
        total_height = max(y_offset, height) + WHITESPACE
        placeholder_with_whitespace = Image.new(
            "RGB", (new_width, total_height), "white"
        )

    placeholder_with_whitespace.paste(placeholder, (0, 0))

    # Paste the updated main image (with rectangles) back into the placeholder
    placeholder_with_whitespace.paste(main_image_with_rectangles, (0, 0))

    # Save the placeholder with close-ups
    return placeholder_with_whitespace


def create_numbered_image(image_path, index):
    """Creates a processed image with the number on top-left"""
    output_dir = "tmp/processed_images"
    os.makedirs(output_dir, exist_ok=True)

    unique_filename = generate_unique_filename(filepath=image_path, prefix="numbered_")
    placeholder_path = os.path.join(output_dir, unique_filename)
    img_with_closeups = create_placeholder_with_closeups(image_path, placeholder_path)

    # Add number overlay
    draw = ImageDraw.Draw(img_with_closeups)
    # Use a font size that scales with the image dimensions
    font_size = int(min(img_with_closeups.width, img_with_closeups.height) * 0.05)
    font = ImageFont.truetype(
        "DejaVuSans.ttf", size=font_size
    )  # Use a better font and size
    draw.text((10, 10), " " + str(index), font=font, fill="white")  # position and color
    return img_with_closeups


def create_grid_figure(image_paths, output_path, grid_shape, start_index=0):
    """
    Creates and saves a grid figure containing multiple pipeline stage images.
    
    This function creates a publication-ready grid of images that represent different
    stages of the image processing pipeline. Each image is:
    1. Loaded and preprocessed to ensure consistent format
    2. Numbered according to its position in the pipeline
    3. Resized to ensure uniform dimensions
    4. Arranged in a grid according to the specified shape
    5. Resized to a target width suitable for publication
    
    The function supports different grid layouts for different publication formats
    (e.g., 3x3 for thesis, 4x2 for journal papers) and can start numbering from
    any index (useful when skipping initial pipeline stages).
    
    Args:
        image_paths (list): List of paths to images representing pipeline stages
        output_path (str): Path where the final grid figure will be saved
        grid_shape (tuple): (rows, columns) defining the grid layout
        start_index (int): Starting index for numbering the images (default: 0)
        
    Returns:
        None: The function saves the output to the specified path
    """
    # ===== DETERMINE IMAGE DIMENSIONS =====
    # Calculate maximum height across all images to ensure uniform sizing
    max_height = 0
    for path in image_paths:
        # Load image using imageio (supports various formats)
        img_data = iio.imread(path)  # Reads as a NumPy array
        
        # Rotate specific images for consistent orientation
        if (
            (path.endswith(".tif"))
            and "bluebirds" in path
            and ("lin_rec2020" in path or "faux_Bayer" in path)
        ):
            img_data = np.rot90(img_data, 1)
            
        # Normalize image data based on its data type
        if img_data.dtype in [np.float32, np.float16]:  # Handle float images
            img_data = np.clip(img_data, 0.0, 1.0)  # Clip to valid range [0, 1]
            img_data = img_data * 255  # Scale to 8-bit range
        elif img_data.dtype == np.uint16:  # Handle 16-bit integer images
            img_data = img_data / 65535.0 * 255  # Scale to 8-bit range
            
        # Convert to 8-bit for PIL and determine maximum height
        img = img_data.astype(np.uint8)
        img = Image.fromarray(img)
        max_height = max(max_height, img.height)

    # ===== CREATE NUMBERED IMAGES =====
    # Process each image to add a number indicating its pipeline stage
    images = []
    for i, path in enumerate(image_paths):
        # Add a numbered label to each image (number = position in pipeline)
        # Start numbering from start_index (allows skipping initial stages for journal version)
        images.append(create_numbered_image(path, i + start_index))

    # ===== STANDARDIZE IMAGE SIZES =====
    # Find maximum width to ensure all images have the same dimensions in the grid
    max_width = max(img.width for img in images)

    # Resize all images to the same dimensions for uniform grid layout
    resized_images = []
    for img in images:
        # Maintain aspect ratio by using the same height and width for all images
        resized = img.resize((max_width, max_height))
        resized_images.append(resized)

    # ===== CREATE GRID LAYOUT =====
    # Calculate total grid dimensions based on the specified shape
    rows, cols = grid_shape
    grid_width = max_width * cols
    grid_height = max_height * rows
    
    # Create a new black background image as the grid canvas
    grid_image = Image.new("RGB", (grid_width, grid_height), "black")

    # Place each image in its position in the grid
    for i, img in enumerate(resized_images):
        # Calculate row and column position based on image index
        row = i // cols  # Integer division for row number
        col = i % cols   # Modulo for column number
        
        # Calculate pixel coordinates for placement
        x = col * max_width
        y = row * max_height
        
        # Paste the image at the calculated position
        grid_image.paste(img, (x, y))

    # ===== RESIZE FOR PUBLICATION =====
    # Resize the final grid to a standard width for publication
    target_width = 1500  # Standard width for publication figures
    
    # Calculate scaling factor and new height to maintain aspect ratio
    wpercent = target_width / float(grid_image.size[0])
    target_height = int((float(grid_image.size[1]) * float(wpercent)))
    
    # Resize using LANCZOS resampling for high quality
    resized_grid_image = grid_image.resize((target_width, target_height), Image.LANCZOS)

    # Save the final image as JPEG (standard format for publications)
    resized_grid_image.save(output_path, "JPEG")
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    # ===== DEFINE IMAGE PROCESSING PIPELINE STAGES =====
    # These images represent progressive stages in the image processing pipeline
    # Each file is named with a prefix indicating its position in the pipeline
    image_paths = [
        # Stage 0: Raw sensor data directly from the camera
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/0_nothing_ISO16000_capt0002.png",
        
        # Stage 1: Bayer pattern with white balance points applied
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/1_bw_points_ISO16000_capt0002_01.png",
        
        # Stage 2: Demosaiced image in linear Rec.2020 color space
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/2_demosaic_linrec2020_d65_ISO16000_capt0002_02.png",
        
        # Stage 3: Denoising applied to remove sensor noise
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/3_denoise_ISO16000_capt0002.arw.png",
        
        # Stage 4: Lens and perspective correction, with cropping
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/4_lenscorrection_perspective_cropISO16000_capt0002.arw_01.png",
        
        # Stage 5: Exposure adjustment to optimize brightness
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/5_exposure_ISO16000_capt0002.arw.png",
        
        # Stage 6: Color calibration for accurate color reproduction
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/6_color_calibration_ISO16000_capt0002.arw_02.png",
        
        # Stage 7: Diffuse or sharpen operations for detail enhancement
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/7_diffuseorsharpen_ISO16000_capt0002.arw_03.png",
        
        # Stage 8: Final color balance and filmic tone mapping
        "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/src/pipeline/darktable_exported/8_colorbalance_filmicISO16000_capt0002.arw_05.png",
    ]

    # ===== GENERATE FIGURES FOR DIFFERENT PUBLICATION FORMATS =====
    
    # Create the thesis version: 3x3 grid showing all stages (0-8)
    # This comprehensive layout is suitable for a thesis where space is less constrained
    output_path_thesis = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/pipeline_stages_figure_thesis.jpg"
    create_grid_figure(image_paths, output_path_thesis, (3, 3))

    # Create the journal paper version: 4x2 grid showing stages 1-8 (skipping stage 0)
    # Journal papers have stricter space limitations, so we omit the raw sensor data stage
    # and use a more compact layout
    output_path_jddc = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/pipeline_stages_figure_jddc.jpg"
    create_grid_figure(image_paths[1:], output_path_jddc, (2, 4), start_index=1)
