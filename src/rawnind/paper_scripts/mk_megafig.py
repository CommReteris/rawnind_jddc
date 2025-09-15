"""
Figure generation for academic papers on image processing techniques.

This script generates composite figures for academic publications, combining
multiple processed images with their metrics into publication-ready layouts.
It creates figures that showcase denoising and compression results with closeup
regions to highlight differences between methods.

The script:
1. Reads YAML configuration files that specify images and their metrics
2. Creates combined figures with original images and closeup regions
3. Generates separate figures for denoising and compression results
4. Formats results for different publication types (JDDC paper, thesis, wiki)
5. Handles color space conversion from Rec.2020 to sRGB for proper display

The generated figures include:
- Full-size images with highlighted regions of interest
- Closeup views of those regions to show fine details
- Metrics (MS-SSIM, bpp) displayed below each image
- Proper reference formatting for different publication contexts

Usage:
    python mk_megafig.py

Output:
    PDF figures saved to the specified output directory.
"""

import yaml
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import math
import numpy as np
from imageio import imwrite
import hashlib
import imageio.v3 as iio  # Modern imageio API for loading images

# REC2020_ICC_FPATH = (
#     "/orb/benoit_phd/src/ext/image_color_processing/data/lin_rec2020_from_dt.icc"
# )

# Reference mapping for different publication types
# Maps method identifiers to their proper citation format in each context
LITERATURE = {
    "jddc": {  # Journal paper format with numeric citations
        "[BM3D]": "[3]",          # BM3D method citation
        "[NIND]": "[10]",         # NIND method citation
        "[OURS]": None,           # Our method (no special citation needed)
        "[JPEGXL]": "[17]",       # JPEG XL citation
        "[COMPDENOISE]": "[24]",  # Compression+Denoising citation
        "[MANYPRIORS]": "[27]",   # ManyPriors method citation
    },
    "thesis": {  # Thesis format with chapter references
        "[BM3D]": None,           # External method (no special citation)
        "[NIND]": None,           # External method (no special citation)
        "[OURS]": "(Ch. 5)",      # Our method in Chapter 5
        "[JPEGXL]": None,         # External method (no special citation)
        "[COMPDENOISE]": "(Ch. 4)", # Our Compression+Denoising in Chapter 4
        "[MANYPRIORS]": "(Ch. 3)", # Our ManyPriors method in Chapter 3
    },
    "wiki": {  # Wiki format with no citations
        "[BM3D]": None,
        "[NIND]": None,
        "[OURS]": None,
        "[JPEGXL]": None,
        "[COMPDENOISE]": None,
        "[MANYPRIORS]": None,
    },
}

# Configuration files that define the images and metrics to include in figures
YAML_FILES = [
    "/orb/benoit_phd/src/rawnind/plot_cfg/Picture2_32.yaml",  # First test image config
    "/orb/benoit_phd/src/rawnind/plot_cfg/Picture1_32.yaml",  # Second test image config
]

# Color space conversion matrix for proper image display
# Rec. 2020 (wide gamut) to Rec. 709 (standard RGB) transformation matrix
REC2020_TO_REC709_MATRIX = np.array(
    [
        [1.6605, -0.5876, -0.0728],  # Red channel conversion
        [-0.1246, 1.1329, -0.0083],  # Green channel conversion
        [-0.0182, -0.1006, 1.1187],  # Blue channel conversion
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

    # Apply Rec. 2020 to Rec. 709 transformation
    rec709_rgb = np.dot(img_array, REC2020_TO_REC709_MATRIX.T)

    # Apply gamma correction to get sRGB
    srgb = linear_to_srgb(rec709_rgb)

    # Convert back to uint8 and PIL.Image
    srgb_image = (srgb * 255).astype(np.uint8)
    return srgb_image


# Setup output directory for processed images
output_dir = "tmp/processed_images"
os.makedirs(output_dir, exist_ok=True)


def create_placeholder_with_closeups(image_path, save_path):
    """
    Create a composite image with the main image and closeup regions.
    
    This function takes an input image, creates a larger canvas, and adds:
    1. The original image with highlighted regions of interest
    2. Closeup views of those regions with matching colored borders
    
    The layout adapts based on image orientation:
    - For landscape images: closeups appear below the original image
    - For portrait images: closeups appear to the right of the original image
    
    The function handles various image formats and color spaces, including
    linear Rec.2020 which is converted to sRGB for display purposes.
    
    Args:
        image_path: Path to the input image file
        save_path: Path where the composite image will be saved
        
    Returns:
        None (saves the composite image to save_path)
    """
    WHITESPACE = 50  # Extra white space around the composite image

    # --- IMAGE LOADING AND PREPROCESSING ---
    # Load image with imageio (supports various formats including HDR)
    img_data = iio.imread(image_path)  # Reads as a NumPy array
    
    # Special case: rotate bluebirds image for better composition
    if (
        (image_path.endswith(".tif"))
        and "bluebirds" in image_path
        and ("lin_rec2020" in image_path or "faux_Bayer" in image_path)
    ):
        img_data = np.rot90(img_data, 1)
        
    # Normalize image data based on dtype
    if img_data.dtype in [np.float32, np.float16]:  # Handle float images
        img_data = np.clip(img_data, 0.0, 1.0)  # Clip to valid range [0, 1]
        img_data = img_data * 255  # Normalize to [0, 255]
    elif img_data.dtype == np.uint16:  # Handle 16-bit integer images
        img_data = img_data / 65535.0 * 255  # Normalize to [0, 255]
        
    # Convert color space from linear Rec.2020 to sRGB for display
    img = convert_rec2020_to_srgb(img_data)
    
    # Convert to PIL Image for further processing
    img = Image.fromarray(img)

    # --- LAYOUT DETERMINATION AND CANVAS CREATION ---
    width, height = img.size
    is_landscape = width > height  # Determine orientation for layout

    # Create appropriate canvas based on orientation
    if is_landscape:
        # For landscape: closeups go below the main image
        new_height = int(height * 1.5)  # Increase height by 50% for closeups
        placeholder_width = width
        placeholder = Image.new("RGB", (placeholder_width, new_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = height // 2  # Height for all close-ups
        x_offset = 0  # Starting X position for closeups
        # Function to determine where to paste closeups
        paste_location = lambda x, y: (x, height)  # paste on the bottom
        # Function to calculate the appropriate resized dimension
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (w / h))
    else:  # portrait
        # For portrait: closeups go to the right of the main image
        new_width = int(width * 1.5)  # Increase width by 50% for closeups
        placeholder_height = height
        placeholder = Image.new("RGB", (new_width, placeholder_height), "white")
        placeholder.paste(img, (0, 0))
        closeup_dimension = width // 2  # Width for all close-ups
        y_offset = 0  # Starting Y position for closeups
        # Function to determine where to paste closeups
        paste_location = lambda x, y: (width, y)  # paste to the right
        # Function to calculate the appropriate resized dimension
        resized_dimension_calculation = lambda w, h: int(closeup_dimension * (h / w))

    # --- SELECT REGIONS OF INTEREST BASED ON IMAGE ---
    # Define regions to highlight and create closeups from
    # Coordinates format: (x1, y1, x2, y2) = (left, top, right, bottom)
    if is_landscape:
        closeup_regions = [
            (2731, 1954, 2890, 2114),  # Region 1
            (1075, 1262, 1758, 1945),  # Region 2
            (1745, 23, 1976, 254),     # Region 3
        ]
    else:
        # Default regions for portrait images
        closeup_regions = [
            (2755, 1873, 3172, 2232),  # Region 1
            (3318, 4586, 3440, 4670),  # Region 2
            (2035, 2677, 2265, 3017),  # Region 3
        ]
        # Special case for bluebirds images in Rec.2020 format
        if (
            image_path.endswith(".tif")
            and "bluebirds" in image_path
            and "lin_rec2020" in image_path
        ):
            closeup_regions = [
                (3001, 2382, 3407, 2734),  # Region 1
                (3687, 5067, 3809, 5151),  # Region 2
                (2341, 3160, 2560, 3482),  # Region 3
            ]
        # Special case for bluebirds images in PNG format
        elif (
            (image_path.endswith(".png"))
            and "bluebirds" in image_path
            and "faux_Bayer" in image_path
        ):
            closeup_regions = [
                (3001, 2402, 3417, 2754),  # Region 1
                (3687, 5092, 3809, 5176),  # Region 2
                (2341, 3180, 2560, 3512),  # Region 3 / claw
            ]
            
    # Colors for the dashed rectangles - each region gets a different color
    colors = ["cyan", "magenta", "yellow"]

    # --- HELPER FUNCTION FOR DRAWING DASHED RECTANGLES ---
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

    # --- CREATE MAIN IMAGE WITH HIGHLIGHTED REGIONS ---
    # Make a copy of the main image for adding region markers
    main_image_with_rectangles = img.copy()
    main_image_draw = ImageDraw.Draw(main_image_with_rectangles)

    # --- PROCESS EACH CLOSEUP REGION ---
    # Draw rectangles on the main image and create closeups
    for idx, (x1, y1, x2, y2) in enumerate(closeup_regions):
        # Draw colored dashed rectangle on the main image to highlight region
        draw_dashed_rectangle(
            main_image_draw, x1, y1, x2, y2, colors[idx], dash_length=20, width=25
        )

        # Create closeup by cropping the region from the original image
        cropped = img.crop((x1, y1, x2, y2))

        # Resize closeup to fit in the layout while maintaining aspect ratio
        if is_landscape:
            resized_width = resized_dimension_calculation(cropped.width, cropped.height)
            resized = cropped.resize((resized_width, closeup_dimension), Image.LANCZOS)
        else:
            resized_height = resized_dimension_calculation(
                cropped.width, cropped.height
            )
            resized = cropped.resize((closeup_dimension, resized_height), Image.LANCZOS)

        # Create a white background for the closeup with matching border color
        if is_landscape:
            resized_with_border = Image.new(
                "RGB", (resized_width, closeup_dimension), "white"
            )
        else:
            resized_with_border = Image.new(
                "RGB", (closeup_dimension, resized_height), "white"
            )
        # Place the closeup on the white background
        resized_with_border.paste(resized, (0, 0))

        # Draw a matching colored dashed border around the closeup
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

        # Add the closeup to the main placeholder image
        if is_landscape:
            placeholder.paste(resized_with_border, paste_location(x_offset, 0))
            x_offset += resized_width  # Update x position for next closeup
        else:
            placeholder.paste(resized_with_border, paste_location(0, y_offset))
            y_offset += resized_height  # Update y position for next closeup

    # --- FINALIZE AND SAVE THE COMPOSITE IMAGE ---
    # Add whitespace to ensure the image has sufficient margins
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

    # Copy the placeholder to the final image with whitespace
    placeholder_with_whitespace.paste(placeholder, (0, 0))

    # Paste the main image with rectangle markers back into the placeholder
    placeholder_with_whitespace.paste(main_image_with_rectangles, (0, 0))

    # Save the final composite image
    placeholder_with_whitespace.save(save_path)


def plot_section(
    fig,
    gs,
    section_data,
    section_name,
    row_offset,
    columns,
    show_only_first=False,
    show_only_last=False,
    include_bpp=True,
    yaml_file: str = "",
):
    """
    Plot a section of images with their metrics in a grid layout.
    
    This function creates a section of the figure containing multiple images from
    the same category (Input, Denoising, or Compression), arranged in a grid with
    method names in the first column and images in subsequent columns.
    
    Each image is displayed with its metrics (MS-SSIM, bpp) and optional parameters
    as captions. Caption display can be controlled to show only on certain rows.
    Dashed lines can be added between rows to visually separate different methods.
    
    Args:
        fig: Matplotlib figure object where the section will be plotted
        gs: Matplotlib GridSpec object defining the grid layout
        section_data: List of dictionaries containing image data (paths, metrics, etc.)
        section_name: Name of the section ("Input", "Denoising", or "Compression")
        row_offset: Starting row index in the grid for this section
        columns: Number of columns in the grid layout
        show_only_first: If True, only show captions on the first row
        show_only_last: If True, only show captions on the last row
        include_bpp: If True, include bits-per-pixel in the caption (for compression)
        yaml_file: Path to the YAML config file (used for conditional logic)
        
    Returns:
        Number of rows used by this section
    """
    # Font sizes for text elements
    font_size = 16  # Caption font size
    font_size_method = 19  # Method name font size

    # Calculate how many rows we need for this section
    total_items = len(section_data)
    rows = math.ceil(total_items / (columns - 1))  # Subtract 1 for the method column

    # Process each image in the section data
    for i, image_data in enumerate(section_data):
        # Calculate row and column position in the grid
        row, col = divmod(i, columns - 1)
        row += row_offset  # Apply row offset for this section

        # Determine whether to show captions for this row
        if show_only_first:
            # Only show captions on the first row of the section
            show_caption = row == row_offset
        elif show_only_last:
            # Only show captions on the last row of the section
            show_caption = row == row_offset + rows - 1
        else:
            # Show captions on all rows
            show_caption = True

        # --- FIRST COLUMN: METHOD NAME ---
        if col == 0:
            # Create subplot for the method name
            ax = fig.add_subplot(gs[row, 0])
            
            # Add the method name rotated 90 degrees
            ax.text(
                x=0.5,
                y=0.5,
                s=image_data.get("method", ""),
                fontsize=font_size_method,
                ha="center",  # Horizontal alignment: center
                va="center",  # Vertical alignment: center
                rotation=90,  # Rotate text 90 degrees
                weight=(
                    "bold" if "Input" in section_name else "normal"
                ),  # Make 'Input' method names bold
            )
            ax.axis("off")  # Hide the axis

        # --- PREPARE IMAGE CAPTION WITH METRICS ---
        # Format MS-SSIM value with 3 decimal places
        caption = f"MS-SSIM: {float(image_data.get('msssim', 0)):.3f}"
        
        # Add bits-per-pixel if requested (for compression results)
        if include_bpp:
            caption += f", {image_data['bpp']} bpp"
            
        # Add any additional parameters if available
        if image_data.get("parameters"):
            # Add newline if caption would be too long
            if len(image_data["parameters"]) + len(caption) > 32:
                caption += "\n"
            else:
                caption += ", "
            caption += f"{image_data['parameters']}"

        # --- IMAGE DISPLAY ---
        # Create subplot for the image, skipping the method column
        ax = fig.add_subplot(gs[row, col + 1])
        
        # Load and display the image
        img = Image.open(image_data["image_path"])
        ax.imshow(img)
        
        # Add the caption if needed for this row
        if show_caption:
            ax.set_title(label=caption, fontsize=font_size, pad=10)
            
        ax.axis("off")  # Hide the axis

        # --- ADD SEPARATOR LINE BETWEEN METHODS ---
        # Determine if we should add a dashed line after this image
        # Only add after the last image in a row (rightmost column)
        show_dashed_line = col == columns - 2
        
        # Special cases where we don't want a dashed line
        if (
            # Don't add line after input sections that aren't "Developed"
            (section_name == "Input" and "Developed" not in image_data["method"])
            # Don't add line after specific rows based on YAML file
            or (row == 5 and "Picture1" in yaml_file)
            or (row == 4 and "Picture2" in yaml_file)
        ):
            show_dashed_line = False

        # Add the dashed line if needed
        if show_dashed_line:
            # Create a subplot that spans the entire row
            fig.add_subplot(gs[row, :])
            line_ax = plt.gca()
            
            # Draw a horizontal dashed line
            line_ax.plot(
                [0, 1],  # X coordinates (full width)
                [0, 0],  # Y coordinates (bottom of cell)
                transform=line_ax.transAxes,  # Use axis coordinates
                color="gray",
                linestyle="--",
                linewidth=0.5,
            )
            line_ax.axis("off")  # Hide the axis
            
    return rows  # Return the number of rows used


def create_figure(section_data, file_suffix_1: str, file_suffix_2: str, yaml_file: str):
    """
    Create and save a publication-ready figure comparing image processing methods.
    
    This function creates a complete figure that combines input images with either
    denoising or compression results, arranged in a grid layout with metrics.
    It handles the entire figure generation pipeline:
    1. Loading configuration from YAML file
    2. Processing images to create composites with closeups
    3. Formatting metrics and method names for the target publication type
    4. Creating the multi-row, multi-column layout with proper spacing
    5. Saving the final figure as a PDF
    
    The figure layout adapts based on the YAML file and processing type:
    - For compression figures: Only shows "Developed" input images
    - For denoising figures: Shows all input images
    - Caption display is controlled based on section
    - Figure dimensions are optimized based on content
    
    Args:
        section_data: List of dictionaries containing image data for the main section
        file_suffix_1: Type of processing ("denoising" or "compression")
        file_suffix_2: Publication type ("jddc", "thesis", or "wiki")
        yaml_file: Path to YAML configuration file defining the figure content
        
    Returns:
        None (saves the figure to a PDF file)
    """
    # --- LOAD CONFIGURATION DATA ---
    with open(yaml_file, "r") as file:
        data = yaml.safe_load(file)

    # --- DEFINE FIGURE LAYOUT PARAMETERS ---
    # Layout parameters based on the specific figure being created
    columns = 5  # Number of columns in the grid (method + 4 images)
    fig_width = 23  # Width in inches
    fig_height_per_row = 5  # Height per row in inches
    
    # Adjust vertical spacing based on figure type
    hspace = (
        0.10  # Tighter spacing for denoising figures with Picture1
        if "denoising" in file_suffix_1.lower() and "Picture1" in yaml_file
        else 0.11  # Standard spacing for other figures
    )

    # --- PROCESS IMAGES AND PREPARE DATA ---
    # Dictionary to store processed data for each section
    processed_data = {"Input": [], "Denoising": [], "Compression": []}
    
    # Process images from YAML data
    for section, items in data.items():
        for method_item in items:
            # Get method name and replace acronyms with proper citations
            method = method_item["method"]
            for method_acro, reference in LITERATURE[file_suffix_2].items():
                method = method.replace(method_acro, reference or "")

            # Process each image for this method
            for img in method_item["images"]:
                # Skip input images in compression section unless they're "Developed"
                if (
                    section == "Compression"
                    and "Input" in method
                    and "Developed" not in method
                ):
                    continue

                # Create a composite image with closeups
                src_fpath = img["src_fpath"]
                unique_filename = generate_unique_filename(filepath=src_fpath)
                placeholder_path = os.path.join(output_dir, unique_filename)

                # Generate the composite image with highlighted regions and closeups
                create_placeholder_with_closeups(
                    image_path=src_fpath, save_path=placeholder_path
                )

                # Add image data to the appropriate section
                processed_data[section].append(
                    {
                        "method": method,
                        "image_path": placeholder_path,
                        "caption": img.get("caption"),
                        "bpp": img.get("bpp", None),  # Bits per pixel (for compression)
                        "msssim": img.get("msssim", None),  # MS-SSIM metric
                        "parameters": img.get("parameters", None),  # Additional parameters
                    }
                )

    # --- CALCULATE FIGURE DIMENSIONS ---
    # Calculate how many rows we need based on content
    total_rows = math.ceil(len(processed_data["Input"]) / (columns - 1)) + math.ceil(
        len(section_data) / (columns - 1)
    )
    
    # Special case for compression figures - they need fewer rows
    if "compression" in file_suffix_1.lower():
        if total_rows > 0:
            total_rows -= 2
            
    # Ensure at least one row
    total_rows = max(1, total_rows)

    # --- CREATE THE FIGURE ---
    # Create the figure with calculated dimensions
    fig = plt.figure(
        figsize=(fig_width, total_rows * fig_height_per_row),
        tight_layout=True,
    )
    
    # Create the grid layout with special width for method column
    gs = plt.GridSpec(
        nrows=total_rows,
        ncols=columns,
        figure=fig,
        width_ratios=[0.1] + [1] * (columns - 1),  # Narrower first column for method names
        hspace=hspace,  # Horizontal spacing between cells
    )

    # --- PLOT INPUT SECTION ---
    row_offset = 0  # Track the current row position
    
    # Handle input section differently based on figure type
    if "compression" in file_suffix_1.lower():
        # For compression: only show "Developed" input images
        input_data = [
            row for row in processed_data["Input"] if "Developed" in row["method"]
        ]
        # Plot input section and update row offset
        row_offset += plot_section(
            fig=fig,
            gs=gs,
            section_data=input_data,
            section_name="Input",
            row_offset=row_offset,
            columns=columns,
            show_only_last=True,  # Only show captions on last row
            include_bpp=True,     # Include bits-per-pixel for compression
            yaml_file=yaml_file,
        )
    else:
        # For denoising: show all input images
        row_offset += plot_section(
            fig=fig,
            gs=gs,
            section_data=processed_data["Input"],
            section_name="Input",
            row_offset=row_offset,
            columns=columns,
            show_only_first=True,  # Only show captions on first row
            include_bpp=False,     # Don't include bits-per-pixel for denoising
            yaml_file=yaml_file,
        )

    # --- PLOT MAIN SECTION (DENOISING OR COMPRESSION) ---
    plot_section(
        fig=fig,
        gs=gs,
        section_data=section_data,
        section_name=file_suffix_1,
        row_offset=row_offset,
        columns=columns,
        include_bpp="compression" in file_suffix_1.lower(),  # Include bpp only for compression
        yaml_file=yaml_file,
    )

    # --- SAVE THE FIGURE ---
    # Create output path for the PDF
    output_path = f"/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/{os.path.basename(yaml_file)}_{file_suffix_1}_{file_suffix_2}.pdf"
    
    # Save figure with tight bounding box to eliminate excess whitespace
    plt.savefig(fname=output_path, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
    
    print(f"Figure saved to {output_path}")  # Confirmation message


# ===== MAIN EXECUTION BLOCK =====
# Generate all figure variants for each publication type and configuration

# Outer loop: iterate over different publication types (journal paper, thesis, wiki)
# Each publication type requires different citation formats
for a_paper, method_reference in LITERATURE.items():
    print(f"Generating figures for publication type: {a_paper}")
    
    # Middle loop: iterate over each test image configuration file
    for yaml_file in YAML_FILES:
        print(f"Processing configuration: {yaml_file}")
        
        # Load the YAML configuration defining images and methods to include
        with open(yaml_file, "r") as file:
            data = yaml.safe_load(file)

        # Container for processed image data organized by section
        processed_data = {"Input": [], "Denoising": [], "Compression": []}

        # Process all images defined in the configuration
        for section, items in data.items():
            # Each section (Input, Denoising, Compression) contains multiple methods
            for method_item in items:
                # Get method name and apply appropriate citation format for this publication
                method = method_item["method"]
                for method_acro, reference in method_reference.items():
                    method = method.replace(method_acro, reference or "")

                # Process each image for this method
                for img in method_item["images"]:
                    # Skip input images in compression section unless they're "Developed"
                    if (
                        section == "Compression"
                        and "Input" in method
                        and "Developed" not in method
                    ):
                        continue

                    # Generate composite image with closeups
                    src_fpath = img["src_fpath"]
                    unique_filename = generate_unique_filename(filepath=src_fpath)
                    placeholder_path = os.path.join(output_dir, unique_filename)
                    
                    # Create the composite image with highlighted regions
                    create_placeholder_with_closeups(
                        image_path=src_fpath, save_path=placeholder_path
                    )

                    # Store the processed image data for figure generation
                    processed_data[section].append(
                        {
                            "method": method,
                            "image_path": placeholder_path,
                            "caption": img.get("caption"),
                            "bpp": img.get("bpp", None),        # Compression rate
                            "msssim": img.get("msssim", None),  # Quality metric
                            "parameters": img.get("parameters", None),  # Model params
                        }
                    )
        
        # Generate two separate figures from the processed data:
        # 1. Denoising figure - comparing denoising methods
        print(f"Creating denoising figure for {os.path.basename(yaml_file)}")
        create_figure(
            section_data=processed_data["Denoising"],
            file_suffix_1="denoising",          # Figure type
            file_suffix_2=a_paper,              # Publication type
            yaml_file=yaml_file,                # Source configuration
        )
        
        # 2. Compression figure - comparing compression methods
        print(f"Creating compression figure for {os.path.basename(yaml_file)}")
        create_figure(
            section_data=processed_data["Compression"],
            file_suffix_1="compression",        # Figure type
            file_suffix_2=a_paper,              # Publication type
            yaml_file=yaml_file,                # Source configuration
        )
        
    print(f"Completed all figures for {a_paper}")

print("All figures generated successfully.")
