#!/usr/bin/env python3
"""
Mosaic Image Combiner for Research Figure Generation

This script combines two mosaic images (Bayer and X-Trans) vertically with a dotted
line separator, creating a visual comparison figure suitable for academic papers and
presentations. It handles image loading, width normalization, and proper alignment
between the two images with a visually appealing separator.

The script is designed specifically for the RawNIND dataset paper to illustrate
the different types of sensor color filter array patterns (Bayer vs. X-Trans) and
their respective image collections. The resulting combined image provides an at-a-glance
visual comparison of both filter array types.

Features:
- Handles transparent PNG inputs while preserving transparency
- Automatically crops images to match widths for perfect alignment
- Creates a professional dotted line separator between images
- Provides a clean command-line interface with sensible defaults
- Robust error handling for image loading and saving operations

Usage:
    python mk_combined_mosaic.py [--mosaic_bayer PATH] [--mosaic_xtrans PATH] [--output PATH]

Default paths point to the standard RawNIND dataset locations, but can be overridden
with command-line arguments for custom visualizations.
"""

from PIL import Image, ImageDraw


def load_image(path):
    """
    Loads an image from the given path and converts it to RGBA mode.
    
    This function opens an image file and explicitly converts it to RGBA mode to ensure
    transparency support throughout the image processing pipeline. It includes error
    handling to catch issues like missing files or corrupt image data.
    
    Parameters:
    - path (str): Path to the image file (typically a PNG file)
    
    Returns:
    - PIL.Image.Image: Loaded image in RGBA mode, or None if loading failed
    
    Note:
    - The conversion to RGBA ensures consistent handling of transparency
    - Error messages are printed to console if loading fails
    - Returns None instead of raising exceptions to allow graceful error handling
    """
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None


def crop_to_width(image, target_width):
    """
    Crops the width of the image to match the target width with center alignment.
    
    This function is used to normalize the widths of two images for proper vertical
    stacking. It crops the image symmetrically from both sides to maintain center
    alignment of the visual content.
    
    The function:
    - Only crops if the image is wider than the target width
    - Crops equally from both left and right sides
    - Preserves the original height and aspect ratio
    - Returns the original image unmodified if it's already narrower than target_width
    
    Parameters:
    - image (PIL.Image.Image): Image to crop
    - target_width (int): Desired width in pixels
    
    Returns:
    - PIL.Image.Image: Cropped image with width equal to target_width (or original
                      image if no cropping was needed)
    
    Example:
        If image is 1000px wide and target_width is 800px, the function will crop
        100px from the left side and 100px from the right side.
    """
    if image.width > target_width:
        # Calculate crop boundaries (center-aligned)
        left = (image.width - target_width) // 2
        right = left + target_width
        
        # Crop the image (preserving full height)
        return image.crop((left, 0, right, image.height))
    
    # Return original image if it's already narrower than target_width
    return image


def create_dotted_line(
    draw, start, end, line_width=2, dot_spacing=10, fill=(0, 0, 0, 255)
):
    """
    Draws a horizontal dotted line between start and end points.
    
    This function creates a visually appealing dotted line by alternating drawn segments
    with blank spaces. The implementation specifically:
    - Only works with horizontal lines (ignores y-coordinate differences)
    - Creates line segments of length dot_spacing
    - Leaves gaps of length dot_spacing between segments
    - Supports RGBA colors for the line (including transparency)
    
    The dotted line is primarily used as a visual separator between the Bayer and
    X-Trans mosaic images in the combined figure.
    
    Parameters:
    - draw (PIL.ImageDraw.Draw): ImageDraw object to draw on
    - start (tuple): (x, y) starting point coordinates
    - end (tuple): (x, y) ending point coordinates (only x is used)
    - line_width (int): Thickness of the dotted line in pixels
    - dot_spacing (int): Length of each dash and each gap in pixels
    - fill (tuple): RGBA color of the line, default is opaque black (0, 0, 0, 255)
    
    Returns:
    - None: The function modifies the draw object in-place
    
    Note:
    - For a more dense dotted pattern, decrease dot_spacing
    - For a more sparse dotted pattern, increase dot_spacing
    - The pattern starts with a drawn segment and ends with either a segment or gap
      depending on the overall line length
    """
    x_start, y = start
    x_end, _ = end
    for x in range(x_start, x_end, dot_spacing * 2):
        draw.line([(x, y), (x + dot_spacing, y)], fill=fill, width=line_width)


def combine_images_with_dotted_line(mosaic_bayer_path, mosaic_xtrans_path, output_path):
    """
    Combines two mosaic images stacked vertically with a horizontal dotted line separator.
    
    This function handles the main workflow for creating a combined image:
    1. Loads both input images (Bayer and X-Trans mosaics)
    2. Normalizes their widths by center-cropping the wider image
    3. Calculates the precise positioning to ensure perfect alignment
    4. Creates a new image with appropriate dimensions
    5. Adds a dotted line separator between the images
    6. Saves the result as a PNG file
    
    The function maintains transparency throughout the process if the input images
    have alpha channels, but saves the final result as RGB for maximum compatibility.
    
    Parameters:
    - mosaic_bayer_path (str): Path to the Bayer pattern mosaic image (placed on top)
    - mosaic_xtrans_path (str): Path to the X-Trans pattern mosaic image (placed on bottom)
    - output_path (str): Path to save the combined image (saved as PNG)
    
    Returns:
    - None: The function has no return value but saves the combined image to disk
           and prints status messages about the operation
           
    Notes:
    - Assumes both input images are PNG files with similar aspect ratios
    - If image loading fails, the function will print an error message and exit
    - Slight (1px) overlaps are used to ensure no white gaps appear at the dotted line
    """
    # Load images
    img_bayer = load_image(mosaic_bayer_path)
    img_xtrans = load_image(mosaic_xtrans_path)

    if img_bayer is None or img_xtrans is None:
        print("One or both images could not be loaded. Exiting.")
        return

    # Find the smaller width and crop both images to match
    target_width = min(img_bayer.width, img_xtrans.width)
    img_bayer = crop_to_width(img_bayer, target_width)
    img_xtrans = crop_to_width(img_xtrans, target_width)

    # Determine the width and height of the combined image
    dotted_line_height = 4  # Thickness of the dotted line
    combined_width = target_width
    combined_height = (
        img_bayer.height + img_xtrans.height + dotted_line_height - 1
    )  # Adjust for seamless alignment

    # Create a new transparent image
    combined_image = Image.new(
        "RGBA", (combined_width, combined_height), (255, 255, 255, 0)
    )

    # Paste Bayer image on the top
    combined_image.paste(img_bayer, (0, 0), img_bayer)

    # Draw the dotted line
    draw = ImageDraw.Draw(combined_image)
    line_y = (
        img_bayer.height + 1
    )  # Slight overlap with the bottom row of the Bayer image
    create_dotted_line(
        draw,
        (0, line_y),
        (combined_width, line_y),
        line_width=dotted_line_height,
        dot_spacing=15,
    )

    # Paste X-Trans image on the bottom
    xtrans_y_position = (
        img_bayer.height + dotted_line_height - 1
    )  # Align perfectly with the dotted line
    combined_image.paste(img_xtrans, (0, xtrans_y_position), img_xtrans)

    # Save the combined image
    try:
        final_image = combined_image.convert("RGB")
        final_image.save(output_path)
        print(f"Combined image saved to {output_path}.")
    except Exception as e:
        print(f"Failed to save combined image: {e}")


# -----------------------------------------------------------------------------
# Main Execution Block
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    # -----------------------------------------------------------------------------
    # Command-line Interface Setup
    # -----------------------------------------------------------------------------
    
    # Create parser with descriptive help text
    parser = argparse.ArgumentParser(
        description="Combine two mosaic images stacked vertically with a dotted line separator.",
        epilog="Example: python mk_combined_mosaic.py --output combined.png",
    )
    
    # Input paths for the mosaic images
    # Default paths point to the standard locations in the RawNIND dataset
    parser.add_argument(
        "--mosaic_bayer",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_Bayer.png",
        help="Path to the Bayer mosaic image (placed on top).",
    )
    parser.add_argument(
        "--mosaic_xtrans",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/mosaic_X-Trans.png",
        help="Path to the X-Trans mosaic image (placed on bottom).",
    )
    
    # Output path for the combined image
    parser.add_argument(
        "--output",
        type=str,
        default="/orb/benoit_phd/datasets/RawNIND/Thumbnails/combined_vertical_mosaic.png",
        help="Output path for the combined mosaic image (PNG format).",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # Execute Image Combination
    # -----------------------------------------------------------------------------
    
    # Call the main function with the provided paths
    # The function handles all image loading, processing, and saving
    combine_images_with_dotted_line(
        mosaic_bayer_path=args.mosaic_bayer,    # Top image (Bayer pattern)
        mosaic_xtrans_path=args.mosaic_xtrans,  # Bottom image (X-Trans pattern)
        output_path=args.output,                # Path to save the combined result
    )
    
    # Note: The function will print success/error messages to console
