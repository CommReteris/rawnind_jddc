"""
MS-SSIM Score Distribution Histogram Generator for Natural Image Noise Dataset.

This script analyzes the distribution of MS-SSIM (Multi-Scale Structural Similarity) scores
across the Natural Image Noise Dataset (NIND) and generates a publication-ready histogram
figure in TikZ/LaTeX format. The MS-SSIM scores measure the similarity between noisy images
and their corresponding ground truth references, with values ranging from 0 to 1 (higher
values indicate greater similarity).

The script:
1. Loads image quality metrics from the NIND YAML metadata file
2. Filters entries to include only noisy images (where file path ≠ ground truth path)
3. Extracts MS-SSIM scores from the filtered entries
4. Computes a histogram with customizable bin parameters
5. Generates a TikZ (LaTeX) figure of the histogram with a logarithmic y-axis
6. Saves the figure as a standalone .tex file that can be compiled for publication

The resulting histogram provides a visual representation of the distribution of noise levels
in the dataset, which is valuable for:
- Understanding the noise characteristics of the dataset
- Ensuring adequate coverage across different noise levels for training
- Presenting dataset statistics in academic publications

Usage:
    python plot_dataset_msssim_distributionv2.py

Output:
    A LaTeX file containing TikZ code for the histogram figure
"""

import yaml
import numpy as np
import os


def read_yaml(file_path):
    """
    Read and parse a YAML file containing dataset metadata.
    
    This function loads the NIND dataset's metadata YAML file, which contains
    information about image pairs, their alignments, and quality metrics
    including MS-SSIM scores that measure similarity between noisy and
    reference images.
    
    Args:
        file_path (str): Path to the YAML file containing dataset metadata
        
    Returns:
        list: A list of dictionaries, where each dictionary contains metadata
              for one image pair, including file paths and quality metrics
              
    Raises:
        FileNotFoundError: If the specified YAML file doesn't exist
        yaml.YAMLError: If the file contains invalid YAML syntax
    """
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return data


def filter_scores(data):
    """
    Filter dataset entries and extract MS-SSIM scores for noisy images.
    
    This function filters the dataset metadata to include only entries where:
    1. The file path ('f_fpath') differs from the ground truth path ('gt_fpath'),
       which indicates a noisy image rather than a ground truth reference
    2. The entry contains a valid MS-SSIM score ('rgb_msssim_score')
    
    Args:
        data (list): List of metadata dictionaries from the YAML file, where
                     each dictionary contains information about an image pair
                     
    Returns:
        list: A list of float values representing MS-SSIM scores for all valid
              noisy images in the dataset
              
    Note:
        MS-SSIM scores range from 0 to 1, where higher values indicate greater
        similarity to the ground truth (less noise/distortion)
    """
    scores = []
    for entry in data:
        if entry.get("f_fpath") != entry.get("gt_fpath"):
            # This is a noisy image (not a ground truth reference)
            score = entry.get("rgb_msssim_score")
            if score is not None:
                # Only include entries with valid MS-SSIM scores
                scores.append(score)
    return scores


def compute_histogram(scores, bin_start=0.40, bin_end=1.00, bin_width=0.05):
    """
    Compute histogram of MS-SSIM scores with customizable binning parameters.
    
    This function creates a histogram of MS-SSIM scores, focused on the typical
    range of values found in the NIND dataset. The default range (0.40-1.00)
    captures the most relevant distribution while excluding extremely noisy
    images with very low scores.
    
    Args:
        scores (list): List of MS-SSIM score values to analyze
        bin_start (float, optional): Lower bound for histogram bins. Defaults to 0.40.
        bin_end (float, optional): Upper bound for histogram bins. Defaults to 1.00.
        bin_width (float, optional): Width of each histogram bin. Defaults to 0.05.
        
    Returns:
        tuple: A tuple containing three elements:
            - counts (numpy.ndarray): The count of scores in each bin
            - bin_centers (numpy.ndarray): The center value of each bin (for plotting)
            - bins (numpy.ndarray): The bin edges including the rightmost edge
            
    Note:
        The default parameters (0.40-1.00 with 0.05 width) create 12 bins,
        which provides a good balance between detail and clarity for publication.
    """
    # Create an array of bin edges from start to end (inclusive)
    bins = np.arange(bin_start, bin_end + bin_width, bin_width)
    
    # Compute histogram counts for each bin
    counts, bin_edges = np.histogram(scores, bins=bins)
    
    # Calculate bin centers for plotting (each center is midway between edges)
    bin_centers = bin_edges[:-1] + bin_width / 2
    
    return counts, bin_centers, bins


def generate_tikz(bin_centers, counts, bins, output_path):
    """
    Generate a publication-ready histogram figure in TikZ/LaTeX format.
    
    This function creates a standalone LaTeX document containing TikZ code for
    a histogram of MS-SSIM scores. The histogram uses a logarithmic y-axis to
    better visualize the distribution across different orders of magnitude.
    
    The generated figure is highly customized for academic publication with:
    - Precise control over dimensions, fonts, and spacing
    - Logarithmic y-axis with explicitly formatted tick labels
    - Grid lines for better readability
    - Carefully chosen bar width and spacing
    - Fixed decimal precision for axis labels
    
    Args:
        bin_centers (numpy.ndarray): Center values of histogram bins
        counts (numpy.ndarray): Count of scores in each bin
        bins (numpy.ndarray): Bin edges (used for axis limits)
        output_path (str): Path where the LaTeX file will be saved
        
    Returns:
        None: The function writes the TikZ code to the specified output file
        
    Note:
        The resulting .tex file is a standalone LaTeX document that can be
        directly compiled with pdflatex to produce a PDF figure, or included
        in a larger LaTeX document.
    """
    # ===== CREATE TIKZ DOCUMENT HEADER =====
    # Define LaTeX preamble with required packages
    tikz_content = r"""\documentclass{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\usepackage{pgfplotstable}
\usepackage{siunitx}
\begin{document}
\begin{tikzpicture}
"""

    # ===== CONFIGURE PLOT APPEARANCE =====
    # Set up the axis environment with detailed styling options
    tikz_content += r"""\begin{axis}[
    width=3.5in,               % Figure width in inches
    height=2.5in,              % Figure height in inches
    ymode=log,                 % Use logarithmic scale for y-axis
    ymin=1,                    % Minimum y value (start at 1 for log scale)
    xmin=0.40,                 % Minimum x value (MS-SSIM score)
    xmax=1.00,                 % Maximum x value (MS-SSIM score)
    
    % Define x-axis tick positions and labels
    xtick={0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00},
    xticklabels={0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,1.00},
    xticklabel style={
        /pgf/number format/fixed,        % Use fixed-point notation
        /pgf/number format/precision=2,  % Two decimal places
        font=\tiny,                      % Small font for better fit
        align=center                     % Center-align tick labels
    },
    
    % Configure y-axis tick marks and labels
    scaled y ticks=false,               % Prevent scientific notation
    ytick={1,10,100,1000,10000},        % Explicit log-scale tick positions
    yticklabels={1,10,100,1000,10000},  % Explicit tick labels
    yticklabel style={
        font=\tiny,                     % Small font for tick labels
    },
    
    % Axis labels
    xlabel={MS-SSIM score},
    ylabel={Number of images (log scale)},
    
    % Grid configuration
    ymajorgrids=true,                   % Show major grid lines on y-axis
    yminorgrids=true,                   % Show minor grid lines on y-axis
    xmajorgrids=true,                   % Show major grid lines on x-axis
    minor tick num=1,                   % One minor tick between major ticks
    
    % Bar plot styling
    bar width=0.035,                    % Width of histogram bars
    enlarge x limits=0.025,             % Add small padding on x-axis
    
    % General styling
    tick label style={font=\tiny},      % Font size for all tick labels
    label style={font=\small},          % Font size for axis labels
    tick align=outside,                 % Place ticks outside the plot area
    axis line style={line width=0.5pt}, % Thin axis lines
    major grid style={line width=0.2pt, dashed},        % Style for major grid
    minor grid style={line width=0.1pt, dashed, gray!50}, % Style for minor grid
]
"""

    # ===== ADD HISTOGRAM DATA =====
    # Start the plot command for a bar chart with specified styling
    tikz_content += r"""\addplot+[ybar, fill=blue!60, draw=black] coordinates {
"""
    
    # Add each data point as a coordinate pair (bin_center, count)
    # Format bin centers to 3 decimal places for precision
    for center, count in zip(bin_centers, counts):
        tikz_content += f"    ({center:.3f}, {count})\n"

    # ===== CLOSE THE TIKZ DOCUMENT =====
    # End the plot command and close all environments
    tikz_content += r"""};
\end{axis}
\end{tikzpicture}
\end{document}
"""

    # ===== SAVE THE OUTPUT FILE =====
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Write the TikZ content to the output file
    with open(output_path, "w") as file:
        file.write(tikz_content)


def main():
    """
    Execute the complete workflow for generating an MS-SSIM score histogram.
    
    This function orchestrates the entire process of:
    1. Loading the NIND dataset metadata
    2. Extracting and filtering MS-SSIM scores
    3. Computing the histogram statistics
    4. Generating the TikZ figure
    5. Saving the output to a LaTeX file
    
    The function includes error handling for missing files and empty datasets,
    printing appropriate error messages when issues are encountered.
    
    Returns:
        None: The function outputs a LaTeX file and prints a confirmation message
              or error message to the console
    """
    # ===== DATA LOADING =====
    # Path to the NIND dataset metadata YAML file
    yaml_file = "/orb/benoit_phd/datasets/RawNIND/RawNIND_masks_and_alignments.yaml"

    # Verify that the input file exists before proceeding
    if not os.path.isfile(yaml_file):
        print(f"Error: YAML file not found at {yaml_file}")
        return

    # Load the dataset metadata from the YAML file
    data = read_yaml(yaml_file)

    # ===== DATA FILTERING =====
    # Extract MS-SSIM scores for all valid noisy images
    scores = filter_scores(data)

    # Verify that we have at least some valid scores to analyze
    if not scores:
        print(
            "No valid 'rgb_msssim_score' entries found where 'f_fpath' != 'gt_fpath'."
        )
        return

    # ===== HISTOGRAM COMPUTATION =====
    # Compute histogram counts and bin positions using default parameters
    # (Range: 0.40-1.00, Bin width: 0.05)
    counts, bin_centers, bins = compute_histogram(scores)

    # ===== FIGURE GENERATION =====
    # Path where the TikZ LaTeX figure will be saved
    output_tikz = "/orb/benoit_phd/wiki/Papers/JDDC/journal_paper/figures/histogram.tex"

    # Generate the TikZ figure with the histogram data
    generate_tikz(bin_centers, counts, bins, output_tikz)

    # Confirm successful generation with output path
    print(f"Histogram TikZ figure has been saved to {output_tikz}")


if __name__ == "__main__":
    main()
