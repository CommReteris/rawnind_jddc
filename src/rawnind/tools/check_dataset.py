"""Quick validity scan over a dataset tree.

Walks a directory, filters out non-image paths, and checks readability of files
using common.libs.libimganalysis. Prints shell mv commands for files deemed
invalid so they can be quarantined easily.

This script is useful for cleaning up datasets before training to ensure all
image files can be properly loaded. Corrupted or unreadable images can cause
training failures or errors during data loading.

Usage:
    python check_dataset.py [--directory DIR] [--num_threads N]

Arguments:
    --directory: Root directory to scan (default: rawproc.DS_BASE_DPATH)
    --num_threads: Number of threads to use for parallel checking (default: 75% of CPU cores)

Output:
    Shell commands (mv source destination) for moving invalid files to quarantine directory
"""

import os
import sys
import tqdm
import argparse
import random

from rawnind.libs import rawproc  # includes DS_BASE_DPATH
from common.libs import pt_helpers
from rawnind.libs import raw
from common.libs import libimganalysis
from common.libs import utilities

# Destination directory for invalid files
BAD_SRC_FILES_DPATH = os.path.join("..", "..", "datasets", "RawNIND", "bad_src_files")


def is_valid_img_mtrunner(fpath):
    """Check if an image file is valid and can be properly loaded.
    
    This function is designed to be used with the utilities.mt_runner
    multi-threading utility. It checks if an image file can be properly 
    loaded and read using libimganalysis.is_valid_img.
    
    Args:
        fpath: Path to the image file to check
        
    Returns:
        Tuple of (file_path, is_valid_boolean) where is_valid_boolean is True
        if the image can be loaded successfully, False otherwise
    """
    return fpath, libimganalysis.is_valid_img(fpath)


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num_threads", type=int, default=os.cpu_count() // 4 * 3,
                        help="Number of threads to use for parallel checking")
    parser.add_argument("--directory", default=rawproc.DS_BASE_DPATH,
                        help="Root directory to scan for image files")
    args = parser.parse_args()

    # Step 1: Collect all image files to check, filtering out non-image files
    list_of_files = []
    print(f"Scanning directory: {args.directory}")
    for fpath in tqdm.tqdm(list(utilities.walk(args.directory)), desc="Collecting files"):
        fpath = os.path.join(*fpath)
        # Skip known non-image files and already quarantined files
        if (
                "bad_src_files" in fpath  # Skip already quarantined files
                or fpath.endswith(".txt")  # Skip text files
                or fpath.endswith(".yaml")  # Skip YAML files
        ):
            continue
        list_of_files.append(fpath)

    # Shuffle files to distribute workload more evenly across threads
    random.shuffle(list_of_files)
    print(f"Found {len(list_of_files)} files to check")

    # Step 2: Check validity of all files in parallel
    print(f"Checking file validity using {args.num_threads} threads")
    results = utilities.mt_runner(is_valid_img_mtrunner, list_of_files,
                                  num_threads=args.num_threads)

    # Step 3: Generate commands to move invalid files
    invalid_count = 0
    for result in results:
        fpath, is_valid = result
        if not is_valid:
            invalid_count += 1
            # Print shell command to move invalid file to quarantine directory
            print(f"mv {fpath} {BAD_SRC_FILES_DPATH}")

    # Print summary
    print(f"\nScan complete. Found {invalid_count} invalid files out of {len(list_of_files)}.")
    if invalid_count > 0:
        print(f"Run the commands above to move invalid files to {BAD_SRC_FILES_DPATH}")
    else:
        print("All files appear to be valid.")
