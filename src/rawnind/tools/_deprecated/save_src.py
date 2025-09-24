"""Source code backup utility for project files.

This script provides functionality to copy source code files from a project
directory to a backup or distribution directory. It selectively copies files
based on their extensions and can optionally filter by specific directories.

Key features:
- Copies source files (.py, .yaml by default) to a destination directory
- Preserves directory structure in the destination
- Can filter files by extension
- Can include only specified directories
- Automatically creates destination directories as needed

Typical use cases:
- Creating source code snapshots for versioning
- Preparing source code for distribution
- Backing up source code files while excluding non-code assets

Command-line usage:
    python save_src.py [DEST_ROOT_DPATH]
    
Where DEST_ROOT_DPATH is the destination root directory for the copied files.
"""

import os
import shutil
import sys
from typing import Optional, List, Tuple

from rawnind.dependencies import utilities

SRC_EXTENSIONS: tuple = ("py", "yaml")
SRC_ROOT_DPATH: str = ".."


def save_src(
        dest_root_dpath: str,
        src_root_dpath: str = SRC_ROOT_DPATH,
        src_extensions: Tuple[str, ...] = SRC_EXTENSIONS,
        included_dirs: Optional[List[str]] = None,
) -> None:
    """Copy source files from a source directory tree to a destination directory tree.
    
    Recursively walks the source directory tree, finds files with matching extensions,
    and copies them to the destination directory while preserving the directory structure.
    
    Args:
        dest_root_dpath: Destination root directory where files will be copied
        src_root_dpath: Source root directory to copy files from (default: parent directory)
        src_extensions: Tuple of file extensions to include (without leading dot)
        included_dirs: Optional list of directories to include; if provided, only files
                      in these directories will be copied
                      
    Returns:
        None
        
    Notes:
        - Creates destination directories if they don't exist
        - Uses utilities.walk() to traverse the source directory
        - Files are identified by their extensions (e.g., "py" for Python files)
        - Maintains the relative directory structure in the destination
    """
    for root, dn, fn in utilities.walk(src_root_dpath):
        if included_dirs is not None and dn not in included_dirs:
            continue
        if not any(fn.endswith("." + ext) for ext in src_extensions):
            continue
        dest_dpath: str = os.path.join(dest_root_dpath, dn)
        os.makedirs(dest_dpath, exist_ok=True)
        shutil.copyfile(os.path.join(root, dn, fn), os.path.join(dest_dpath, fn))


if __name__ == "__main__":
    # Command-line execution entry point
    # When run directly, the script requires exactly one argument:
    # the destination directory where source files will be copied

    # Validate command-line arguments
    assert len(sys.argv) == 2, "usage: python save_src.py [DEST_ROOT_DPATH]"

    # Call save_src with the specified destination directory
    # Uses default values for source directory and extensions
    save_src(dest_root_dpath=sys.argv[1])
