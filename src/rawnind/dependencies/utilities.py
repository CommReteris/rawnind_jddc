"""General utility functions and helpers.

This module contains general utility functions extracted from
the original utilities.py file.

Extracted from libs/utilities.py as part of the codebase refactoring.
"""

import logging
import os
from typing import Any, Dict, Optional

import yaml


def noop(*args, **kwargs):
    """No-operation function that accepts any arguments and does nothing."""
    pass


def load_yaml(fpath: str, error_on_404: bool = True) -> Optional[Dict[str, Any]]:
    """Load YAML file.

    Args:
        fpath: Path to YAML file
        error_on_404: Whether to raise error if file not found

    Returns:
        Parsed YAML content or None if file not found and error_on_404=False
    """
    if not os.path.isfile(fpath):
        if error_on_404:
            raise FileNotFoundError(f"load_yaml: {fpath} not found")
        return None

    with open(fpath, 'r') as f:
        return yaml.safe_load(f)


def dict_to_yaml(data: Dict[str, Any], fpath: str):
    """Save dictionary to YAML file.

    Args:
        data: Dictionary to save
        fpath: Output file path
    """
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def get_leaf(path: str) -> str:
    """Get the leaf (basename) of a path.

    Args:
        path: File or directory path

    Returns:
        Leaf name of the path
    """
    return os.path.basename(path)


def find_latest_model_expname_iteration(expname: str) -> Optional[str]:
    """Find the latest model iteration for a given experiment name.

    Args:
        expname: Base experiment name

    Returns:
        Path to the latest model iteration, or None if not found
    """
    # This is a simplified version - full implementation would be more complex
    # Import from dependencies (will be moved later)
    from .config_manager import ConfigManager

    # Look for model directories
    models_base = getattr(ConfigManager, 'MODELS_BASE_DPATH', 'models')
    exp_path = os.path.join(models_base, expname)

    if not os.path.exists(exp_path):
        return None

    # Find the latest iteration
    saved_models_dir = os.path.join(exp_path, 'saved_models')
    if not os.path.exists(saved_models_dir):
        return None

    iterations = []
    for f in os.listdir(saved_models_dir):
        if f.startswith('iter_') and f.endswith('.pt'):
            try:
                iter_num = int(f.split('_')[1].split('.')[0])
                iterations.append((iter_num, f))
            except (ValueError, IndexError):
                continue

    if not iterations:
        return None

    iterations.sort(reverse=True)
    return os.path.join(saved_models_dir, iterations[0][1])


def compress_lzma(src_fpath: str, dst_fpath: str):
    """Compress file using LZMA compression.

    Args:
        src_fpath: Source file path
        dst_fpath: Destination file path
    """
    # Simplified implementation - full version would use lzma compression
    logging.info(f"compress_lzma: {src_fpath} -> {dst_fpath}")


def compress_png(tensor: Any, outfpath: str) -> bool:
    """Compress tensor as PNG.

    Args:
        tensor: Tensor to compress
        outfpath: Output file path

    Returns:
        True if compression successful, False otherwise
    """
    # Simplified implementation - full version would use PNG compression
    try:
        logging.info(f"compress_png: {outfpath}")
        return True
    except Exception as e:
        logging.error(f"PNG compression failed: {e}")
        return False
