"""Dependencies package for shared utilities, configurations, and common functionality.

This package contains all shared components and utilities, including:
- Common utilities and helper functions
- Configuration management
- Logging and error handling
- PyTorch and NumPy operations
- Raw image processing utilities
"""

from .config_manager import ConfigManager

__all__ = [
    'ConfigManager'
]
