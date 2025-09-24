"""Configuration management utilities.

This module provides centralized configuration management for the
refactored codebase, including loading, parsing, and managing
configuration files.

Extracted from config/ directory as part of the codebase refactoring.
"""

import os
from typing import Any, Dict

# Import from dependencies (will be moved later)
from .json_saver import load_yaml


class ConfigManager:
    """Manages configuration files and settings.

    This class provides centralized access to configuration files
    and manages configuration loading and parsing.
    """

    # Base paths for different types of models
    MODELS_BASE_DPATH = os.path.join("..", "..", "models")

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            Dictionary containing configuration data
        """
        return load_yaml(config_path, error_on_404=True)

    @staticmethod
    def get_training_config(model_type: str, config_name: str) -> Dict[str, Any]:
        """Get training configuration for a specific model type.

        Args:
            model_type: Type of model (e.g., 'denoiser', 'compression')
            config_name: Name of configuration file

        Returns:
            Dictionary containing training configuration
        """
        config_path = os.path.join("config", f"train_{model_type}_{config_name}.yaml")
        return ConfigManager.load_config(config_path)

    @staticmethod
    def get_test_config(config_name: str) -> Dict[str, Any]:
        """Get test configuration.

        Args:
            config_name: Name of test configuration file

        Returns:
            Dictionary containing test configuration
        """
        config_path = os.path.join("config", f"test_{config_name}.yaml")
        return ConfigManager.load_config(config_path)

    @staticmethod
    def get_model_config(model_type: str, config_name: str) -> Dict[str, Any]:
        """Get model-specific configuration.

        Args:
            model_type: Type of model (e.g., 'dc', 'denoise')
            config_name: Name of model configuration file

        Returns:
            Dictionary containing model configuration
        """
        config_path = os.path.join("config", f"{model_type}_models_definitions.yaml")
        return ConfigManager.load_config(config_path)

    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            config_path: Path where to save the configuration
        """
        # Import from dependencies (will be moved later)
        from .json_saver import dict_to_yaml

        dict_to_yaml(config, config_path)

    @staticmethod
    def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries.

        Args:
            base_config: Base configuration dictionary
            override_config: Configuration dictionary with overrides

        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        for key, value in override_config.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        return merged
