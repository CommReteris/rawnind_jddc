"""Experiment management utilities for training.

This module provides utilities for managing training experiments,
including model iteration management, cleanup, and experiment tracking.

Extracted from tools/ as part of the codebase refactoring.
"""

import logging
import os
from typing import List, Optional

# Import from dependencies package (will be moved later)
from ..dependencies.utilities import load_yaml


class ExperimentManager:
    """Manages training experiments and model iterations.

    This class provides utilities for managing training experiments,
    including finding best model iterations, cleaning up unused models,
    and managing experiment directories.
    """

    @staticmethod
    def find_best_expname_iteration(expname: str) -> Optional[str]:
        """Find the latest model iteration for a given experiment name.

        Args:
            expname: Base experiment name

        Returns:
            Path to the latest model iteration, or None if not found
        """
        # Import from dependencies (will be moved later)
        from ..dependencies.utilities import find_latest_model_expname_iteration

        try:
            return find_latest_model_expname_iteration(expname)
        except Exception as e:
            logging.error(f"Error finding latest model iteration: {e}")
            return None

    @staticmethod
    def cleanup_saved_models_iterations(save_dpath: str, keep_iterations: List[int]):
        """Clean up saved model iterations, keeping only specified ones.

        Args:
            save_dpath: Directory containing saved models
            keep_iterations: List of iteration numbers to keep
        """
        models_dir = os.path.join(save_dpath, "saved_models")
        if not os.path.exists(models_dir):
            return

        keepers = [f"iter_{step}" for step in keep_iterations]

        for fn in os.listdir(models_dir):
            if fn.partition(".")[0] not in keepers:
                fpath = os.path.join(models_dir, fn)
                logging.info(f"cleanup_models: rm {fpath}")
                os.remove(fpath)

    @staticmethod
    def cleanup_saved_models_unused_test_images(save_dpath: str):
        """Clean up unused test images from saved models directory.

        Args:
            save_dpath: Directory containing saved models
        """
        # This is a placeholder - full implementation would be more complex
        logging.info(f"cleanup_saved_models_unused_test_images: {save_dpath}")

    @staticmethod
    def rm_empty_models(save_dpath: str):
        """Remove empty model files.

        Args:
            save_dpath: Directory containing saved models
        """
        models_dir = os.path.join(save_dpath, "saved_models")
        if not os.path.exists(models_dir):
            return

        for fn in os.listdir(models_dir):
            fpath = os.path.join(models_dir, fn)
            if os.path.getsize(fpath) == 0:
                logging.info(f"rm_empty_models: rm {fpath}")
                os.remove(fpath)

    @staticmethod
    def rm_nonbest_model_iterations(save_dpath: str, best_steps: List[int]):
        """Remove model iterations that are not among the best performing.

        Args:
            save_dpath: Directory containing saved models
            best_steps: List of step numbers that are considered best
        """
        models_dir = os.path.join(save_dpath, "saved_models")
        if not os.path.exists(models_dir):
            return

        keepers = [f"iter_{step}" for step in best_steps]

        for fn in os.listdir(models_dir):
            if fn.partition(".")[0] not in keepers:
                fpath = os.path.join(models_dir, fn)
                logging.info(f"rm_nonbest_model_iterations: rm {fpath}")
                os.remove(fpath)

    @staticmethod
    def get_best_steps_from_results(results_fpath: str) -> List[int]:
        """Extract best step numbers from training results.

        Args:
            results_fpath: Path to training results YAML file

        Returns:
            List of best step numbers
        """
        try:
            results = load_yaml(results_fpath, error_on_404=False)
            if results and "best_step" in results:
                return list(results["best_step"].values())
            return []
        except Exception as e:
            logging.error(f"Error loading results from {results_fpath}: {e}")
            return []
