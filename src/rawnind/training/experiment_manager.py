"""Experiment management utilities for training.

This module provides utilities for managing training experiments,
including model iteration management, cleanup, and experiment tracking.

Extracted from tools/ as part of the codebase refactoring.
"""

import os
import time
import shutil
import yaml
import re
import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple, Literal

# Import from dependencies package
from ..dependencies.json_saver import load_yaml


class ExperimentManager:
    """Manages training experiments and model iterations.

    This class provides utilities for managing training experiments,
    including finding best model iterations, cleaning up unused models,
    and managing experiment directories.
    """

    def __init__(self, models_root_path: str = "models"):
        """Initialize the experiment manager.

        Args:
            models_root_path: Root path for model storage
        """
        self.models_root_path = models_root_path

    @staticmethod
    def _get_model_type(expname: str) -> Literal["rawnind_dc", "rawnind_denoise"]:
        """Get the model's root directory path for a given experiment."""
        if expname.startswith("DenoiserTraining") or expname.startswith(
            "train_denoise"
        ):
            return "rawnind_denoise"
        elif (
            expname.startswith("DCTraining")
            or expname.startswith("train_dc")
            or "dc" in expname
        ):
            return "rawnind_dc"
        else:
            raise ValueError(
                f"Unable to determine whether experiment is train_dc or train_denoise: {expname}"
            )

    @staticmethod
    def _get_model_load_metric(
        model_dpath: str, model_type: Literal["rawnind_dc", "rawnind_denoise"]
    ) -> str:
        """Get the model's load metric for a given experiment."""
        with open(os.path.join(model_dpath, "trainres.yaml"), "r") as f:
            trainres = yaml.safe_load(f)
        if model_type == "rawnind_dc":
            for akey in trainres["best_step"]:
                if akey.startswith("val_combined"):
                    return akey
        elif model_type == "rawnind_denoise":
            # load args.yaml to determine loss
            with open(os.path.join(model_dpath, "args.yaml"), "r") as f:
                args = yaml.safe_load(f)
            # load trainres.yaml
            for akey in trainres["best_step"]:
                if akey.startswith(f"val_{args['loss']}"):
                    return akey
        raise ValueError(f"Unable to determine load metric for {model_dpath}")

    @staticmethod
    def _get_next_expname_iteration_dpath(model_dpath: str) -> str:
        """Get the next model iteration path."""
        if model_dpath.endswith("_"):
            new_dpath = model_dpath + "-1"
        elif model_dpath[-2] == "-" or model_dpath[-3] == "-":
            digit = model_dpath.split("-")[-1]
            new_dpath = model_dpath[: -len(digit)] + str(int(digit) + 1)
        else:
            raise ValueError(
                f"Unable to determine next model iteration for {model_dpath}"
            )
        return new_dpath

    def find_latest_model_expname_iteration(
        self,
        expname: str,
        model_type: Optional[Literal["rawnind_dc", "rawnind_denoise"]] = None,
        look_no_further: bool = False,
    ) -> str:
        """Find the best model training iteration for a given experiment."""
        if model_type is None:
            model_type = self._get_model_type(expname)
        model_dpath = os.path.join(self.models_root_path, model_type, expname)
        load_metric = self._get_model_load_metric(model_dpath, model_type)
        # load trainres.yaml
        with open(os.path.join(model_dpath, "trainres.yaml"), "r") as f:
            trainres = yaml.safe_load(f)
        try:
            best_iteration = trainres["best_step"][load_metric]
        except KeyError as e:
            raise ValueError(
                f"Unable to find best iteration w/{load_metric} for in {os.path.join(model_dpath, 'trainres.yaml')}"
            ) from e
        model_fpath = os.path.join(
            model_dpath, "saved_models", f"iter_{best_iteration}.pt"
        )
        if look_no_further:
            return model_fpath

        next_dpath = self._get_next_expname_iteration_dpath(model_dpath)
        next_expname = os.path.basename(next_dpath)
        try:
            return self.find_latest_model_expname_iteration(next_expname, model_type)
        except (ValueError, FileNotFoundError):
            if os.path.exists(model_fpath):
                return expname
            else:
                print(f"Best model iteration does not exist: {model_fpath}")
                raise ValueError(f"Best model iteration does not exist: {model_fpath}")

    @staticmethod
    def cleanup_saved_models_iterations(
        save_dpath: str,
        keep_iterations: List[int],
        model_type: Literal["compression", "denoising"] = None,
        delete: bool = True,
    ) -> int:
        """Clean up saved model iterations, keeping only specified ones.

        Args:
            save_dpath: Directory containing saved models
            keep_iterations: List of iteration numbers to keep
            model_type: Type of model for determining best steps
            delete: Whether to actually delete files

        Returns:
            Total bytes that would be saved
        """
        models_dir = Path(save_dpath) / "saved_models"
        if not models_dir.exists():
            return 0

        # If model_type is provided, load best steps from trainres.yaml
        if model_type:
            yaml_path = Path(save_dpath) / "trainres.yaml"
            if yaml_path.exists():
                best_steps = self._load_best_steps_from_yaml(yaml_path, model_type)
                keep_iterations.extend(best_steps)

        keepers = [f"iter_{step}" for step in set(keep_iterations)]
        total_bytes_saved = 0

        # Compile regex to match files like iter_{number}.pt and iter_{number}.pt.opt
        pattern = re.compile(r"^iter_(\d+)\.pt(?:\.opt)?$")

        for file in models_dir.iterdir():
            if not file.is_file():
                continue

            match = pattern.match(file.name)
            if match:
                step_str = f"iter_{match.group(1)}"
                if step_str not in keepers:
                    file_size = file.stat().st_size
                    total_bytes_saved += file_size
                    if delete:
                        try:
                            file.unlink()
                            logging.info(f"Deleted: {file}")
                        except Exception as e:
                            logging.error(f"Error deleting {file}: {e}")
                    else:
                        logging.info(f"Would delete: {file}")

        return total_bytes_saved

    def _load_best_steps_from_yaml(self, yaml_path: Path, model_type: str) -> List[int]:
        """Load the best_step values from the YAML file based on the model type."""
        with open(yaml_path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file {yaml_path}: {e}")
                return []

        best_step = data.get("best_step", {})
        steps = []

        if model_type == "compression":
            # Collect steps for keys starting with 'val_combined'
            for key, step in best_step.items():
                if key.startswith("val_combined"):
                    steps.append(step)
        elif model_type == "denoising":
            # Collect steps for keys starting with 'val_msssim_loss'
            for key, step in best_step.items():
                if key.startswith("val_msssim_loss"):
                    steps.append(step)
        else:
            logging.warning(f"Unknown model type: {model_type}")

        return steps

    def cleanup_saved_models_unused_test_images(
        self,
        save_dpath: str,
        important_models: Set[str] = None,
        exclude_substring: str = "bm3d",
        delete: bool = True,
    ) -> Tuple[List[str], int]:
        """Clean up unused test images from saved models directory.

        Args:
            save_dpath: Directory containing saved models
            important_models: Set of important model names to preserve
            exclude_substring: Substring to exclude from cleanup
            delete: Whether to actually delete files

        Returns:
            Tuple of (models_cleaned, total_bytes_saved)
        """
        base_dir = Path(save_dpath)
        models_to_clean = []
        total_bytes_saved = 0

        if important_models is None:
            important_models = set()

        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue
            if model_dir.name in important_models:
                continue
            if (
                exclude_substring
                and exclude_substring.lower() in model_dir.name.lower()
            ):
                logging.info(
                    f"Excluding model due to '{exclude_substring}' in name: {model_dir.name}"
                )
                continue

            # Find all .tif and .exr files recursively
            image_files = list(model_dir.rglob("*.tif")) + list(
                model_dir.rglob("*.exr")
            )
            if not image_files:
                continue  # No images to clean in this model

            # Calculate total size for this model
            model_size = sum(f.stat().st_size for f in image_files)
            total_bytes_saved += model_size
            models_to_clean.append(str(model_dir))

            if delete:
                for file in image_files:
                    try:
                        file.unlink()
                        logging.info(f"Deleted image: {file}")
                    except Exception as e:
                        logging.error(f"Error deleting {file}: {e}")

        return models_to_clean, total_bytes_saved

    @staticmethod
    def rm_empty_models(
        root_models_dpaths: List[str],
        delete: bool = True,
        time_limit_start: int = 60,
        time_limit_trainlog: int = 5 * 60,
        time_limit_trainres: int = 15 * 60,
        time_limit_saved_models: int = 60 * 60,
        time_limit_trainres_empty: int = 60 * 60,
    ) -> List[str]:
        """Remove empty model directories.

        Args:
            root_models_dpaths: List of root model directory paths
            delete: Whether to actually delete directories
            time_limit_*: Various time limits for determining if models are empty

        Returns:
            List of directories that were removed or would be removed
        """
        removed_dirs = []

        for root_models_dpath in root_models_dpaths:
            if not os.path.exists(root_models_dpath):
                continue

            # Clean empty .pt files in saved_models if present
            saved_models_path = os.path.join(root_models_dpath, "saved_models")
            if os.path.exists(saved_models_path):
                for f in os.listdir(saved_models_path):
                    file_path = os.path.join(saved_models_path, f)
                    if os.path.getsize(file_path) == 0:
                        removed_files.append(file_path)
                        if delete:
                            os.unlink(file_path)
                            logging.info(f"Deleted empty file: {file_path}")

            # Get a list of subdirectories in the model path
            models_dpaths = [
                f.path for f in os.scandir(root_models_dpath) if f.is_dir()
            ]

            # Loop through each subdirectory
            for model_dpath in models_dpaths:
                # Skip backups and old directories
                if os.path.basename(model_dpath) in ["backups", "old"]:
                    continue

                # Get the last modified time of the model directory
                mod_time = os.stat(model_dpath).st_mtime

                # Spare if there are saved models
                if os.path.exists(os.path.join(model_dpath, "saved_models")):
                    if len(os.listdir(os.path.join(model_dpath, "saved_models"))) > 0:
                        # and trainres.yaml's best_step key is not an empty dictionary
                        if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                            with open(os.path.join(model_dpath, "trainres.yaml")) as f:
                                trainres = yaml.load(f, Loader=yaml.FullLoader)
                            if (
                                (  # check that at least one key in trainres["best_step"] starts with "val"
                                    trainres["best_step"] != {}
                                    and any(
                                        akey.startswith("val")
                                        for akey in trainres["best_step"].keys()
                                    )
                                )
                                or time.time() - mod_time < time_limit_trainres_empty
                            ):
                                continue
                # Spare if the model was just created
                if time.time() - mod_time < time_limit_start:
                    continue
                # Spare if the train.log file was modified within the time limit
                if os.path.exists(os.path.join(model_dpath, "train.log")):
                    trainlog_mod_time = os.stat(
                        os.path.join(model_dpath, "train.log")
                    ).st_mtime
                    if time.time() - trainlog_mod_time < time_limit_trainlog:
                        continue
                # Spare if the trainres.yaml file was modified within the time limit
                if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                    trainres_mod_time = os.stat(
                        os.path.join(model_dpath, "trainres.yaml")
                    ).st_mtime
                    if time.time() - trainres_mod_time < time_limit_trainres:
                        continue
                # Spare if saved_models is empty but trainres.yaml exists and the model was created within the time limit
                if os.path.exists(os.path.join(model_dpath, "saved_models")):
                    if len(os.listdir(os.path.join(model_dpath, "saved_models"))) == 0:
                        if os.path.exists(os.path.join(model_dpath, "trainres.yaml")):
                            if time.time() - mod_time < time_limit_saved_models:
                                continue

                # Remove empty .pt files in saved_models
                saved_models_path = os.path.join(model_dpath, "saved_models")
                if os.path.exists(saved_models_path):
                    for f in os.listdir(saved_models_path):
                        file_path = os.path.join(saved_models_path, f)
                        if os.path.getsize(file_path) == 0:
                            removed_files.append(file_path)
                            if delete:
                                os.unlink(file_path)
                                logging.info(f"Deleted empty file: {file_path}")
                # Remove empty .pt files in saved_models
                saved_models_path = os.path.join(model_dpath, "saved_models")
                if os.path.exists(saved_models_path):
                    for f in os.listdir(saved_models_path):
                        file_path = os.path.join(saved_models_path, f)
                        if os.path.getsize(file_path) == 0:
                            removed_files.append(file_path)
                            if delete:
                                os.unlink(file_path)
                                logging.info(f"Deleted empty file: {file_path}")
                # No directory removal for clean API; only files
                removed_dirs.append(model_dpath)
        return removed_dirs

    def rm_nonbest_model_iterations(
        self, save_dpath: str, best_steps: List[int], delete: bool = True
    ) -> int:
        """Remove model iterations that are not among the best performing.

        Args:
            save_dpath: Directory containing saved models
            best_steps: List of step numbers that are considered best
            delete: Whether to actually delete files

        Returns:
            Total bytes saved
        """
        models_dir = Path(save_dpath) / "saved_models"
        if not models_dir.exists():
            return 0

        keepers = [f"iter_{step}" for step in best_steps]
        total_bytes_saved = 0

        pattern = re.compile(r"^iter_(\d+)\.pt(?:\.opt)?$")

        for file in models_dir.iterdir():
            if not file.is_file():
                continue

            match = pattern.match(file.name)
            if match:
                step_str = f"iter_{match.group(1)}"
                if step_str not in keepers:
                    file_size = file.stat().st_size
                    total_bytes_saved += file_size
                    if delete:
                        try:
                            file.unlink()
                            logging.info(f"rm_nonbest_model_iterations: deleted {file}")
                        except Exception as e:
                            logging.error(f"Error deleting {file}: {e}")
                    else:
                        logging.info(
                            f"rm_nonbest_model_iterations: would delete {file}"
                        )

        return total_bytes_saved

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

    def cleanup_experiment_directory(
        self,
        experiment_dir: str,
        keep_best_only: bool = True,
        clean_images: bool = False,
        delete: bool = True,
    ) -> dict:
        """Comprehensive cleanup of an experiment directory.

        Args:
            experiment_dir: Path to experiment directory
            keep_best_only: Whether to keep only best model iterations
            clean_images: Whether to clean test images
            delete: Whether to actually delete files

        Returns:
            Dictionary with cleanup statistics
        """
        stats = {"models_cleaned": 0, "images_cleaned": 0, "bytes_saved": 0}

        if not os.path.exists(experiment_dir):
            return stats

        # Clean up model iterations
        if keep_best_only:
            trainres_path = os.path.join(experiment_dir, "trainres.yaml")
            best_steps = self.get_best_steps_from_results(trainres_path)
            if best_steps:
                bytes_saved = self.rm_nonbest_model_iterations(
                    experiment_dir, best_steps, delete=delete
                )
                stats["bytes_saved"] += bytes_saved
                stats["models_cleaned"] = len(best_steps)

        # Clean up test images
        if clean_images:
            models_cleaned, bytes_saved = self.cleanup_saved_models_unused_test_images(
                experiment_dir, delete=delete
            )
            stats["images_cleaned"] = len(models_cleaned)
            stats["bytes_saved"] += bytes_saved

        return stats


# Factory functions for clean API compatibility
def find_latest_model_expname_iteration(expname: str) -> str:
    """Find the latest model iteration for a given experiment name.

    Args:
        expname: Base experiment name

    Returns:
        Path to the latest model iteration
    """
    manager = ExperimentManager()
    return manager.find_latest_model_expname_iteration(expname)


def cleanup_experiments(
    root_models_paths: List[str],
    keep_best_only: bool = True,
    clean_images: bool = False,
    delete: bool = True,
) -> dict:
    """Clean up multiple experiment directories.

    Args:
        root_models_paths: List of root model directory paths
        keep_best_only: Whether to keep only best iterations
        clean_images: Whether to clean test images
        delete: Whether to actually delete files

    Returns:
        Dictionary with overall cleanup statistics
    """
    manager = ExperimentManager()
    total_stats = {
        "experiments_processed": 0,
        "total_bytes_saved": 0,
        "total_models_cleaned": 0,
        "total_images_cleaned": 0,
    }

    for root_path in root_models_paths:
        if not os.path.exists(root_path):
            continue

        for exp_dir in os.listdir(root_path):
            exp_path = os.path.join(root_path, exp_dir)
            if os.path.isdir(exp_path):
                stats = manager.cleanup_experiment_directory(
                    exp_path, keep_best_only, clean_images, delete
                )
                total_stats["experiments_processed"] += 1
                total_stats["total_bytes_saved"] += stats["bytes_saved"]
                total_stats["total_models_cleaned"] += stats["models_cleaned"]
                total_stats["total_images_cleaned"] += stats["images_cleaned"]

    return total_stats
