"""Model loading utilities for inference.

This module provides functionality for loading and managing trained models
for inference purposes. It handles model checkpoint loading, device placement,
and model state management.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import logging
import os

import torch


# Import from dependencies package (will be moved later)


class ModelLoader:
    """Handles loading and management of trained models for inference.

    This class provides static methods for loading model checkpoints,
    finding best model iterations, and managing model state.
    """

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, device=None) -> None:
        """Load pre-trained weights into a model.

        This static method loads model weights from a saved checkpoint file into the
        provided model instance, moving the weights to the specified device.

        Args:
            model: PyTorch model instance to load weights into
            path: File path to the saved model state dictionary (.pt or .pth file)
            device: Optional device to load the model weights onto (e.g., "cuda:0" or "cpu")
                   If None, the weights are loaded onto the default device

        Returns:
            None

        Raises:
            FileNotFoundError: If the specified path does not exist
            RuntimeError: If the state dict structure doesn't match the model architecture

        Notes:
            - The method verifies the file exists before attempting to load it
            - A log message is generated upon successful loading
            - In case of a missing file, the debugger is invoked before raising the exception
              (when run in a debugging environment)
        """
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            logging.info(f"Loaded model from {path}")
        else:
            breakpoint()
            raise FileNotFoundError(path)

    @staticmethod
    def get_best_step(
            model_dpath: str,
            suffix: str,
            prefix: str = "val",
    ) -> dict:
        """Find the best-performing model checkpoint based on a specific metric.

        This method locates the best iteration of a trained model by examining the
        training results file (trainres.yaml) and identifying which training step
        achieved the best performance according to the specified metric.

        Args:
            model_dpath: Path to the model's experiment directory containing
                        the trainres.yaml file and saved_models subdirectory
            suffix: Metric name suffix to use for finding the best step
                   (e.g., "msssim", "psnr", "combined_loss")
            prefix: Metric name prefix, typically "val" for validation metrics
                   or "test" for test metrics

        Returns:
            dict: Dictionary containing:
                - "fpath": Full path to the best model checkpoint file
                - "step_n": The iteration/step number of the best checkpoint

        Raises:
            FileNotFoundError: If the trainres.yaml file doesn't exist in model_dpath
            KeyError: If the specified metric (prefix_suffix) isn't found in the results file

        Notes:
            - The metric is constructed as "{prefix}_{suffix}" (e.g., "val_msssim")
            - The method assumes that model checkpoints follow the naming pattern
              "iter_{step_number}.pt" and are stored in a "saved_models" subdirectory
            - The trainres.yaml file is expected to have a "best_step" section that
              maps metric names to their best iteration numbers
        """
        # Import utilities from dependencies (will be moved later)
        from ..dependencies.utilities import load_yaml

        jsonfpath = os.path.join(model_dpath, "trainres.yaml")
        if not os.path.isfile(jsonfpath):
            raise FileNotFoundError(
                "get_best_checkpoint: jsonfpath not found: {}".format(jsonfpath)
            )
        results = load_yaml(jsonfpath, error_on_404=False)
        metric = "{}_{}".format(prefix, suffix)
        try:
            best_step = results["best_step"][metric]
        except KeyError as e:
            raise KeyError(f'"{metric}" not found in {jsonfpath=}') from e
        return {
            "fpath" : os.path.join(model_dpath, "saved_models", f"iter_{best_step}.pt"),
            "step_n": best_step,
        }
