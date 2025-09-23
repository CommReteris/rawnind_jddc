"""Training classes for joint denoise+compress models.

This module contains trainer classes for models that perform both denoising
and compression, extracted from the original training scripts per the partition plan.

Consolidated from:
- train_dc_bayer2prgb.py
- train_dc_prgb2prgb.py
"""

import multiprocessing
import os
import sys
import logging
from typing import Optional
from collections.abc import Iterable
import torch

from . import training_loops
from ..dependencies import pytorch_helpers
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw


class DCTrainingBayerToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.BayerImageToImageNNTraining,
):
    """Train a joint denoise+compress model from Bayer to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_dc_bayer2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_dc_bayer2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args) -> None:
        if not args.in_channels:
            args.in_channels = 4
        super().autocomplete_args(args)


class DCTrainingProfiledRGBToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.PRGBImageToImageNNTraining,
):
    """Train a joint denoise+compress model from profiled RGB to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_dc_prgb2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_dc_prgb2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args):
        if not args.in_channels:
            args.in_channels = 3
        super().autocomplete_args(args)


# Clean API factory functions (no CLI dependencies)
def create_dc_bayer_trainer(**kwargs) -> DCTrainingBayerToProfiledRGB:
    """Create a Bayer-to-RGB denoising+compression trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DCTrainingBayerToProfiledRGB instance
    """
    return DCTrainingBayerToProfiledRGB(launch=False, **kwargs)


def create_dc_rgb_trainer(**kwargs) -> DCTrainingProfiledRGBToProfiledRGB:
    """Create an RGB-to-RGB denoising+compression trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DCTrainingProfiledRGBToProfiledRGB instance
    """
    return DCTrainingProfiledRGBToProfiledRGB(launch=False, **kwargs)


# Legacy CLI support (for backward compatibility, but deprecated)
def _legacy_cli_main():
    """Legacy CLI entry point - deprecated in favor of clean API."""
    logging.warning("Legacy CLI interface is deprecated. Use clean API factory functions instead.")
    
    # Handle multiprocessing for proc2proc or opencv arguments
    if any("proc2proc" in arg or "opencv" in arg for arg in sys.argv):
        try:
            print("setting multiprocessing.set_start_method('spawn')")
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("multiprocessing.set_start_method('spawn') failed")
            logging.info("multiprocessing.set_start_method('spawn') failed - method already set")

    # Determine which trainer to use based on arguments
    if any("bayer" in arg.lower() for arg in sys.argv):
        trainer = DCTrainingBayerToProfiledRGB()
    else:
        trainer = DCTrainingProfiledRGBToProfiledRGB()

    trainer.training_loop()


if __name__ == "__main__":
    _legacy_cli_main()
