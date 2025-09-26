"""Training classes for pure denoising models.

This module contains trainer classes for models that perform only denoising,
<<<<<<< HEAD
extracted from the original training scripts.
=======
extracted from the original training scripts per the partition plan.

Consolidated from:
- train_denoiser_bayer2prgb.py
- train_denoiser_prgb2prgb.py
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
"""

import os
import statistics
import time
import logging
import sys
<<<<<<< HEAD
from typing import Optional
from collections.abc import Iterable
import multiprocessing
=======
import multiprocessing
from typing import Optional
from collections.abc import Iterable
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
import torch

from . import training_loops
from ..dependencies import pytorch_helpers
<<<<<<< HEAD
from ..dependencies import rawproc
=======
from ..dependencies import raw_processing as rawproc
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
from ..dependencies import raw_processing as raw

APPROX_EXPOSURE_DIFF_PENALTY = 1 / 10000


class DenoiserTrainingBayerToProfiledRGB(
    training_loops.DenoiserTraining,
    training_loops.BayerImageToImageNNTraining,
<<<<<<< HEAD
    training_loops.BayerDenoiser,
=======
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
):
    """Train a denoiser from Bayer to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_denoise_bayer2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiserTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_denoise_bayer2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args) -> None:
        if not args.in_channels:
            args.in_channels = 4
        super().autocomplete_args(args)


class DenoiserTrainingProfiledRGBToProfiledRGB(
    training_loops.DenoiserTraining,
    training_loops.PRGBImageToImageNNTraining,
):
    """Train a denoiser from profiled RGB to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_denoise_prgb2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiserTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_denoise_prgb2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args):
        if not args.in_channels:
            args.in_channels = 3
        super().autocomplete_args(args)


<<<<<<< HEAD
if __name__ == "__main__":
    # Handle multiprocessing for proc2proc or opencv arguments
    if any("proc2proc" in arg or "opencv" in arg for arg in sys.argv):
        try:
            print("setting multiprocessing.set_start_method('spawn')")
            multiprocessing.set_start_method("spawn")
        except RuntimeError:
            print("multiprocessing.set_start_method('spawn') failed")
            pass

    # try:
    #     os.nice(1)
    # except OSError:
    #     pass

    # Determine which trainer to use based on arguments
    if any("bayer" in arg.lower() for arg in sys.argv):
        denoiserTraining = DenoiserTrainingBayerToProfiledRGB()
    else:
        denoiserTraining = DenoiserTrainingProfiledRGBToProfiledRGB()

    denoiserTraining.training_loop()
=======
# Clean API factory functions (no CLI dependencies)
def create_bayer_denoiser_trainer(**kwargs) -> DenoiserTrainingBayerToProfiledRGB:
    """Create a Bayer-to-RGB denoiser trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DenoiserTrainingBayerToProfiledRGB instance
    """
    return DenoiserTrainingBayerToProfiledRGB(launch=False, **kwargs)


def create_rgb_denoiser_trainer(**kwargs) -> DenoiserTrainingProfiledRGBToProfiledRGB:
    """Create an RGB-to-RGB denoiser trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DenoiserTrainingProfiledRGBToProfiledRGB instance
    """
    return DenoiserTrainingProfiledRGBToProfiledRGB(launch=False, **kwargs)



>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
