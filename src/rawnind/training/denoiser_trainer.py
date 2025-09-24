"""Training classes for pure denoising models.

This module contains trainer classes for models that perform only denoising,
extracted from the original training scripts per the partition plan.

Consolidated from:
- train_denoiser_bayer2prgb.py
- train_denoiser_prgb2prgb.py
"""

import os
import statistics
import time
import logging
import sys
import multiprocessing
from typing import Optional
from collections.abc import Iterable
import torch

from . import training_loops
from ..dependencies import pytorch_helpers
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw

APPROX_EXPOSURE_DIFF_PENALTY = 1 / 10000


class DenoiserTrainingBayerToProfiledRGB(
    training_loops.DenoiserTraining,
    training_loops.BayerImageToImageNNTraining,
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



