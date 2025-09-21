"""Training classes for pure denoising models.

This module contains trainer classes for models that perform only denoising,
extracted from the original training scripts.
"""

import os
import statistics
import time
import logging
import sys
from typing import Optional
from collections.abc import Iterable
import multiprocessing
import torch

from . import training_loops
from ..dependencies import pytorch_helpers
from ..dependencies import rawproc
from ..dependencies import raw_processing as raw

APPROX_EXPOSURE_DIFF_PENALTY = 1 / 10000


class DenoiserTrainingBayerToProfiledRGB(
    training_loops.DenoiserTraining,
    training_loops.BayerImageToImageNNTraining,
    training_loops.BayerDenoiser,
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