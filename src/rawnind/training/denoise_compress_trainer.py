"""Training classes for joint denoise+compress models.

This module contains trainer classes for models that perform both denoising
and compression, extracted from the original training scripts.
"""

import multiprocessing
import os
import sys
from typing import Optional
from collections.abc import Iterable
import torch
import logging

from . import training_loops
from ..dependencies import pytorch_helpers
from ..dependencies import rawproc
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
        denoiserTraining = DCTrainingBayerToProfiledRGB()
    else:
        denoiserTraining = DCTrainingProfiledRGBToProfiledRGB()

    denoiserTraining.training_loop()