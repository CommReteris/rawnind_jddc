"""Train a denoiser from Bayer to profiled RGB.

This entrypoint wires abstract_trainer mixins, loads defaults from
config/train_denoise_bayer2prgb.yaml, and runs training_loop().
"""
import os
import sys
import statistics
import time
import logging
from typing import Optional
from collections.abc import Iterable
import torch

from rawnind.training import training_loops
from rawnind.dependencies import pytorch_helpers
from rawnind.dependencies import raw_processing as rawproc
from rawnind.dependencies import raw_processing as raw

APPROX_EXPOSURE_DIFF_PENALTY = 1 / 10000


class DenoiserTrainingBayerToProfiledRGB(
    training_loops.DenoiserTraining,
    training_loops.BayerImageToImageNNTraining,
    training_loops.BayerDenoiser,
):
    CLS_CONFIG_FPATHS = training_loops.DenoiserTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_denoise_bayer2prgb.yaml")
    ]

    def __init__(self, launch=False, **kwargs) -> None:
        super().__init__(launch=launch, **kwargs)

    def autocomplete_args(self, args) -> None:
        if not args.in_channels:
            args.in_channels = 4
        super().autocomplete_args(args)


if __name__ == "__main__":
    # try:
    #     os.nice(1)
    # except OSError:
    #     pass
    denoiserTraining = DenoiserTrainingBayerToProfiledRGB()
    denoiserTraining.training_loop()
