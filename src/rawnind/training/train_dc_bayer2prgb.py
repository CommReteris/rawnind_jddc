"""Train a joint denoise+compress model from Bayer to profiled RGB.

This entrypoint wires abstract_trainer mixins, loads defaults from
config/train_dc_bayer2prgb.yaml, and runs training_loop().
"""
import os
import sys
from typing import Optional
from collections.abc import Iterable
import torch
import logging

from rawnind.training import training_loops


class DCTrainingBayerToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.BayerImageToImageNNTraining,
):
    CLS_CONFIG_FPATHS = training_loops.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_dc_bayer2prgb.yaml")
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
    # logging.getLogger().setLevel(logging.DEBUG)
    denoiserTraining = DCTrainingBayerToProfiledRGB()
    denoiserTraining.training_loop()
