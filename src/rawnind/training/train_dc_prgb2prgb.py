"""Train a joint denoise+compress model from profiled RGB to profiled RGB.

This entrypoint wires abstract_trainer mixins, loads defaults from
config/train_dc_prgb2prgb.yaml, and runs training_loop().
"""
import multiprocessing
import os
import logging
import sys
from collections.abc import Iterable

from rawnind.training import training_loops
from rawnind.dependencies import pytorch_helpers
from rawnind.dependencies import raw_processing as raw
from rawnind.dependencies import raw_processing as rawproc


class DCTrainingProfiledRGBToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.PRGBImageToImageNNTraining,
):
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
    denoiserTraining = DCTrainingProfiledRGBToProfiledRGB()
    denoiserTraining.training_loop()
