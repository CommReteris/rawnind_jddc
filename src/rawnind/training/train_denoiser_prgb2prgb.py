import os
import statistics
import logging
import sys
from collections.abc import Iterable
import multiprocessing

from rawnind.training import training_loops
from rawnind.dependencies import pytorch_helpers
from rawnind.dependencies import raw_processing as raw
from rawnind.dependencies import raw_processing as rawproc


class DenoiserTrainingProfiledRGBToProfiledRGB(
    training_loops.DenoiserTraining, training_loops.PRGBImageToImageNNTraining
):
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
    # check if the args contain proc2proc anywhere
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
    denoiserTraining = DenoiserTrainingProfiledRGBToProfiledRGB()
    denoiserTraining.training_loop()
