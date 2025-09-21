"""Unit tests for dataset classes.

This module contains comprehensive unit tests for all dataset classes
in the dataset package, ensuring proper functionality and data integrity.
"""

import random
import time
import unittest

import torch

# Import raw processing (will be moved to dependencies later)
from ...libs import rawproc, arbitrary_proc_fun

# Import dataset classes
from ..clean_datasets import CleanProfiledRGBCleanBayerImageCropsDataset, \
    CleanProfiledRGBCleanProfiledRGBImageCropsDataset
from ..noisy_datasets import CleanProfiledRGBNoisyBayerImageCropsDataset, \
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
from ..validation_datasets import CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset, \
    CleanProfiledRGBNoisyBayerImageCropsValidationDataset, \
    CleanProfiledRGBNoisyBayerImageCropsTestDataloader, \
    CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader

# Constants from original rawds.py
BREAKPOINT_ON_ERROR = True
COLOR_PROFILE = "lin_rec2020"
TOY_DATASET_LEN = 25  # debug option


class DataLoadersUnitTests(unittest.TestCase):
    def test_CleanProfiledRGBNoisyBayerImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            num_crops=4,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyBayerImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(
            content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH],
            num_crops=4,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyProfiledRGBImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBCleanBayerImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBCleanBayerImageCropsDataset(
            content_fpaths=rawproc.EXTRARAW_CONTENT_FPATHS, num_crops=4, crop_size=256
        )
        print(
            f"Time to load CleanProfiledRGBCleanBayerImageCropsDataset dataset: {time.time() - pretime}"
        )
        pretime = time.time()
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
            self.assertEqual(image["gain"], 1.0)
        for i in range(len(ds)):
            self.assertGreater(
                len(ds._dataset[i]["crops"]), 0, f"{ds._dataset[i]} has no crops."
            )
            acrop = random.choice(ds._dataset[i]["crops"])
        print(
            f"Time to check CleanProfiledRGBCleanBayerImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBCleanProfiledRGBImageCropsDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBCleanProfiledRGBImageCropsDataset(
            content_fpaths=rawproc.EXTRARAW_CONTENT_FPATHS, num_crops=4, crop_size=256
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (4, 3, 256, 256))
            self.assertEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBCleanProfiledRGBImageCropsDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(self):
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(
            content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH],
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (3, 256, 256))
            self.assertEqual(image["mask_crops"].shape, (3, 256, 256))
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to load CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyBayerImageCropsValidationDataset(self):
        pretime = time.time()
        test_reserve = [
            "ursulines-red",
            "stefantiek",
            "ursulines-building",
            "MuseeL-Bobo",
            "CourtineDeVillersDebris",
            "Vaxt-i-trad",
            "Pen-pile",
            "MuseeL-vases",
        ]
        ds = CleanProfiledRGBNoisyBayerImageCropsValidationDataset(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=test_reserve,
        )
        print(
            f"Time to load CleanProfiledRGBNoisyBayerImageCropsValidationDataset dataset: {time.time() - pretime}"
        )
        pretime = time.time()
        for i in (0, -1):
            image = ds[i]
            self.assertEqual(image["x_crops"].shape, (3, 256, 256))
            self.assertEqual(image["y_crops"].shape, (4, 128, 128))
            self.assertEqual(image["mask_crops"].shape, (3, 256, 256))
            self.assertEqual(image["rgb_xyz_matrix"].shape, (4, 3))
        for imagedict in ds:
            self.assertGreaterEqual(imagedict["x_crops"].shape[-1], 256)
            self.assertGreaterEqual(imagedict["x_crops"].shape[-2], 256)
            self.assertNotEqual(image["gain"], 1.0)
        print(
            f"Time to check CleanProfiledRGBNoisyBayerImageCropsValidationDataset dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyBayerImageCropsTestDataloader(self):
        MAX_ITERS = 20
        pretime = time.time()
        ds = CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i, output in enumerate(ds.get_images()):
            self.assertEqual(output["x_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["y_crops"].shape, (1, 4, 128, 128))
            self.assertEqual(output["mask_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["rgb_xyz_matrix"].shape, (1, 4, 3))
            self.assertNotEqual(output["gain"], 1.0)
            if i >= MAX_ITERS:
                break
        print(
            f"Time to run {min(MAX_ITERS, i)} iterations of CleanProfiledRGBNoisyBayerImageCropsTestDataloader dataset: {time.time() - pretime}"
        )

    def test_CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(self):
        MAX_ITERS = 20
        pretime = time.time()
        ds = CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
            content_fpath=rawproc.RAWNIND_CONTENT_FPATH,
            crop_size=256,
            test_reserve=["MuseeL-Bobo-alt-A7C", "MuseeL-yombe-A7C"],
        )
        for i, output in enumerate(ds.get_images()):
            self.assertEqual(output["x_crops"].shape, (1, 3, 256, 256))
            self.assertEqual(output["mask_crops"].shape, (1, 3, 256, 256))
            self.assertNotEqual(output["gain"], 1.0)
            if i >= MAX_ITERS:
                break
        print(
            f"Time to run {min(MAX_ITERS, i)} iterations of CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader dataset: {time.time() - pretime}"
        )


if __name__ == "__main__":
    # the usual logging init
    import logging
    import os
    import sys

    LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"# python {' '.join(sys.argv)}")

    cleanRGB_noisyBayer_ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
        content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH], num_crops=4, crop_size=256
    )
    cleanRGB_noisyRGB_ds = CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(
        content_fpaths=[rawproc.RAWNIND_CONTENT_FPATH], num_crops=4, crop_size=256
    )