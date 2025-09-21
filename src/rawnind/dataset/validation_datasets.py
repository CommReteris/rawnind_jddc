"""Validation and test dataset implementations.

This module contains dataset classes for validation and testing scenarios,
including center-crop datasets and test dataloaders.

Extracted from rawds.py as part of the codebase refactoring.
"""

import random
import logging
import os
import sys
import math
import time
import unittest
from typing import Literal, NamedTuple, Optional, Union, TypedDict
import tqdm

import torch

# Import from dependencies package (will be moved later)
from ..dependencies.pytorch_helpers import get_device
from ..dependencies.utilities import dict_to_yaml, load_yaml
from ..dependencies.pt_losses import losses, metrics

# Import raw processing (will be moved to dependencies later)
from ..libs import raw, rawproc, arbitrary_proc_fun

# Import base classes
from .base_dataset import RawImageDataset, TestDataLoader
from .noisy_datasets import CleanProfiledRGBNoisyBayerImageCropsDataset, \
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset

BREAKPOINT_ON_ERROR = True
COLOR_PROFILE = "lin_rec2020"
TOY_DATASET_LEN = 25  # debug option

# Constants from original rawds.py
MAX_MASKED: float = 0.5
MAX_RANDOM_CROP_ATTEMPS = 10
MASK_MEAN_MIN = 0.8
ALIGNMENT_MAX_LOSS = 0.035
OVEREXPOSURE_LB = 0.99


class CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset(
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
):
    """Validation dataset for profiled RGB denoising models.

    This class extends the training dataset to provide deterministic center crops
    for validation and evaluation purposes. Instead of random crops, it consistently
    selects the middle crop from each image's crop list, ensuring reproducible
    validation metrics across training runs.

    The validation dataset supports the same data pairing modes and processing
    options as the training dataset but provides single crops per image rather
    than multiple random crops, making it suitable for consistent model evaluation
    during training and final performance assessment.
    """

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            toy_dataset: bool = False,
            match_gain: bool = False,
            arbitrary_proc_method: bool = False,
            data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            arbitrary_proc_method=arbitrary_proc_method,
            data_pairing=data_pairing,
        )

    def __getitem__(self, i: int):
        """Returns a center crop triplet (ximage, yimage, mask).

        Args:
            i (int): Image index

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]: center crop triplet
        """
        image: dict = self._dataset[i]
        crop_n = len(image["crops"]) // 2
        crop = image["crops"][crop_n]

        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])

            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1]: crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0]: crop["coordinates"][0] + gt_img.shape[2],
            ]
            try:
                whole_img_mask = whole_img_mask.expand(gt_img.shape)
            except RuntimeError as e:
                logging.error(e)
                breakpoint()
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)

        if self.crop_size == 0:
            height, width = gt_img.shape[-2:]
            height = height - height % 256
            width = width - width % 256
            min_crop_size = 256
            x_crop = gt_img[..., :height, :width]
            noisy_img = y_crop = noisy_img[..., :height, :width]
            whole_img_mask = mask_crop = whole_img_mask[..., :height, :width]
        else:
            min_crop_size = self.crop_size
            x_crop, y_crop, mask_crop = self.center_crop(
                gt_img, noisy_img, whole_img_mask
            )
        if x_crop.shape[-1] < min_crop_size or x_crop.shape[-2] < min_crop_size:
            logging.warning(
                f"CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset.__getitem__: not enough pixels in "
                f"{crop['gt_linrec2020_fpath']}; deleting from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            return self.__getitem__(i)

        output = {
            "x_crops"   : x_crop.float(),
            "y_crops"   : y_crop.float(),
            "mask_crops": mask_crop,
            "gt_fpath"  : crop["gt_linrec2020_fpath"],
            "y_fpath"   : crop["f_linrec2020_fpath"],
        }
        if self.match_gain:
            output["y_crops"] *= image["rgb_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["rgb_gain"]
        if self.arbitrary_proc_method:
            output["x_crops"] = arbitrary_proc_fun.arbitrarily_process_images(
                output["x_crops"],
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
            output["y_crops"] = arbitrary_proc_fun.arbitrarily_process_images(
                output["y_crops"],
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
        return output


class CleanProfiledRGBNoisyBayerImageCropsValidationDataset(
    CleanProfiledRGBNoisyBayerImageCropsDataset
):
    """Dataset of clean (profiled RGB) - noisy (Bayer) images from rawNIND."""

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            toy_dataset=False,
            match_gain: bool = False,
            data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            data_pairing=data_pairing,
        )

    def __getitem__(self, i):
        image: dict = self._dataset[i]
        crop_n = len(image["crops"]) // 2
        crop = image["crops"][crop_n]

        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])

            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )
            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1]: crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0]: crop["coordinates"][0] + gt_img.shape[2],
            ]
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)

        if self.crop_size == 0:
            height, width = gt_img.shape[-2:]
            height = height - height % 256
            width = width - width % 256
            min_crop_size = 256
            x_crop = gt_img[..., :height, :width]
            noisy_img = y_crop = noisy_img[..., : height // 2, : width // 2]
            whole_img_mask = mask_crop = whole_img_mask[..., :height, :width]
        else:
            min_crop_size = self.crop_size
            x_crop, y_crop, mask_crop = self.center_crop(
                gt_img, noisy_img, whole_img_mask
            )
        if x_crop.shape[-1] < min_crop_size or x_crop.shape[-2] < min_crop_size:
            logging.warning(
                f"CleanProfiledRGBNoisyBayerImageCropsValidationDataset.__getitem__: not enough pixels in {crop['gt_linrec2020_fpath']}; deleting from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            return self.__getitem__(i)
        output = {
            "x_crops"       : x_crop.float(),
            "y_crops"       : y_crop.float(),
            "mask_crops"    : mask_crop,
            "rgb_xyz_matrix": torch.tensor(image["rgb_xyz_matrix"]),
            "gt_fpath"      : crop["gt_linrec2020_fpath"],
            "y_fpath"       : crop["f_bayer_fpath"],
        }
        if self.match_gain:
            output["y_crops"] *= image["raw_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["raw_gain"]
        return output


class CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
    CleanProfiledRGBNoisyBayerImageCropsDataset, TestDataLoader
):
    """Dataloader of clean (profiled RGB) - noisy (Bayer) images crops from rawNIND."""

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            toy_dataset=False,
            match_gain: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            min_msssim_score=min_msssim_score,
            max_msssim_score=max_msssim_score,
        )

    def get_images(self):
        """Yield test images one crop at a time. Replaces __getitem__ s.t. the image is not re-loaded many times."""
        for image in self._dataset:
            rgb_xyz_matrix = torch.tensor(image["rgb_xyz_matrix"])
            for crop in image["crops"]:
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"]).float()

                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image["best_alignment"]
                )
                whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                    :,
                    crop["coordinates"][1]: crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0]: crop["coordinates"][0] + gt_img.shape[2],
                ].expand(gt_img.shape)

                height, width = gt_img.shape[-2:]
                if self.match_gain:
                    noisy_img *= image["raw_gain"]
                    out_gain = 1.0
                else:
                    out_gain = image["raw_gain"]
                if self.crop_size == 0:
                    height = height - height % 256
                    width = width - width % 256
                    if height == 0 or width == 0:
                        continue
                    yield (
                        {
                            "x_crops"       : gt_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "y_crops"       : noisy_img[
                                ...,
                                : height // 2,
                                : width // 2,
                            ].unsqueeze(0),
                            "mask_crops"    : whole_img_mask[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "rgb_xyz_matrix": rgb_xyz_matrix.unsqueeze(0),
                            "gt_fpath"      : crop["gt_linrec2020_fpath"],
                            "y_fpath"       : crop["f_bayer_fpath"],
                            "gain"          : torch.tensor(out_gain),
                        }
                    )
                else:
                    y = x = 0
                    while y < height:
                        while x < width:
                            if (
                                    y + self.crop_size <= height
                                    and x + self.crop_size <= width
                            ):
                                yield (
                                    {
                                        "x_crops"       : gt_img[
                                            ...,
                                            y: y + self.crop_size,
                                            x: x + self.crop_size,
                                        ].unsqueeze(0),
                                        "y_crops"       : noisy_img[
                                            ...,
                                            y // 2: y // 2 + self.crop_size // 2,
                                            x // 2: x // 2 + self.crop_size // 2,
                                        ].unsqueeze(0),
                                        "mask_crops"    : whole_img_mask[
                                            ...,
                                            y: y + self.crop_size,
                                            x: x + self.crop_size,
                                        ].unsqueeze(0),
                                        "rgb_xyz_matrix": rgb_xyz_matrix.unsqueeze(0),
                                        "gt_fpath"      : crop["gt_linrec2020_fpath"],
                                        "y_fpath"       : crop["f_bayer_fpath"],
                                        "gain"          : torch.tensor(out_gain),
                                    }
                                )
                            x += self.crop_size
                        x = 0
                        y += self.crop_size


class CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset, TestDataLoader
):
    """Dataloader of clean (profiled RGB) - noisy (profiled RGB) images crops from rawNIND."""

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            toy_dataset=False,
            match_gain: bool = False,
            arbitrary_proc_method: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(
            content_fpaths=content_fpaths,
            num_crops=1,
            crop_size=crop_size,
            test_reserve=test_reserve,
            alignment_max_loss=alignment_max_loss,
            mask_mean_min=mask_mean_min,
            test=True,
            bayer_only=bayer_only,
            toy_dataset=toy_dataset,
            match_gain=match_gain,
            arbitrary_proc_method=arbitrary_proc_method,
            min_msssim_score=min_msssim_score,
            max_msssim_score=max_msssim_score,
        )

    def get_images(self):
        """Yield test images one crop at a time. Replaces __getitem__ s.t. the image is not re-loaded many times."""
        for image in self._dataset:
            for crop in image["crops"]:
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                noisy_img = pt_helpers.fpath_to_tensor(
                    crop["f_linrec2020_fpath"]
                ).float()

                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image["best_alignment"]
                )
                whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                    :,
                    crop["coordinates"][1]: crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0]: crop["coordinates"][0] + gt_img.shape[2],
                ].expand(gt_img.shape)
                height, width = gt_img.shape[-2:]
                if self.match_gain:
                    noisy_img *= image["rgb_gain"]
                    out_gain = 1.0
                else:
                    out_gain = image["rgb_gain"]

                if self.arbitrary_proc_method:
                    gt_img = arbitrary_proc_fun.arbitrarily_process_images(
                        gt_img,
                        randseed=crop["gt_linrec2020_fpath"],
                        method=self.arbitrary_proc_method,
                    )
                    noisy_img = arbitrary_proc_fun.arbitrarily_process_images(
                        noisy_img,
                        randseed=crop["gt_linrec2020_fpath"],
                        method=self.arbitrary_proc_method,
                    )
                if self.crop_size == 0:
                    height = height - height % 256
                    width = width - width % 256
                    if height == 0 or width == 0:
                        continue
                    yield (
                        {
                            "x_crops"   : gt_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "y_crops"   : noisy_img[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "mask_crops": whole_img_mask[
                                ...,
                                :height,
                                :width,
                            ].unsqueeze(0),
                            "gt_fpath"  : crop["gt_linrec2020_fpath"],
                            "y_fpath"   : crop["f_linrec2020_fpath"],
                            "gain"      : torch.tensor(out_gain),
                        }
                    )
                else:
                    x = y = 0
                    while y < height:
                        while x < width:
                            if (
                                    y + self.crop_size <= height
                                    and x + self.crop_size <= width
                            ):
                                yield (
                                    {
                                        "x_crops"   : gt_img[
                                            ...,
                                            y: y + self.crop_size,
                                            x: x + self.crop_size,
                                        ].unsqueeze(0),
                                        "y_crops"   : noisy_img[
                                            ...,
                                            y: y + self.crop_size,
                                            x: x + self.crop_size,
                                        ].unsqueeze(0),
                                        "mask_crops": whole_img_mask[
                                            ...,
                                            y: y + self.crop_size,
                                            x: x + self.crop_size,
                                        ].unsqueeze(0),
                                        "gt_fpath"  : crop["gt_linrec2020_fpath"],
                                        "y_fpath"   : crop["f_linrec2020_fpath"],
                                        "gain"      : torch.tensor(out_gain),
                                    }
                                )
                            x += self.crop_size
                        x = 0
                        y += self.crop_size