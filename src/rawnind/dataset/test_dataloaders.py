"""Test dataloader classes for dataset validation and evaluation.

This module contains dataloader classes specifically designed for testing and
validation purposes, providing efficient iteration over test datasets.
"""

import logging
import os
import sys
from typing import Optional

import torch

from .base_dataset import RawImageDataset
from .noisy_datasets import CleanProfiledRGBNoisyBayerImageCropsDataset, \
    CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
from ..dependencies.pytorch_helpers import fpath_to_tensor as pt_helpers
from ..dependencies.utilities import load_yaml as utilities
# Import raw processing (will be moved to dependencies later)
from ..libs import rawproc, arbitrary_proc_fun

# Constants from original rawds.py
ALIGNMENT_MAX_LOSS = 0.035
MASK_MEAN_MIN = 0.8


class TestDataLoader:
    """Mixin-like helper that yields processed images without using PyTorch DataLoader.

    Classes inheriting this should implement get_images(), which yields dictionaries
    with keys like x_crops, y_crops, mask_crops, and optionally rgb_xyz_matrix.
    """
    OUTPUTS_IMAGE_FILES = False

    def __init__(self, **kwargs):
        """Accept arbitrary keyword arguments for configuration; subclasses may consume them."""
        pass

    def __getitem__(self, i):
        """Disabled random access; use the iterator or get_images() instead."""
        raise TypeError(
            f"{type(self).__name__} is its own data loader: "
            "call get_images instead of __getitem__ (or use built-in __iter__)."
        )

    def __iter__(self):
        """Iterator alias for get_images()."""
        return self.get_images()

    def batched_iterator(self):
        """Yield batched tensors by adding a batch dimension when needed.

        If get_images() yields per-image tensors of shape [C,H,W], they are expanded
        to [1,C,H,W]; if they already include [N,C,H,W], they are passed through.
        """
        single_to_batch = lambda x: torch.unsqueeze(x, 0)
        identity = lambda x: x
        if hasattr(
                self, "get_images"
        ):  # TODO should combine this ifelse with an iterator selection
            for res in self.get_images():
                batch_fun = single_to_batch if res["y_crops"].dim() == 3 else identity
                res["y_crops"] = batch_fun(res["y_crops"]).float()
                res["x_crops"] = batch_fun(res["x_crops"]).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res
        else:
            for i in range(len(self._dataset)):
                res = self.__getitem__(i)
                batch_fun = single_to_batch if res["y_crops"].dim() == 3 else identity
                res["y_crops"] = batch_fun(res["y_crops"]).float()
                res["x_crops"] = batch_fun(res["x_crops"]).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res

    @staticmethod
    def _content_fpaths_to_test_reserve(content_fpaths: list[str]) -> list[str]:
        """Extract test reserve directory names from dataset content files.

        Parses YAML content files to extract directory names (excluding 'gt' directories)
        that should be reserved for testing purposes, ensuring proper train/test splits.

        Args:
            content_fpaths: List of paths to YAML files containing dataset metadata.

        Returns:
            List of directory names to reserve for testing.
        """
        # add all images to test_reserve:
        test_reserve = []
        for content_fpath in content_fpaths:
            for image in utilities.load_yaml(content_fpath, error_on_404=True):
                # get the directory name of the image (not the full path)
                dn = os.path.basename(os.path.dirname(image["f_fpath"]))
                if dn == "gt":
                    continue
                test_reserve.append(dn)
        return test_reserve


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