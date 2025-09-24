"""
Base dataset classes for raw image processing.

This module contains the foundational dataset classes extracted from
legacy_rawds.py, including base cropping logic, clean/clean and clean/noisy
dataset handling, and mask computation for overexposure clipping.
"""

import random
import logging
import os
import sys
import math
import time
from typing import Literal, NamedTuple, Optional, Union, TypedDict

import torch
import rawpy as rp


from ..dependencies import pytorch_helpers as pt_helpers
from ..dependencies.raw_processing import *

MAX_MASKED: float = 0.5  # Must ensure that we don't send a crop with this more than this many masked pixels
MAX_RANDOM_CROP_ATTEMPS = 10

MASK_MEAN_MIN = 0.8  # 14+11+1 = 26 images out of 1145 = 2.3 %
ALIGNMENT_MAX_LOSS = 0.035  # eliminates 6+3+2 + 1+4+2+6+3 = 27 images out of 1145 = 2.4 %
OVEREXPOSURE_LB = 0.99

TOY_DATASET_LEN = 25  # debug option


class RawDatasetOutput(TypedDict):
    x_crops: torch.Tensor
    y_crops: Optional[torch.Tensor]
    mask_crops: torch.BoolTensor
    rgb_xyz_matrix: Optional[torch.Tensor]
    gain: float


class RawImageDataset:
    """
    Base class for raw image datasets providing cropping functionality.
    """

    def __init__(self, num_crops: int, crop_size: int):
        self.num_crops = num_crops
        assert crop_size % 2 == 0
        self.crop_size = crop_size

    def random_crops(
        self,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        whole_img_mask: torch.BoolTensor,
    ):  # -> Union[tuple[torch.Tensor, Optional[torch.Tensor], torch.BoolTensor], bool]:
        """
        Crop an image into num_crops cs*cs crops without exceeding MAX_MASKED threshold.

        Returns x_crops, (optionally) y_crops, mask_crops
        """
        vdim, hdim = ximg.shape[-2:]
        max_start_v, max_start_h = vdim - self.crop_size, hdim - self.crop_size
        x_crops_dims = (self.num_crops, ximg.shape[-3], self.crop_size, self.crop_size)
        x_crops = torch.empty(x_crops_dims)
        mask_crops = torch.BoolTensor(x_crops.shape)
        if yimg is not None:
            assert rawproc.shape_is_compatible(
                ximg.shape, yimg.shape
            ), f"ximg and yimg should already be aligned. {ximg.shape=}, {yimg.shape=}"
            y_crops_dims = (
                self.num_crops,
                yimg.shape[-3],
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
            )
            y_crops = torch.empty(y_crops_dims)
        else:
            y_crops = None
        # mask_crops = torch.BoolTensor(self.num_crops, self.crop_size, self.crop_size)
        for crop_i in range(self.num_crops):
            # try a random crop
            self.make_a_random_crop(
                crop_i,
                x_crops,
                y_crops,
                mask_crops,
                max_start_v,
                max_start_h,
                ximg,
                yimg,
                whole_img_mask,
            )
            # ensure there are sufficient valid pixels
            attempts: int = 0
            while mask_crops[crop_i].sum() / self.crop_size**2 < MAX_MASKED:
                if attempts >= MAX_RANDOM_CROP_ATTEMPS:
                    return False
                self.make_a_random_crop(
                    crop_i,
                    x_crops,
                    y_crops,
                    mask_crops,
                    max_start_v,
                    max_start_h,
                    ximg,
                    yimg,
                    whole_img_mask,
                )
                attempts += 1
        if yimg is not None:
            return x_crops, y_crops, mask_crops
        return x_crops, mask_crops

    def make_a_random_crop(
        self,
        crop_i: int,
        x_crops: torch.Tensor,
        y_crops: Optional[torch.Tensor],
        mask_crops: torch.BoolTensor,
        max_start_v: int,
        max_start_h: int,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        whole_img_mask: torch.BoolTensor,
    ) -> None:
        """
        Make a random crop at specified index of ximg without validity check.

        Modifies x_crops[crop_i] and mask_crops[crop_i] in-place.
        """
        hstart = random.randrange(max_start_h)
        vstart = random.randrange(max_start_v)
        hstart -= hstart % 2  # maintain Bayer pattern
        vstart -= vstart % 2  # maintain Bayer pattern
        # print(
        #    f"{x_crops.shape=}, {ximg.shape=}, {vstart=}, {hstart=}, {self.crop_size=}, {max_start_h=}, {max_start_v=}"
        # )  # dbg
        x_crops[crop_i] = ximg[
            ..., vstart : vstart + self.crop_size, hstart : hstart + self.crop_size
        ]
        if yimg is not None:
            yimg_divisor = (yimg.shape[0] == 4) + 1
            y_crops[crop_i] = yimg[
                ...,
                vstart // yimg_divisor : vstart // yimg_divisor
                + self.crop_size // yimg_divisor,
                hstart // yimg_divisor : hstart // yimg_divisor
                + self.crop_size // yimg_divisor,
            ]
        mask_crops[crop_i] = whole_img_mask[
            ..., vstart : vstart + self.crop_size, hstart : hstart + self.crop_size
        ]

    def center_crop(
        self,
        ximg: torch.Tensor,
        yimg: Optional[torch.Tensor],
        mask: torch.BoolTensor,
    ):  # -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        height, width = ximg.shape[-2:]
        ystart = height // 2 - (self.crop_size // 2)
        xstart = width // 2 - (self.crop_size // 2)
        ystart -= ystart % 2
        xstart -= xstart % 2
        xcrop = ximg[
            ...,
            ystart : ystart + self.crop_size,
            xstart : xstart + self.crop_size,
        ]
        mask_crop = mask[
            ...,
            ystart : ystart + self.crop_size,
            xstart : xstart + self.crop_size,
        ]
        if yimg is not None:
            if yimg.size(-3) == 4:
                shape_divisor = 2
            elif yimg.size(-3) == 3:
                shape_divisor = 1
            else:
                raise ValueError(
                    f"center_crop: invalid number of channels: {yimg.size(-3)=}"
                )
            ycrop = yimg[
                ...,
                ystart // shape_divisor : ystart // shape_divisor
                + self.crop_size // shape_divisor,
                xstart // shape_divisor : xstart // shape_divisor
                + self.crop_size // shape_divisor,
            ]
            return xcrop, ycrop, mask_crop
        return xcrop, mask_crop


class CleanCleanImageDataset(RawImageDataset):
    """
    Dataset for clean-clean image pairs.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)

    def get_mask(self, ximg: torch.Tensor, metadata: dict) -> torch.BoolTensor:
        # we only ever apply the mask to RGB images so interpolate if Bayer
        if ximg.shape[0] == 4:
            ximg = torch.nn.functional.interpolate(
                ximg.unsqueeze(0), scale_factor=2
            ).squeeze(0)
            return (
                (ximg.max(0).values < metadata["overexposure_lb"])
                .unsqueeze(0)
                .repeat(3, 1, 1)
            )
        # because color transform has already been applied we can mask individual channels
        return ximg < metadata["overexposure_lb"]  # .all(-3)


class CleanNoisyDataset(RawImageDataset):
    """
    Base class for clean-noisy paired datasets.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self._dataset = []

    def __len__(self):
        return len(self._dataset)
    def __getitem__(self, i: int) -> RawDatasetOutput:
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        return len(self._dataset)