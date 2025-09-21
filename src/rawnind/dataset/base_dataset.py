"""Base dataset classes and utilities.

This module contains the core dataset functionality extracted from
rawds.py, including base classes for image datasets and common utilities.

Extracted from rawds.py as part of the codebase refactoring.
"""

import os
import random
from typing import Optional, TypedDict

import torch

# Import from dependencies package (will be moved later)
from ..dependencies.pt_losses import losses, metrics
# Import raw processing (will be moved to dependencies later)
from ..libs import rawproc

BREAKPOINT_ON_ERROR = True

COLOR_PROFILE = "lin_rec2020"
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")

MAX_MASKED: float = 0.5  # Must ensure that we don't send a crop with this more than this many masked pixels
MAX_RANDOM_CROP_ATTEMPS = 10

MASK_MEAN_MIN = 0.8  # 14+11+1 = 26 images out of 1145 = 2.3 %
ALIGNMENT_MAX_LOSS = (
    0.035  # eliminates 6+3+2 + 1+4+2+6+3 = 27 images out of 1145 = 2.4 %
)
OVEREXPOSURE_LB = 0.99

TOY_DATASET_LEN = 25  # debug option


class RawDatasetOutput(TypedDict):
    """Output format for raw dataset items."""
    x_crops: torch.Tensor
    y_crops: Optional[torch.Tensor]
    mask_crops: torch.BoolTensor
    rgb_xyz_matrix: Optional[torch.Tensor]
    gain: float


class RawImageDataset:
    """Base class for patch-based datasets built from RAW/processed images.

    The class provides utilities to sample multiple random crops from a pair of
    aligned images and to ensure masks contain sufficient valid pixels.

    Subclasses are expected to implement data loading and masking policy by
    overriding __getitem__ and optionally get_mask-like helpers.
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
    ):
        """Extract multiple random crops from input images while ensuring sufficient valid pixels.

        Generates random spatial crops from input images, ensuring each crop has enough
        valid (unmasked) pixels for training. Maintains Bayer pattern alignment by ensuring
        crop coordinates are even. Handles both single-image (denoising) and paired-image
        (clean-noisy) scenarios.

        Args:
            ximg: Input image tensor with shape (..., C, H, W).
            yimg: Optional target image tensor. If provided, must be shape-compatible with ximg.
            whole_img_mask: Boolean mask indicating valid pixels, shape (..., C, H, W).

        Returns:
            If successful: tuple of (x_crops, y_crops, mask_crops) where y_crops is None
            if yimg is None, or (x_crops, mask_crops) if yimg is None.
            If failed: False when unable to find sufficient valid pixels after MAX_RANDOM_CROP_ATTEMPS.
        """
        vdim, hdim = ximg.shape[-2:]
        max_start_v, max_start_h = vdim - self.crop_size, hdim - self.crop_size
        x_crops_dims = (self.num_crops, ximg.shape[-3], self.crop_size, self.crop_size)
        x_crops = torch.empty(x_crops_dims)
        mask_crops = torch.BoolTensor(x_crops.shape)
        if yimg is not None:
            assert rawproc.shape_is_compatible(ximg.shape, yimg.shape), (
                f"ximg and yimg should already be aligned. {ximg.shape=}, {yimg.shape=}"
            )
            y_crops_dims = (
                self.num_crops,
                yimg.shape[-3],
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
                self.crop_size // ((yimg.shape[-3] == 4) + 1),
            )
            y_crops = torch.empty(y_crops_dims)
        else:
            y_crops = None

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
            while mask_crops[crop_i].sum() / self.crop_size ** 2 < MAX_MASKED:
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
        """Extract a single random crop and store it at the specified index.

        Randomly selects crop coordinates and extracts a crop_size×crop_size region
        from the input images and mask. Maintains Bayer pattern alignment by ensuring
        coordinates are even. Handles different channel counts between input and target.

        Args:
            crop_i: Index where to store the extracted crop in the output tensors.
            x_crops: Output tensor for input image crops.
            y_crops: Output tensor for target image crops (None if not provided).
            mask_crops: Output tensor for mask crops.
            max_start_v: Maximum valid vertical start coordinate.
            max_start_h: Maximum valid horizontal start coordinate.
            ximg: Source input image tensor.
            yimg: Source target image tensor (None if single-image scenario).
            whole_img_mask: Source mask tensor.

        Note:
            Modifies x_crops[crop_i], y_crops[crop_i] (if applicable), and
            mask_crops[crop_i] in-place without performing validity checks.
        """
        hstart = random.randrange(max_start_h)
        vstart = random.randrange(max_start_v)
        hstart -= hstart % 2  # maintain Bayer pattern
        vstart -= vstart % 2  # maintain Bayer pattern

        x_crops[crop_i] = ximg[
            ..., vstart: vstart + self.crop_size, hstart: hstart + self.crop_size
        ]
        if yimg is not None:
            yimg_divisor = (yimg.shape[0] == 4) + 1
            y_crops[crop_i] = yimg[
                ...,
                vstart // yimg_divisor: vstart // yimg_divisor
                                        + self.crop_size // yimg_divisor,
                hstart // yimg_divisor: hstart // yimg_divisor
                                        + self.crop_size // yimg_divisor,
            ]
        mask_crops[crop_i] = whole_img_mask[
            ..., vstart: vstart + self.crop_size, hstart: hstart + self.crop_size
        ]

    def center_crop(
            self,
            ximg: torch.Tensor,
            yimg: Optional[torch.Tensor],
            mask: torch.BoolTensor,
    ):
        """Extract a center crop from input images and mask.

        Takes a crop from the center of the input images, maintaining Bayer pattern
        alignment by ensuring coordinates are even. Handles different channel counts
        between input and target images for various dataset scenarios.

        Args:
            ximg: Input image tensor with shape (..., C, H, W).
            yimg: Optional target image tensor. May have different channel count than ximg.
            mask: Boolean mask tensor indicating valid pixels.

        Returns:
            If yimg is provided: tuple (xcrop, ycrop, mask_crop)
            If yimg is None: tuple (xcrop, mask_crop)
        """
        height, width = ximg.shape[-2:]
        ystart = height // 2 - (self.crop_size // 2)
        xstart = width // 2 - (self.crop_size // 2)
        ystart -= ystart % 2
        xstart -= xstart % 2
        xcrop = ximg[
            ...,
            ystart: ystart + self.crop_size,
            xstart: xstart + self.crop_size,
        ]
        mask_crop = mask[
            ...,
            ystart: ystart + self.crop_size,
            xstart: xstart + self.crop_size,
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
                ystart // shape_divisor: ystart // shape_divisor
                                         + self.crop_size // shape_divisor,
                xstart // shape_divisor: xstart // shape_divisor
                                         + self.crop_size // shape_divisor,
            ]
            return xcrop, ycrop, mask_crop
        return xcrop, mask_crop
