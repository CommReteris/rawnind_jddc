"""Bayer pattern dataset classes for raw image processing.

This module contains dataset classes specifically designed for handling Bayer pattern
images, including color space conversions and demosaicing operations.
"""

import logging
import os
import sys
from typing import Optional

import torch

from .base_dataset import RawImageDataset
# Import raw processing (will be moved to dependencies later)
from ..libs import raw


class ProfiledRGBBayerImageDataset(RawImageDataset):
    """Mixin for datasets that process Profiled RGB to Bayer transformations.

    Provides utility methods for converting camera RGB images to standard color
    profiles, which is essential for training models that work with Bayer patterns
    while maintaining color accuracy.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)

    @staticmethod
    def camRGB_to_profiledRGB_img(
            camRGB_img: torch.Tensor,
            metadata: dict,
            output_color_profile="lin_rec2020",
    ) -> torch.Tensor:
        return raw.camRGB_to_profiledRGB_img(camRGB_img, metadata, output_color_profile)