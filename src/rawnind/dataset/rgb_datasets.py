"""RGB dataset classes for profiled RGB image processing.

This module contains dataset classes specifically designed for handling profiled RGB
images, providing utilities for color space transformations and processing.
"""

import logging
import os
import sys
from typing import Optional

import torch

from .base_dataset import RawImageDataset


class ProfiledRGBProfiledRGBImageDataset(RawImageDataset):
    """Mixin for datasets that work with profiled RGB to profiled RGB transformations.

    This class provides a base for datasets where both input and target images
    are in profiled RGB color spaces, typically used for image enhancement
    or processing tasks that don't require color space conversion.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)