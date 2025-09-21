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
            output_color_profile=COLOR_PROFILE,
    ) -> torch.Tensor:
        return raw.camRGB_to_profiledRGB_img(camRGB_img, metadata, output_color_profile)


class ProfiledRGBProfiledRGBImageDataset(RawImageDataset):
    """Mixin for datasets that work with profiled RGB to profiled RGB transformations.

    This class provides a base for datasets where both input and target images
    are in profiled RGB color spaces, typically used for image enhancement
    or processing tasks that don't require color space conversion.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)


class CleanCleanImageDataset(RawImageDataset):
    """Base class for datasets containing clean image pairs for self-supervised learning.

    Used for training scenarios where the model learns from clean images without
    synthetic noise, relying on natural image variations or processing artifacts
    for supervision signal.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)

    def get_mask(self, ximg: torch.Tensor, metadata: dict) -> torch.BoolTensor:
        """Generate a boolean mask for valid (non-overexposed) pixels.

        Creates a mask that identifies pixels below the overexposure threshold.
        Handles both Bayer (4-channel) and RGB (3-channel) images by applying
        appropriate interpolation and thresholding strategies.

        Args:
            ximg: Input image tensor with shape (C, H, W) where C is 3 or 4.
            metadata: Dictionary containing 'overexposure_lb' threshold value.

        Returns:
            Boolean tensor indicating valid pixels (True = valid, False = overexposed).
        """
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
    """Base class for datasets containing clean-noisy image pairs for supervised denoising.

    This class provides a foundation for datasets where each sample consists of a clean
    (ground-truth) image paired with a corresponding noisy version. It's used for
    training supervised denoising models where the network learns to map from noisy
    inputs to clean targets.

    The class inherits cropping and masking utilities from RawImageDataset and maintains
    an internal dataset list that subclasses are expected to populate.
    """

    def __init__(self, num_crops: int, crop_size: int):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self._dataset = []

    def __len__(self):
        return len(self._dataset)


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
