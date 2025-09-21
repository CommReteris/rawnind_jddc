"""Noisy dataset implementations for supervised denoising.

This module contains dataset classes for clean-noisy image pairs used in
supervised denoising training scenarios.

Extracted from rawds.py as part of the codebase refactoring.
"""

import logging
import random
from typing import Literal, Optional

import torch

# Import base classes
from .base_dataset import CleanNoisyDataset, ProfiledRGBBayerImageDataset, \
    ProfiledRGBProfiledRGBImageDataset, RawDatasetOutput
# Import from dependencies package (will be moved later)
from ..dependencies.pt_losses import losses, metrics
# Import raw processing (will be moved to dependencies later)
from ..libs import arbitrary_proc_fun, rawproc

BREAKPOINT_ON_ERROR = True
COLOR_PROFILE = "lin_rec2020"
TOY_DATASET_LEN = 25  # debug option

# Constants from original rawds.py
MAX_MASKED: float = 0.5
MAX_RANDOM_CROP_ATTEMPS = 10
MASK_MEAN_MIN = 0.8
ALIGNMENT_MAX_LOSS = 0.035
OVEREXPOSURE_LB = 0.99


class CleanProfiledRGBNoisyBayerImageCropsDataset(
    CleanNoisyDataset, ProfiledRGBBayerImageDataset
):
    """Dataset for supervised denoising training with clean profiled RGB targets and noisy Bayer inputs.

    This dataset pairs clean profiled RGB ground-truth images with corresponding noisy Bayer
    pattern inputs from the rawNIND dataset. It's designed for training supervised denoising
    models that learn to map from noisy Bayer space to clean profiled RGB output.

    The dataset supports advanced filtering based on alignment quality, mask coverage,
    and MS-SSIM scores to ensure high-quality training pairs. It provides flexible
    data pairing modes (clean-noisy, clean-clean, noisy-noisy) and optional gain
    matching for consistent exposure levels.

    Key features:
    - Pre-computed alignment and masking for efficiency
    - MSSSIM-based quality filtering
    - Support for train/test splits via reserved image sets
    - Multiple data pairing strategies for different training scenarios
    - Integrated color matrix metadata for proper color space transformations
    """

    def __init__(
            self,
            content_fpaths: list[str],
            num_crops: int,
            crop_size: int,
            test_reserve: list,
            bayer_only: bool = True,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            test: bool = False,
            toy_dataset: bool = False,
            data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
            match_gain: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.match_gain = match_gain
        assert bayer_only

        for content_fpath in content_fpaths:
            contents = utilities.load_yaml(content_fpath, error_on_404=True)
            for image in contents:
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                if not image["is_bayer"]:
                    continue

                # check that the image is (/not) reserved for testing
                if (not test and image["image_set"] in test_reserve) or (
                        test and image["image_set"] not in test_reserve
                ):
                    continue
                try:
                    if (
                            min_msssim_score
                            and min_msssim_score > image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} < {min_msssim_score}"
                        )
                        continue
                    if (
                            max_msssim_score
                            and max_msssim_score != 1.0
                            and max_msssim_score < image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} > {max_msssim_score}"
                        )
                        continue
                except KeyError:
                    raise KeyError(
                        f"{image} does not contain msssim score (required with {min_msssim_score=})"
                    )
                if (
                        image["best_alignment_loss"] > alignment_max_loss
                        or image["mask_mean"] < mask_mean_min
                ):
                    logging.info(
                        f"{type(self).__name__}.__init__: rejected {image['f_fpath']}"
                    )
                    continue
                image["crops"] = sorted(
                    image["crops"], key=lambda d: d["coordinates"]
                )
                if len(image["crops"]) > 0:
                    self._dataset.append(image)
                else:
                    logging.warning(
                        f"{type(self).__name__}.__init__: {image['f_fpath']} has no crops."
                    )
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        assert len(self) > 0, (
            f"{type(self).__name__} has no images. {content_fpaths=}, {test_reserve=}"
        )
        self.data_pairing = data_pairing

    def __getitem__(self, i: int) -> RawDatasetOutput:
        image: dict = self._dataset[i]
        crop = random.choice(image["crops"])
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
        else:
            raise ValueError(f"return_data={self.data_pairing} not supported")

        try:
            x_crops, y_crops, mask_crops = self.random_crops(
                gt_img, noisy_img, whole_img_mask
            )
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)

        output = {
            "x_crops"       : x_crops.float(),
            "y_crops"       : y_crops.float(),
            "mask_crops"    : mask_crops,
            "rgb_xyz_matrix": torch.tensor(image["rgb_xyz_matrix"]),
        }
        if self.match_gain:
            output["y_crops"] *= image["raw_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["raw_gain"]
        return output


class CleanProfiledRGBNoisyProfiledRGBImageCropsDataset(
    CleanNoisyDataset, ProfiledRGBProfiledRGBImageDataset
):
    """Dataset for supervised denoising training with profiled RGB inputs and targets.

    This dataset provides clean-noisy pairs of demosaiced profiled RGB images from the
    rawNIND dataset. It's designed for training supervised denoising models that work
    entirely in profiled RGB color space, avoiding the complexity of Bayer pattern
    processing while maintaining realistic noise characteristics.

    The dataset supports multiple data pairing strategies (clean-noisy, clean-clean,
    noisy-noisy) for different training scenarios and includes optional arbitrary
    processing methods for data augmentation. All images are pre-aligned and masked
    for optimal training quality.

    Key features:
    - Profiled RGB input/output for simplified color processing
    - Pre-computed alignment and masking from rawNIND pipeline
    - Multiple data pairing modes for flexible training strategies
    - Optional arbitrary processing for enhanced data augmentation
    - MS-SSIM quality filtering for consistent training pairs
    - Support for gain matching and exposure normalization
    """

    def __init__(
            self,
            content_fpaths: list[str],
            num_crops: int,
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = ALIGNMENT_MAX_LOSS,
            mask_mean_min: float = MASK_MEAN_MIN,
            test: bool = False,
            toy_dataset: bool = False,
            data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",
            match_gain: bool = False,
            arbitrary_proc_method: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.match_gain = match_gain
        self.arbitrary_proc_method = arbitrary_proc_method
        if self.arbitrary_proc_method:
            assert self.match_gain, (
                f"{type(self).__name__}: arbitrary_proc_method requires match_gain"
            )
        self.data_pairing = data_pairing

        for content_fpath in content_fpaths:
            contents = utilities.load_yaml(content_fpath, error_on_404=True)
            for image in contents:
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break

                # check that the image is (/not) reserved for testing
                if (
                        (not test and image["image_set"] in test_reserve)
                        or (test and image["image_set"] not in test_reserve)
                        or (  # check that there is a bayer version available if bayer_only is True
                        bayer_only and not image["is_bayer"]
                )
                ):
                    continue
                try:
                    if (
                            min_msssim_score
                            and min_msssim_score > image["rgb_msssim_score"]
                    ):
                        continue
                    if (
                            max_msssim_score
                            and max_msssim_score != 1.0
                            and max_msssim_score < image["rgb_msssim_score"]
                    ):
                        print(
                            f"Skipping {image['f_fpath']} with {image['rgb_msssim_score']} > {max_msssim_score}"
                        )
                        continue
                except KeyError:
                    raise KeyError(
                        f"{image} does not contain msssim score (required with {min_msssim_score=})"
                    )

                if (
                        image["best_alignment_loss"] > alignment_max_loss
                        or image["mask_mean"] < mask_mean_min
                ):
                    logging.info(
                        f"{type(self).__name__}.__init__: rejected {image['f_fpath']}"
                    )
                    continue
                image["crops"] = sorted(
                    image["crops"], key=lambda d: d["coordinates"]
                )
                if len(image["crops"]) > 0:
                    self._dataset.append(image)
                else:
                    logging.warning(
                        f"{type(self).__name__}.__init__: {image['f_fpath']} has no crops."
                    )
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(self, i: int):
        """Returns a random crop triplet (ximage, yimage, mask).

        Args:
            i (int): Image index

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]: random crop triplet
        """
        image = self._dataset[i]
        crop = random.choice(image["crops"])
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
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            whole_img_mask = torch.ones_like(gt_img)

        output = {}
        if self.match_gain:
            noisy_img *= image["rgb_gain"]
            output["gain"] = 1.0
        else:
            output["gain"] = image["rgb_gain"]
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

        try:
            x_crops, y_crops, mask_crops = self.random_crops(
                gt_img, noisy_img, whole_img_mask
            )
        except TypeError:
            logging.warning(
                f"{crop} does not contain sufficient valid pixels; removing from dataset"
            )
            self._dataset[i]["crops"].remove(crop)
            if len(self._dataset[i]["crops"]) == 0:
                logging.warning(
                    f"{self._dataset[i]} does not contain anymore valid crops. Removing whole image from dataset."
                )
                self._dataset.remove(self._dataset[i])
            return self.__getitem__(i)

        output["x_crops"] = x_crops.float()
        output["y_crops"] = y_crops.float()
        output["mask_crops"] = mask_crops

        return output
