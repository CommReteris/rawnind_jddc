
"""
Noisy dataset classes for raw image processing.

This module contains noisy dataset classes for both Bayer and RGB image processing.
"""

import logging
import random
from typing import Literal, Optional

import torch

from rawnind.dependencies import pytorch_helpers as pt_helpers, raw_processing as rawproc, load_yaml
from .base_dataset import CleanNoisyDataset, RawDatasetOutput, TOY_DATASET_LEN
from .bayer_datasets import ProfiledRGBBayerImageDataset
from .rgb_datasets import ProfiledRGBProfiledRGBImageDataset
from .base_dataset import ALIGNMENT_MAX_LOSS, MASK_MEAN_MIN
from ..dependencies.arbitrary_processing import arbitrarily_process_images as arbitrary_proc_fun


class CleanProfiledRGBNoisyBayerImageCropsDataset(
    CleanNoisyDataset, ProfiledRGBBayerImageDataset
):
    """
    Dataset of clean-noisy raw images from rawNIND.

    Load from raw files using rawpy.
    Returns float crops, (highlight and anomaly) mask, metadata

    Alignment and masks are pre-computed.
    Output metadata contains color_matrix.
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
        data_pairing: Literal["x_y", "x_x", "y_y"] = "x_y",  # x_y, x_x, y_y
        match_gain: bool = False,
        min_msssim_score: Optional[float] = 0.0,
        max_msssim_score: Optional[float] = 1.0,
    ):
        """
        content_fpaths points to a yaml file containing:
            - best_alignment
            - f_bayer_fpath
            - gt_linrec2020_fpath
            - mask_fpath
            - best_alignment_loss
            - mask_mean

        return_data
        """
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.match_gain = match_gain
        assert bayer_only
        # contents: list[dict] = utilities.load_yaml(content_fpath)
        for content_fpath in content_fpaths:
            contents = load_yaml(
                content_fpath, error_on_404=True
            )  # python 3.8 incompat
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
                        f'{type(self).__name__}.__init__: rejected {image["f_fpath"]} (alignment or mask criteria)'
                    )
                    continue
                image["crops"] = sorted(
                    image["crops"], key=lambda d: d["coordinates"]
                )  # for testing
                if len(image["crops"]) > 0:
                    self._dataset.append(image)
                else:
                    logging.warning(
                        f'{type(self).__name__}.__init__: {image["f_fpath"]} has no crops.'
                    )
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        assert (
            len(self) > 0
        ), f"{type(self).__name__} has no images. {content_fpaths=}, {test_reserve=}"
        self.data_pairing = data_pairing

    def __getitem__(self, i: int) -> RawDatasetOutput:
        image: dict = self._dataset[i]
        # load x, y, mask
        crop = random.choice(image["crops"])
        if self.data_pairing == "x_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            # gt_img = self.crop_rgb_to_bayer(gt_img, metadata)

            # align x, y

            gt_img, noisy_img = rawproc.shift_images(
                gt_img, noisy_img, image["best_alignment"]
            )

            whole_img_mask = pt_helpers.fpath_to_tensor(image["mask_fpath"])[
                :,
                crop["coordinates"][1] : crop["coordinates"][1] + gt_img.shape[1],
                crop["coordinates"][0] : crop["coordinates"][0] + gt_img.shape[2],
            ]
            whole_img_mask = whole_img_mask.expand(gt_img.shape)
        elif self.data_pairing == "x_x":
            gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        elif self.data_pairing == "y_y":
            gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
            noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
            whole_img_mask = torch.ones_like(gt_img)
        else:
            raise ValueError(f"return_data={self.data_pairing} not supported")

        # crop x, y, mask, add alignment to mask
        try:
            x_crops, y_crops, mask_crops = self.random_crops(
                gt_img, noisy_img, whole_img_mask
            )
        except AssertionError as e:
            logging.info(crop)
            raise AssertionError(f"{self} {e} with {crop=}")
        except RuntimeError as e:
            logging.error(e)
            logging.error(f"{gt_img.shape=}, {noisy_img.shape=}, {whole_img_mask.shape=}")
            raise RuntimeError(f"{self} {e} with {crop=}")
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
        # hardcoded_rgbm = torch.tensor(
        #     [
        #         [0.7034, -0.0804, -0.1014],
        #         [-0.4420, 1.2564, 0.2058],
        #         [-0.0851, 0.1994, 0.5758],
        #         [0.0000, 0.0000, 0.0000],
        #     ]
        # )
        output = {
            "x_crops": x_crops,
            "y_crops": y_crops,
            "mask_crops": mask_crops,
            # "rgb_xyz_matrix": hardcoded_rgbm  # TODO RM DBG
            "rgb_xyz_matrix": torch.tensor(image["rgb_xyz_matrix"]),
        }
        if self.match_gain:
            output["y_crops"] *= image["raw_gain"]
            output["gain"] = 1.0