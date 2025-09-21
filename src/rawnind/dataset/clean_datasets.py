"""Clean dataset implementations for self-supervised learning.

This module contains dataset classes for clean image pairs used in
self-supervised learning scenarios.

Extracted from rawds.py as part of the codebase refactoring.
"""

import logging
import random
from typing import NamedTuple

import torch
import tqdm

# Import base classes
from .base_dataset import CleanCleanImageDataset, ProfiledRGBBayerImageDataset, \
    ProfiledRGBProfiledRGBImageDataset, RawDatasetOutput
# Import from dependencies package (will be moved later)
from ..dependencies.pt_losses import losses, metrics
# Import raw processing (will be moved to dependencies later)
from ..libs import arbitrary_proc_fun

BREAKPOINT_ON_ERROR = True
COLOR_PROFILE = "lin_rec2020"
TOY_DATASET_LEN = 25  # debug option


class _ds_item(NamedTuple):
    overexposure_lb: float
    rgb_xyz_matrix: torch.Tensor
    crops: list[dict[str, str]]


class CleanProfiledRGBCleanBayerImageCropsDataset(
    CleanCleanImageDataset, ProfiledRGBBayerImageDataset
):
    """Dataset of clean pRGB targets with Bayer-space inputs from pre-cropped files.

    Uses metadata produced by tools/prep_image_dataset to sample crops and return
    tensors suitable for training pRGB<-Bayer models.
    """

    def __init__(
            self,
            content_fpaths: list[str],
            num_crops: int,
            crop_size: int,
            toy_dataset: bool = False,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.num_crops = num_crops
        self._dataset: list[_ds_item] = []
        for content_fpath in content_fpaths:
            logging.info(
                f"CleanProfiledRGBCleanBayerImageCropsDataset.__init__: loading {content_fpath}"
            )
            ds_content = utilities.load_yaml(content_fpath, error_on_404=True)
            for all_metadata in tqdm.tqdm(ds_content):
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                useful_metadata = {
                    "overexposure_lb": all_metadata["overexposure_lb"],
                    "rgb_xyz_matrix" : torch.tensor(all_metadata["rgb_xyz_matrix"]),
                    "crops"          : all_metadata["crops"],
                }
                if not useful_metadata["crops"]:
                    logging.warning(
                        f"CleanProfiledRGBCleanBayerImageCropsDataset.__init__: image {all_metadata} has no useful "
                        f"crops; not adding to dataset."
                    )
                else:
                    self._dataset.append(useful_metadata)
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(self, i: int) -> RawDatasetOutput:
        metadata = self._dataset[i]
        crop: dict[str, str] = random.choice(metadata["crops"])
        try:
            gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
            rgbg_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"]).float()
        except ValueError as e:
            logging.error(e)
            return self.__getitem__(random.randrange(len(self)))
        mask = self.get_mask(rgbg_img, metadata)
        try:
            x_crops, y_crops, mask_crops = self.random_crops(gt, rgbg_img, mask)
        except AssertionError as e:
            logging.info(crop)
            raise AssertionError(f"{self} {e} with {crop=}")
        except RuntimeError as e:
            logging.error(e)
            logging.error(f"{gt.shape=}, {rgbg_img.shape=}, {mask.shape=}")
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
        return {
            "x_crops"       : x_crops,
            "y_crops"       : y_crops,
            "mask_crops"    : mask_crops,
            "rgb_xyz_matrix": metadata["rgb_xyz_matrix"],
            "gain"          : 1.0,
        }

    def __len__(self) -> int:
        return len(self._dataset)


class CleanProfiledRGBCleanProfiledRGBImageCropsDataset(
    CleanCleanImageDataset, ProfiledRGBProfiledRGBImageDataset
):
    """Dataset for self-supervised learning from clean profiled RGB images.

    This dataset loads pre-cropped clean images in profiled RGB color space and applies
    random crops for training self-supervised models. The dataset supports optional
    arbitrary processing methods to simulate different processing pipelines during training.

    The dataset is built from metadata generated by tools/crop_dataset.py and
    tools/prep_image_dataset_extraraw.py, providing efficient access to pre-processed
    image crops without requiring full-size image loading at runtime.

    Used for training models that map from profiled RGB inputs to profiled RGB outputs,
    typically for image enhancement or style transfer tasks where clean images serve
    as both input and target through data augmentation strategies.
    """

    def __init__(
            self,
            content_fpaths: list[str],
            num_crops: int,
            crop_size: int,
            toy_dataset: bool = False,
            arbitrary_proc_method: bool = False,
    ):
        super().__init__(num_crops=num_crops, crop_size=crop_size)
        self.arbitrary_proc_method = arbitrary_proc_method
        self.num_crops = num_crops
        self._dataset: list[_ds_item] = []
        for content_fpath in content_fpaths:
            logging.info(
                f"CleanProfiledRGBCleanProfiledRGBImageCropsDataset.__init__: loading {content_fpath}"
            )
            ds_content = utilities.load_yaml(content_fpath, error_on_404=True)
            for all_metadata in tqdm.tqdm(ds_content):
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                useful_metadata = {
                    "overexposure_lb": all_metadata["overexposure_lb"],
                    "crops"          : all_metadata["crops"],
                }
                if not useful_metadata["crops"]:
                    logging.warning(
                        f"CleanProfiledRGBCleanProfiledRGBImageCropsDataset.__init__: image {all_metadata} has no "
                        f"useful crops; not adding to dataset."
                    )
                else:
                    self._dataset.append(useful_metadata)
        logging.info(f"initialized {type(self).__name__} with {len(self)} images.")
        if len(self) == 0:
            if BREAKPOINT_ON_ERROR:
                breakpoint()
            else:
                exit(-1)

    def __getitem__(self, i: int) -> RawDatasetOutput:
        metadata = self._dataset[i]
        crop: dict[str, str] = random.choice(metadata["crops"])
        try:
            gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
            rgbg_img = pt_helpers.fpath_to_tensor(
                crop["gt_bayer_fpath"]
            ).float()  # used to compute the overexposure mask
        except ValueError as e:
            logging.error(e)
            return self.__getitem__(random.randrange(len(self)))
        mask = self.get_mask(rgbg_img, metadata)
        if self.arbitrary_proc_method:
            gt = arbitrary_proc_fun.arbitrarily_process_images(
                gt,
                randseed=crop["gt_linrec2020_fpath"],
                method=self.arbitrary_proc_method,
            )
        try:
            x_crops, mask_crops = self.random_crops(gt, None, mask)
        except AssertionError as e:
            logging.info(crop)
            raise AssertionError(f"{self} {e} with {crop=}")
        except RuntimeError as e:
            logging.error(e)
            logging.error(f"{gt.shape=}, {rgbg_img.shape=}, {mask.shape=}")
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
        return {"x_crops": x_crops, "mask_crops": mask_crops, "gain": 1.0}

    def __len__(self) -> int:
        return len(self._dataset)