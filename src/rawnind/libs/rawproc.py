"""Utilities for RAW image post-processing used across the project.

This module groups small, self-contained numerical utilities for:
- tone transfer functions (scene-linear <-> PQ)
- gain matching and alignment between noisy/clean pairs
- mask shifting and loss-mask creation
- simple color conversions used by the training/evaluation code

The goal is to keep these helpers pure and framework-agnostic when possible,
while supporting both NumPy ndarrays and torch Tensors for convenience.

All functions are intended to be side-effect free unless explicitly documented.
"""

import os
import shutil
import subprocess
import unittest
from typing import Union

import colour  # colour-science, needed for the PQ OETF(-1) transfer function
import numpy as np
import scipy.ndimage
import torch

# 
from common.libs import np_imgops
from rawnind.libs import raw

# LOSS_THRESHOLD: float = 0.33
LOSS_THRESHOLD: float = 0.4
GT_OVEREXPOSURE_LB: float = 1.0
KEEPERS_QUANTILE: float = 0.9999
MAX_SHIFT_SEARCH: int = 128
GAMMA = 2.2
DS_DN = "RawNIND"
DATASETS_ROOT = os.path.join("..", "..", "datasets")
DS_BASE_DPATH: str = os.path.join(DATASETS_ROOT, DS_DN)
BAYER_DS_DPATH: str = os.path.join(DS_BASE_DPATH, "src", "Bayer")
LINREC2020_DS_DPATH: str = os.path.join(DS_BASE_DPATH, "proc", "lin_rec2020")
MASKS_DPATH = os.path.join(DS_BASE_DPATH, f"masks_{LOSS_THRESHOLD}")
RAWNIND_CONTENT_FPATH = os.path.join(
    DS_BASE_DPATH, "RawNIND_masks_and_alignments.yaml"
)  # used by tools/prep_image_dataset.py and libs/rawds.py

NEIGHBORHOOD_SEARCH_WINDOW = 3
EXTRARAW_DS_DPATH = os.path.join("..", "..", "datasets", "extraraw")
EXTRARAW_CONTENT_FPATHS = (
    os.path.join(EXTRARAW_DS_DPATH, "trougnouf", "crops_metadata.yaml"),
    os.path.join(EXTRARAW_DS_DPATH, "raw-pixls", "crops_metadata.yaml"),
    # os.path.join(EXTRARAW_DS_DPATH, "SID", "crops_metadata.yaml"), # could be useful for testing
)


def np_l1(img1: np.ndarray, img2: np.ndarray, avg: bool = True) -> Union[float, np.ndarray]:
    """Compute per-element L1 distance between two images.

    Args:
        img1: First image (NumPy array) of identical shape as img2.
        img2: Second image (NumPy array) of identical shape as img1.
        avg: If True, return the mean L1 value over all elements; otherwise return the element-wise map.

    Returns:
        A scalar float if avg is True, otherwise a NumPy array of absolute differences with the same shape as inputs.
    """
    if avg:
        return np.abs(img1 - img2).mean()
    return np.abs(img1 - img2)


def gamma(img: np.ndarray, gamma_val: float = GAMMA, in_place: bool = False) -> np.ndarray:
    """Apply gamma correction to a NumPy image.

    Only strictly positive values are gamma-encoded; non-positive values are preserved
    as-is to avoid creating NaNs when operating on linear-light data that may contain
    small negative values (e.g., after filtering).

    Args:
        img: Input NumPy array. Broadcastable operations are applied element-wise.
        gamma_val: Gamma exponent to apply (default 2.2). Effective transform is x**(1/gamma).
        in_place: If True, modify the input array in place; otherwise operate on a copy.

    Returns:
        NumPy array with gamma applied to positive entries.
    """
    res = img if in_place else img.copy()
    res[res > 0] = res[res > 0] ** (1 / gamma_val)
    return res


def gamma_pt(img: torch.Tensor, gamma_val: float = GAMMA, in_place: bool = False) -> torch.Tensor:
    """Apply gamma correction to a torch Tensor.

    Mirrors gamma() but operates on torch tensors and preserves device/dtype.
    Only strictly positive values are gamma-encoded; non-positive values are preserved.

    Args:
        img: Input tensor.
        gamma_val: Gamma exponent to apply (default 2.2). Effective transform is x**(1/gamma).
        in_place: If True, modify the tensor in place; otherwise operate on a clone.

    Returns:
        Tensor with gamma applied to positive entries.
    """
    res = img if in_place else img.clone()
    res[res > 0] = res[res > 0] ** (1 / gamma_val)
    return res


def scenelin_to_pq(
        img: Union[np.ndarray, torch.Tensor], compat=True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Scene linear input signal to PQ opto-electronic transfer function (OETF).
    See also:
        https://en.wikipedia.org/wiki/Perceptual_quantizer
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2100-2-201807-I!!PDF-E.pdf
        https://www.itu.int/dms_pubrec/itu-r/rec/bt/R-REC-BT.2124-0-201901-I!!PDF-E.pdf
    https://github.com/colour-science/colour/blob/develop/colour/models/rgb/transfer_functions/itur_bt_2100.py
    : oetf_BT2100_PQ
    """
    if isinstance(img, np.ndarray):
        # in develop branch: oetf_BT2100_PQ
        return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ(img)
    elif isinstance(img, torch.Tensor):
        # translation of colour.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ
        # into PyTorch
        def spow(a, p):
            a_p = torch.sign(a) * torch.abs(a) ** p
            return a_p.nan_to_num()

        def eotf_inverse_ST2084(C, L_p):
            m_1 = 2610 / 4096 * (1 / 4)
            m_2 = 2523 / 4096 * 128
            c_1 = 3424 / 4096
            c_2 = 2413 / 4096 * 32
            c_3 = 2392 / 4096 * 32
            Y_p = spow(C / L_p, m_1)

            N = spow((c_1 + c_2 * Y_p) / (c_3 * Y_p + 1), m_2)

            return N

        def eotf_BT1886(V, L_B=0, L_W=1):
            # V = to_domain_1(V)

            gamma = 2.40
            gamma_d = 1 / gamma

            n = L_W ** gamma_d - L_B ** gamma_d
            a = n ** gamma
            b = L_B ** gamma_d / n
            if compat:
                L = a * (V + b) ** gamma
            else:
                L = a * torch.clamp(V + b, min=0) ** gamma
            return L
            # return as_float(from_range_1(L))

        def oetf_BT709(L):
            E = torch.where(L < 0.018, L * 4.5, 1.099 * spow(L, 0.45) - 0.099)
            # return as_float(from_range_1(E))
            return E

        def ootf_BT2100_PQ(E):
            return 100 * eotf_BT1886(oetf_BT709(59.5208 * E))

        return eotf_inverse_ST2084(ootf_BT2100_PQ(img), 10000)
    else:
        raise NotImplementedError(f"{type(img)=}")


def pq_to_scenelin(
        img: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    PQ non-linear to scene linear signal, inverse opto-electronic transfer function (OETF^-1).
    https://github.com/colour-science/colour/blob/develop/colour/models/rgb/transfer_functions/itur_bt_2100.py
    : oetf_inverse_BT2100_PQ
    """
    return colour.models.rgb.transfer_functions.itur_bt_2100.oetf_inverse_PQ_BT2100(img)


def match_gain(
        anchor_img: Union[np.ndarray, torch.Tensor],
        other_img: Union[np.ndarray, torch.Tensor],
        return_val: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Match average intensity (gain) between two images.

    Supports single images shaped [C,H,W] and batched images shaped [N,C,H,W].

    Args:
        anchor_img: Reference image whose mean will be matched.
        other_img: Image/tensor to be rescaled to match the anchor mean.
        return_val: If True, return the scalar gain value; otherwise return other_img scaled.

    Returns:
        Either the scaled image/tensor or a scalar gain value depending on return_val.
    """
    if anchor_img.ndim == 4:
        anchor_avg = anchor_img.mean((-1, -2, -3)).view(-1, 1, 1, 1)
        other_avg = other_img.mean((-1, -2, -3)).view(-1, 1, 1, 1)
    elif anchor_img.ndim == 3:  # used to prep dataset w/ RAF (EXR) source
        anchor_avg = anchor_img.mean()
        other_avg = other_img.mean()
    else:
        raise ValueError(f"{anchor_img.ndim=}")
    if return_val:
        return anchor_avg / other_avg
    return other_img * (anchor_avg / other_avg)


def shift_images(
        anchor_img: Union[np.ndarray, torch.Tensor],  # gt
        target_img: Union[np.ndarray, torch.Tensor],  # y
        shift: tuple,  # [int, int],  # python bw compat 2022-11-10
        # crop_to_bayer: bool = True,
        # maintain_shape: bool = False,  # probably not needed w/ crop_to_bayer
) -> Union[tuple, tuple]:
    #  ) -> Union[tuple[np.ndarray, np.ndarray], tuple[torch.Tensor, torch.Tensor]]:  # python bw compat 2022-11-10
    """Shift two aligned images by an integer number of pixels and crop consistently.

    This helper is primarily used to align a clean reference (anchor_img) and a
    target image (target_img) after a coarse shift search. It supports either:
    - both inputs as RGB-like tensors/arrays shaped [..., 3, H, W], or
    - target as a Bayer mosaic shaped [..., 4, H, W] while anchor is RGB.

    When target is Bayer, its effective spatial sampling is half-resolution per
    color plane. Therefore, for odd shifts the function removes one last row/column
    from both tensors to keep shapes compatible.

    Args:
        anchor_img: Reference image to which target is aligned. Shape [..., C, H, W] with C!=4.
        target_img: Image to shift and crop. Shape [..., C, H, W]; may be Bayer with C=4.
        shift: Tuple of (dy, dx), positive meaning downward and rightward shifts for anchor.
            The function applies inverse cropping to target to retain overlapping region.

    Returns:
        A tuple (anchor_img_out, target_img_out), both cropped to a common field of view.
    """
    anchor_img_out = anchor_img
    target_img_out = target_img
    target_is_bayer = target_img.shape[0] == 4
    if anchor_img.shape[0] == 4:
        raise NotImplementedError("shift_images: Bayer anchor_img is not implemented.")
    target_shift_divisor = target_is_bayer + 1
    if shift[0] > 0:  # y
        anchor_img_out = anchor_img_out[..., shift[0]:, :]
        target_img_out = target_img_out[
            ..., : -(shift[0] // target_shift_divisor) or None, :
        ]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., :-1, :]
            target_img_out = target_img_out[..., :-1, :]

    elif shift[0] < 0:
        anchor_img_out = anchor_img_out[..., : shift[0], :]
        target_img_out = target_img_out[..., -shift[0] // target_shift_divisor:, :]
        if shift[0] % 2:
            anchor_img_out = anchor_img_out[..., 1:, :]
            target_img_out = target_img_out[..., 1:, :]
    if shift[1] > 0:  # x
        anchor_img_out = anchor_img_out[..., shift[1]:]
        target_img_out = target_img_out[
            ..., : -(shift[1] // target_shift_divisor) or None
        ]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., :-1]
            target_img_out = target_img_out[..., :-1]
    elif shift[1] < 0:
        anchor_img_out = anchor_img_out[..., : shift[1]]
        target_img_out = target_img_out[..., -shift[1] // target_shift_divisor:]
        if shift[1] % 2:
            anchor_img_out = anchor_img_out[..., 1:]
            target_img_out = target_img_out[..., 1:]
    # try:
    assert shape_is_compatible(anchor_img_out.shape, target_img_out.shape), (
        f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    )
    # except AssertionError as e:
    #    print(e)
    #    breakpoint()

    # assert (
    #     anchor_img_out.shape[1:]
    #     == np.multiply(target_img_out.shape[1:], target_shift_divisor)
    # ).all(), f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    # if maintain_shape:  # unused -> deprecated
    #     assert isinstance(anchor_img_out, torch.Tensor)
    #     xpad = anchor_img.size(-1) - anchor_img_out.size(-1)
    #     ypad = anchor_img.size(-2) - anchor_img_out.size(-2)
    #     anchor_img_out = torch.nn.functional.pad(anchor_img_out, (xpad, 0, ypad, 0))
    #     target_img_out = torch.nn.functional.pad(target_img_out, (xpad, 0, ypad, 0))
    return anchor_img_out, target_img_out


#  def shape_is_compatible(shape1: tuple[int, int, int], shape2: tuple[int, int, int]):  # python bw compat 2022-11-10
def shape_is_compatible(shape1: tuple, shape2: tuple):
    """Returns True if shape1 == shape2 (after debayering if necessary)."""
    return np.all(
        np.multiply(shape1[-2:], (shape1[-3] == 4) + 1)
        == np.multiply(shape2[-2:], (shape2[-3] == 4) + 1)
    )


def shift_mask(
        mask: Union[np.ndarray, torch.Tensor],
        # shift: tuple[int, int],# python bw compat 2022-11-10
        shift: tuple,
        crop_to_bayer: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Shift single (anchor) image in x/y directions and crop accordingly.

    crop_to_bayer: cf shift_images

    TODO / FIXME: is this necessary? (or is mask already shifted when it's computed/created?)
    """
    mask_out = mask
    if shift[0] > 0:
        mask_out = mask_out[..., shift[0]:, :]
        if crop_to_bayer and shift[0] % 2:
            mask_out = mask_out[..., :-1, :]
    elif shift[0] < 0:
        mask_out = mask_out[..., : shift[0], :]
        if crop_to_bayer and shift[0] % 2:
            mask_out = mask_out[..., 1:, :]
    if shift[1] > 0:
        mask_out = mask_out[..., shift[1]:]
        if crop_to_bayer and shift[1] % 2:
            mask_out = mask_out[..., :-1]
    elif shift[1] < 0:
        mask_out = mask_out[..., : shift[1]]
        if crop_to_bayer and shift[1] % 2:
            mask_out = mask_out[..., 1:]

    return mask_out

    # mask_out = mask
    # if shift[0] > 0:  # y
    #     mask_out = mask_out[..., shift[0] :, :]
    #     if target_is_bayer and shift[0] % 2:
    #         mask_out = mask_out[..., :-1, :]
    # elif shift[0] < 0:
    #     mask_out = mask_out[..., : shift[0], :]
    #     if target_is_bayer and shift[0] % 2:
    #         mask_out = mask_out[..., 1:, :]
    # if shift[1] > 0:  # x
    #     mask_out = mask_out[..., shift[1] :]

    #     if target_is_bayer and shift[1] % 2:
    #         mask_out = mask_out[..., :-1]
    # elif shift[1] < 0:
    #     mask_out = mask_out[..., : shift[1]]
    #     if target_is_bayer and shift[1] % 2:
    #         mask_out = mask_out[..., 1:]

    # assert (
    #     anchor_img_out.shape[1:]
    #     == np.multiply(target_img_out.shape[1:], target_shift_divisor)
    # ).all(), f"{anchor_img_out.shape=}, {target_img_out.shape=}"
    # if maintain_shape:  # unused -> deprecated
    #     assert isinstance(anchor_img_out, torch.Tensor)
    #     xpad = anchor_img.size(-1) - anchor_img_out.size(-1)
    #     ypad = anchor_img.size(-2) - anchor_img_out.size(-2)
    #     anchor_img_out = torch.nn.functional.pad(anchor_img_out, (xpad, 0, ypad, 0))
    #     target_img_out = torch.nn.functional.pad(target_img_out, (xpad, 0, ypad, 0))
    return anchor_img_out, target_img_out


def make_overexposure_mask(
        anchor_img: np.ndarray, gt_overexposure_lb: float = GT_OVEREXPOSURE_LB
) -> np.ndarray:
    """Create a boolean mask of non-overexposed pixels from a multi-channel image.

    A pixel is considered valid (mask==True) if all channels are strictly below
    the provided overexposure lower bound.

    Args:
        anchor_img: Image shaped [C, H, W] in linear space.
        gt_overexposure_lb: Lower bound threshold for overexposure in the same units as anchor_img.

    Returns:
        A 2D boolean NumPy array of shape [H, W] where True indicates a valid pixel.
    """
    return (anchor_img < gt_overexposure_lb).all(axis=0)


# def make_loss_mask(
#     anchor_img: np.ndarray,
#     target_img: np.ndarray,
#     loss_threshold: float = LOSS_THRESHOLD,
#     gt_overexposure_lb: float = GT_OVEREXPOSURE_LB,
#     keepers_quantile: float = KEEPERS_QUANTILE,
# ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
#     """Return a loss mask between the two (aligned) images.
#
#     loss_map is the sum of l1 loss over all 4 channels
#
#     0: ignore: if loss_map >= threshold, or anchor_img >= gt_overexposure_lb
#     1: apply loss
#
#     # TODO different keepers_quantile would make a good illustration that noise is not spatially invariant
#     """
#     loss_map = np_l1(anchor_img, match_gain(anchor_img, target_img), avg=False)
#     loss_map = loss_map.sum(axis=0)
#     loss_mask = np.ones_like(loss_map)
#     loss_mask[(anchor_img >= gt_overexposure_lb).any(axis=0)] = 0.
#     reject_threshold = min(loss_threshold, np.quantile(loss_map, keepers_quantile))
#     if reject_threshold == 0:
#         reject_threshold = 1.
#     print(f'{reject_threshold=}')
#     loss_mask[loss_map >= reject_threshold] = 0.
#     return loss_mask# if not return map else (loss_mask, loss_map)
def make_loss_mask(
        anchor_img: np.ndarray,
        target_img: np.ndarray,
        loss_threshold: float = LOSS_THRESHOLD,
        keepers_quantile: float = KEEPERS_QUANTILE,
        verbose: bool = False,
        # ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:  # # python bw compat 2022-11-10
) -> Union[np.ndarray, tuple]:  # # python bw compat 2022-11-10
    """Compute a binary mask that ignores mismatched regions between two aligned images.

    The method computes a per-pixel L1 map between gamma-encoded images and sums over
    channels. Pixels whose loss exceeds a robust threshold are rejected (mask==0), while
    others are kept (mask==1). A morphological opening is applied to remove isolated
    pixels.

    Args:
        anchor_img: Reference image shaped [C, H, W], aligned with target_img.
        target_img: Image to compare against anchor_img, same shape.
        loss_threshold: Absolute upper bound on acceptable per-pixel loss (after channel sum).
        keepers_quantile: Quantile of the loss distribution used as an adaptive threshold.
            The effective threshold is min(loss_threshold, quantile(loss_map)).
        verbose: If True, prints the final threshold.

    Returns:
        A float mask array of shape [H, W] with values in {0.0, 1.0}.
    """
    # loss_map = np_l1(
    #     scenelin_to_pq(anchor_img),
    #     scenelin_to_pq(match_gain(anchor_img, target_imf)),
    #     avg=False,
    # )
    loss_map = np_l1(
        gamma(anchor_img), gamma(match_gain(anchor_img, target_img)), avg=False
    )
    loss_map = loss_map.sum(axis=0)
    loss_mask = np.ones_like(loss_map)
    reject_threshold = min(loss_threshold, np.quantile(loss_map, keepers_quantile))
    if reject_threshold == 0:
        reject_threshold = 1.0
    if verbose:
        print(f"{reject_threshold=}")
    loss_mask[loss_map >= reject_threshold] = 0.0
    loss_mask = scipy.ndimage.binary_opening(loss_mask.astype(np.uint8)).astype(
        np.float32
    )
    return loss_mask  # if not return map else (loss_mask, loss_map)


def find_best_alignment(
        anchor_img: np.ndarray,
        target_img: np.ndarray,
        max_shift_search: int = MAX_SHIFT_SEARCH,
        return_loss_too: bool = False,
        verbose: bool = False,
        # ) -> Union[tuple[int, int], tuple[tuple[int, int], float]]: # python bw compat 2022-11-10
) -> Union[tuple, tuple]:  # python bw compat 2022-11-10
    """Search for the integer (dy, dx) shift minimizing the mean L1 difference.

    The search starts at (0,0) and iteratively explores a local neighborhood around
    the current best shift until convergence or until the Manhattan distance exceeds
    max_shift_search.

    Args:
        anchor_img: Reference image [C, H, W].
        target_img: Image to align [C, H, W]. Its gain is matched to anchor internally.
        max_shift_search: Early stop when |dy|+|dx| reaches this value.
        return_loss_too: If True, also return the minimal loss value.
        verbose: If True, print intermediate best shifts and losses.

    Returns:
        Either (dy, dx) or ((dy, dx), loss) depending on return_loss_too.
    """
    target_img = match_gain(anchor_img, target_img)
    assert np.isclose(anchor_img.mean(), target_img.mean(), atol=1e-07), (
        f"{anchor_img.mean()=}, {target_img.mean()=}"
    )
    # current_best_shift: tuple[int, int] = (0, 0)  # python bw compat 2022-11-10
    # shifts_losses: dict[tuple[int, int], float] = {# python bw compat 2022-11-10
    current_best_shift: tuple = (0, 0)  # python bw compat 2022-11-10
    shifts_losses: dict = {  # python bw compat 2022-11-10
        current_best_shift: np_l1(anchor_img, target_img, avg=True)
    }
    if verbose:
        print(f"{shifts_losses=}")

    def explore_neighbors(
            initial_shift: tuple[int, int],
            shifts_losses: dict[tuple[int, int], float] = shifts_losses,
            anchor_img: np.ndarray = anchor_img,
            target_img: np.ndarray = target_img,
            search_window=NEIGHBORHOOD_SEARCH_WINDOW,
    ) -> None:
        """Explore initial_shift's neighbors and update shifts_losses."""
        for yshift in range(-search_window, search_window + 1, 1):
            for xshift in range(-search_window, search_window + 1, 1):
                current_shift = (initial_shift[0] + yshift, initial_shift[1] + xshift)
                if current_shift in shifts_losses:
                    continue
                shifts_losses[current_shift] = np_l1(
                    *shift_images(anchor_img, target_img, current_shift)
                )
                if verbose:
                    print(f"{current_shift=}, {shifts_losses[current_shift]}")

    while (
            min(shifts_losses.values()) > 0
            and abs(current_best_shift[0]) + abs(current_best_shift[1]) < max_shift_search
    ):
        explore_neighbors(current_best_shift)
        new_best_shift = min(shifts_losses, key=shifts_losses.get)
        if new_best_shift == current_best_shift:
            if return_loss_too:
                return new_best_shift, float(min(shifts_losses.values()))
            return new_best_shift
        current_best_shift = new_best_shift
    if return_loss_too:
        return current_best_shift, float(min(shifts_losses.values()))
    return current_best_shift


def img_fpath_to_np_mono_flt_and_metadata(fpath: str):
    if fpath.endswith(".exr"):
        return np_imgops.img_fpath_to_np_flt(fpath), {"overexposure_lb": 1.0}
    return raw.raw_fpath_to_mono_img_and_metadata(fpath)


def get_best_alignment_compute_gain_and_make_loss_mask(kwargs: dict) -> dict:
    """End-to-end mask creation for a pair of clean/noisy images.

    Given dataset-relative paths for a GT image and a matching file (noisy or
    alternative processing), this function:
    - loads the images in linear space
    - demosaics if needed
    - finds the best integer-pixel alignment
    - computes gain ratios (raw and RGB)
    - builds an overexposure mask and a content-difference mask
    - writes the final mask to disk and returns a metadata dictionary

    The function is designed to be used from multiprocessing pools and therefore
    receives its parameters through a single kwargs dict.

    Expected kwargs keys:
        image_set, gt_file_endpath, f_endpath, ds_dpath[, masks_dpath]

    Returns:
        A dictionary with keys:
            gt_fpath, f_fpath, image_set, best_alignment, best_alignment_loss,
            mask_fpath, mask_mean, is_bayer, rgb_xyz_matrix, overexposure_lb,
            raw_gain, rgb_gain
    """

    def make_mask_name(image_set: str, gt_file_endpath: str, f_endpath: str) -> str:
        return f"{kwargs['image_set']}-{kwargs['gt_file_endpath']}-{kwargs['f_endpath']}.png".replace(
            os.sep, "_"
        )

    assert set(("image_set", "gt_file_endpath", "f_endpath")).issubset(kwargs.keys())
    gt_fpath = os.path.join(
        kwargs["ds_dpath"], kwargs["image_set"], kwargs["gt_file_endpath"]
    )
    f_fpath = os.path.join(kwargs["ds_dpath"], kwargs["image_set"], kwargs["f_endpath"])
    is_bayer = not (gt_fpath.endswith(".exr") or gt_fpath.endswith(".tif"))
    gt_img, gt_metadata = img_fpath_to_np_mono_flt_and_metadata(gt_fpath)
    f_img, f_metadata = img_fpath_to_np_mono_flt_and_metadata(f_fpath)
    mask_name = make_mask_name(
        kwargs["image_set"], kwargs["gt_file_endpath"], kwargs["f_endpath"]
    )
    print(f"get_best_alignment_and_make_loss_mask: {mask_name=}")
    loss_mask = make_overexposure_mask(gt_img, gt_metadata["overexposure_lb"])
    # demosaic before finding alignment
    if is_bayer:
        raw_gain = float(match_gain(gt_img, f_img, return_val=True))
        gt_rgb = raw.demosaic(gt_img, gt_metadata)
        f_rgb = raw.demosaic(f_img, f_metadata)
        rgb_xyz_matrix = gt_metadata["rgb_xyz_matrix"].tolist()

    else:
        gt_rgb = gt_img
        f_rgb = f_img
        rgb_xyz_matrix = None
        raw_gain = None
    best_alignment, best_alignment_loss = find_best_alignment(
        gt_rgb, f_rgb, return_loss_too=True
    )
    rgb_gain = float(match_gain(gt_rgb, f_rgb, return_val=True))
    # gt_rgb_mean = gt_rgb.mean()
    # gain = match_gain(gt_rgb, f_rgb, return_val=True)

    print(f"{kwargs['gt_file_endpath']=}, {kwargs['f_endpath']=}, {best_alignment=}")
    gt_img_aligned, target_img_aligned = shift_images(gt_rgb, f_rgb, best_alignment)
    # align the overexposure mask generated from potentially bayer gt
    loss_mask = shift_mask(loss_mask, best_alignment)
    # add content anomalies between two images to the loss mask
    # try:
    assert gt_img_aligned.shape == target_img_aligned.shape, (
        f"{gt_img_aligned.shape=} is not equal to {target_img_aligned.shape} ({best_alignment=}, {loss_mask.shape=}, {kwargs=})"
    )

    loss_mask = make_loss_mask(gt_img_aligned, target_img_aligned) * loss_mask
    # except ValueError as e:
    #     print(f'get_best_alignment_and_make_loss_mask error {e=}, {kwargs=}, {loss_mask.shape=}, {gt_img.shape=}, {target_img.shape=}, {best_alignment=}, {gt_img_aligned.shape=}, {target_img_aligned.shape=}, {loss_mask.shape=}')
    #     breakpoint()
    #     raise ValueError
    print(
        f"{kwargs['image_set']=}: {loss_mask.min()=}, {loss_mask.max()=}, {loss_mask.mean()=}"
    )
    # save the mask
    masks_dpath = kwargs.get("masks_dpath", MASKS_DPATH)
    os.makedirs(masks_dpath, exist_ok=True)
    mask_fpath = os.path.join(masks_dpath, mask_name)
    np_imgops.np_to_img(loss_mask, mask_fpath, precision=8)
    return {
        "gt_fpath"           : gt_fpath,
        "f_fpath"            : f_fpath,
        "image_set"          : kwargs["image_set"],
        "best_alignment"     : list(best_alignment),
        "best_alignment_loss": best_alignment_loss,
        "mask_fpath"         : mask_fpath,
        "mask_mean"          : float(loss_mask.mean()),
        "is_bayer"           : is_bayer,
        "rgb_xyz_matrix"     : rgb_xyz_matrix,
        "overexposure_lb"    : gt_metadata["overexposure_lb"],
        "raw_gain"           : raw_gain,
        "rgb_gain"           : rgb_gain,
        # "gt_rgb_mean": gt_rgb_mean,
    }


def camRGB_to_lin_rec2020_images(
        camRGB_images: torch.Tensor, rgb_xyz_matrices: torch.Tensor
) -> torch.Tensor:
    """Convert debayered camera RGB images to linear Rec.2020.

    Args:
        camRGB_images: Tensor of shape [N, 3, H, W] in camera RGB space (linear).
        rgb_xyz_matrices: Tensor of shape [N, 3, 3+] providing per-image RGB->XYZ matrices.
            Only the first 3x3 block is used; extra columns (if any) are ignored.

    Returns:
        Tensor of shape [N, 3, H, W] in linear Rec.2020 color space on the same device.
    """
    # cam_to_xyzd65 = torch.linalg.inv(rgb_xyz_matrices[:, :3, :])
    # bugfix for https://github.com/pytorch/pytorch/issues/86465
    cam_to_xyzd65 = torch.linalg.inv(rgb_xyz_matrices[:, :3, :].cpu()).to(
        camRGB_images.device
    )
    xyz_to_lin_rec2020 = torch.tensor(
        [
            [1.71666343, -0.35567332, -0.25336809],
            [-0.66667384, 1.61645574, 0.0157683],
            [0.01764248, -0.04277698, 0.94224328],
        ],
        device=camRGB_images.device,
    )
    color_matrices = xyz_to_lin_rec2020 @ cam_to_xyzd65

    orig_dims = camRGB_images.shape
    # print(orig_dims)
    lin_rec2020_images = (
            color_matrices @ camRGB_images.reshape(orig_dims[0], 3, -1)
    ).reshape(orig_dims)

    return lin_rec2020_images


def demosaic(rggb_img: torch.Tensor) -> torch.Tensor:
    """Demosaic an RGGB Bayer mosaic to camera RGB.

    Supports both single images [4, H, W] and batches [N, 4, H, W]. The output
    preserves the input device and dtype when converting back to torch.

    Args:
        rggb_img: Tensor with channel order [R, G(R), G(B), B] in the first dimension.

    Returns:
        Tensor of shape [3, H, W] or [N, 3, H, W] in camera RGB.
    """
    mono_img: np.ndarray = raw.rggb_to_mono_img(rggb_img)
    if len(mono_img.shape) == 3:
        return torch.from_numpy(raw.demosaic(mono_img, {"bayer_pattern": "RGGB"}))
    new_shape: list[int] = list(mono_img.shape)
    new_shape[-3] = 3
    demosaiced_image: np.ndarray = np.empty_like(mono_img, shape=new_shape)
    for i, img in enumerate(mono_img):
        demosaiced_image[i] = raw.demosaic(mono_img[i], {"bayer_pattern": "RGGB"})
    return torch.from_numpy(demosaiced_image).to(rggb_img.device)


def dt_proc_img(src_fpath: str, dest_fpath: str, xmp_fpath: str, compression: bool = True) -> None:
    """Process a RAW image with Darktable using a provided XMP sidecar.

    This is a thin wrapper around the external `darktable-cli` command that exports
    a 16-bit TIFF according to the specified XMP processing parameters.

    Args:
        src_fpath: Path to the input RAW file.
        dest_fpath: Path where the output TIFF will be written. Must end with .tif and must not exist.
        xmp_fpath: Path to the XMP sidecar containing the processing recipe.
        compression: Placeholder flag for future control of TIFF compression (currently unused).

    Raises:
        AssertionError: If darktable-cli is not available, dest path already exists, or the
            command fails to produce the output file within the timeout.
    """
    assert shutil.which("darktable-cli")
    assert dest_fpath.endswith(".tif")
    assert not os.path.isfile(dest_fpath), f"{dest_fpath} already exists"
    assert not os.path.isfile(dest_fpath), dest_fpath
    conversion_cmd: tuple = (
        "darktable-cli",
        src_fpath,
        xmp_fpath,
        dest_fpath,
        "--core",
        "--conf",
        "plugins/imageio/format/tiff/bpp=16",
    )
    # print(f"dt_proc_img: {' '.join(conversion_cmd)=}")
    subprocess.call(conversion_cmd, timeout=15 * 60)
    assert os.path.isfile(dest_fpath), f"{dest_fpath} was not written by darktable-cli"


class Test_Rawproc(unittest.TestCase):
    def test_camRGB_to_lin_rec2020_images_mt(self):
        self.longMessage = True
        rgb_xyz_matrices = torch.rand(10, 4, 3)
        images = torch.rand(10, 3, 128, 128)
        batched_conversion = camRGB_to_lin_rec2020_images(images, rgb_xyz_matrices)
        for i in range(images.shape[0]):
            single_conversion = camRGB_to_lin_rec2020_images(
                images[i].unsqueeze(0), rgb_xyz_matrices[i].unsqueeze(0)
            )
            self.assertTrue(
                torch.allclose(
                    single_conversion,
                    batched_conversion[i: i + 1],
                    atol=1e-04,
                    rtol=1e-04,
                )
            )

    def test_match_gains(self):
        self.longMessage = True
        anchor_img = torch.rand(3, 128, 128)
        target_img = torch.rand(3, 128, 128)
        target_img = match_gain(anchor_img, target_img)
        self.assertAlmostEqual(
            anchor_img.mean().item(), target_img.mean().item(), places=5
        )
        anchor_batch = torch.rand(10, 3, 128, 128)
        anchor_batch[1] *= 10
        target_batch = torch.rand(10, 3, 128, 128)
        target_batch[1] /= 10
        target_batch[5] /= 5
        target_batch[7] += 0.5
        target_batch[9] /= 90
        print(f"{anchor_batch.mean()=}, {target_batch.mean()=}")
        target_batch = match_gain(anchor_batch, target_batch)
        print(f"{anchor_batch.mean()=}, {target_batch.mean()=}")
        self.assertGreaterEqual(target_batch[1].mean(), 2.5)
        self.assertGreaterEqual(target_batch[5].mean(), 0.25)
        self.assertGreaterEqual(target_batch[1].mean(), target_batch.mean())
        for i in range(anchor_batch.shape[0]):
            self.assertAlmostEqual(
                anchor_batch[i].mean().item(), target_batch[i].mean().item(), places=5
            )


if __name__ == "__main__":
    unittest.main()
