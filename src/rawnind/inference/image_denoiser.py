"""Single image denoising functionality.

This module provides tools for denoising individual images using trained models,
including comprehensive metrics computation and output processing.

Extracted from tools/denoise_image.py as part of the codebase refactoring.
"""

import os
import sys
from typing import Optional

import torch
import yaml

# Import from dependencies package (will be moved later)
from ..dependencies.pt_losses import metrics as pt_losses_metrics
from ..dependencies.pytorch_helpers import fpath_to_tensor
from ..dependencies.utilities import load_yaml

# Import raw processing from dependencies
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw

# Import inference components
from .model_factory import get_and_load_test_object
from .base_inference import ImageToImageNN

DENOISED_DN = "denoised_images"
METRICS_DN = "denoised_images_metrics"


# CLI interface removed - use clean API functions instead:
# from rawnind.inference import create_rgb_denoiser, load_model_from_checkpoint, compute_image_metrics



def load_image(fpath, device) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load an image and optional RGB->XYZ matrix for processing.

    Args:
        fpath: Path to input image (.exr/.tif or RAW supported by rawnind.libs.raw).
        device: Target device for returned tensors.

    Returns:
        Tuple (img, rgb_xyz_matrix) where rgb_xyz_matrix may be None.
    """
    img, metadata = fpath_to_tensor(
        fpath, incl_metadata=True, device=device, crop_to_multiple=16
    )
    rgb_xyz_matrix = metadata.get("rgb_xyz_matrix", None)
    if rgb_xyz_matrix is not None:
        rgb_xyz_matrix = torch.tensor(rgb_xyz_matrix).unsqueeze(0)
    return img, rgb_xyz_matrix


def process_image_base(
        test_obj: ImageToImageNN,
        out_img: torch.Tensor,
        gt_img: Optional[torch.Tensor] = None,
        in_img: Optional[torch.Tensor] = None,
        rgb_xyz_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Map a model's raw output to a comparable linear Rec.2020 image.

    Applies test_obj.process_net_output when available to convert the network
    output into a linear Rec.2020 image, optionally matching exposure to a
    reference image (ground-truth or input). Falls back to simple gain matching
    when the test object does not define process_net_output.

    Args:
        test_obj: Inference object providing process_net_output and settings.
        out_img: Raw network output tensor (B,C,H,W or C,H,W).
        gt_img: Optional ground-truth tensor for exposure reference.
        in_img: Optional input tensor for exposure reference if GT is missing.
        rgb_xyz_matrix: Optional per-image color transform matrix required by
            some pipelines.

    Returns:
        Linear Rec.2020 image suitable for metric computation and saving.
    """
    if gt_img is not None:
        ref_img = gt_img
    elif in_img is not None and in_img.shape[-3] == out_img.shape[-3]:
        ref_img = in_img
    elif in_img is not None and in_img.shape[-3] == 4 and out_img.shape[-3] == 3:
        # demosaic image to get input mean for match_gain
        ref_img = rawproc.demosaic(in_img).unsqueeze(0)
        ref_img = rawproc.camRGB_to_lin_rec2020_images(ref_img, rgb_xyz_matrix)
    else:
        ref_img = None
    if hasattr(test_obj, "process_net_output"):
        print(f"Mean of network output: {out_img.mean()}")
        out_img = test_obj.process_net_output(out_img, rgb_xyz_matrix, ref_img)
        print(f"Mean after process_net_output: {out_img.mean()}")
    elif ref_img is not None:
        print(f"Mean of network output: {out_img.mean()}")
        out_img = rawproc.match_gain(ref_img, out_img)
        print(f"Mean after match_gain: {out_img.mean()}")
    else:
        pass
    if out_img.mean() > 1 or out_img.mean() < 0:
        print(
            f"WARNING: mean of output image is outside of valid range ({out_img.mean()=})"
        )
        out_img = rawproc.match_gain(ref_img, out_img)
        print(f"Mean after matching gain: {out_img.mean()}")
    return out_img


def apply_nonlinearity(img: torch.Tensor, nonlinearity: str) -> torch.Tensor:
    """Apply a named transfer function to an image tensor.

    Args:
        img: Linear image tensor in [0,1].
        nonlinearity: One of the keys supported by ImageToImageNN.get_transfer_function
            (e.g., 'pq', 'gamma22').

    Returns:
        Transformed tensor (clone of input).
    """
    # Import here to avoid circular imports
    from ..libs.abstract_trainer import ImageToImageNN
    return ImageToImageNN.get_transfer_function(nonlinearity)(
        img.clone()
    )


def compute_metrics(
        in_img: torch.Tensor,
        gt_img: torch.Tensor,
        metrics: list[str] = [],
        prefix=None,
) -> dict:
    """Compute a set of metrics between two images.

    Args:
        in_img: Predicted image tensor.
        gt_img: Ground-truth image tensor aligned to in_img.
        metrics: List of metric names registered in common.pt_losses.metrics.
        prefix: Optional prefix added to metric keys.

    Returns:
        Dictionary mapping metric names to float values.
    """
    metrics_results = {}
    for metric in metrics:
        # Instantiate the metric class first, then call it
        metric_fn = pt_losses_metrics[metric]()
        metrics_results[metric] = float(metric_fn(in_img, gt_img))
    if prefix is not None:
        metrics_results = {f"{prefix}_{k}": v for k, v in metrics_results.items()}
    return metrics_results


def save_image(image, fpath: str, src_fpath: Optional[str] = None):
    """Save a linear Rec.2020 image to disk, preserving metadata when possible.

    Args:
        image: Tensor with shape (C,H,W) or (1,C,H,W); C must be 3.
        fpath: Destination filepath (.exr or .tif supported by rawnind.libs.raw).
        src_fpath: Optional original source path to carry over metadata.
    """
    assert image.shape[-3] == 3
    if len(image.shape) == 4:
        image = image.squeeze(0)
    raw.hdr_nparray_to_file(image.numpy(), fpath, "lin_rec2020")


def save_metrics(metrics: dict, fpath: str):
    """Write a metrics dictionary to a YAML file.

    Args:
        metrics: Mapping from metric names to values.
        fpath: Destination YAML filepath.
    """
    with open(fpath, "w") as f:
        yaml.dump(metrics, f)


def denoise_image_from_to_fpath(
        in_img_fpath: str, out_img_fpath: str, test_obj: ImageToImageNN
):
    """Denoise a single image file and write the processed output.

    Args:
        in_img_fpath: Input image path (RAW or image file supported by loaders).
        out_img_fpath: Destination path for the denoised image.
        test_obj: Inference object used to run the model and processing.
    """
    img, rgb_xyz_matrix = load_image(in_img_fpath, device=test_obj.device)

    model_results = test_obj.infer(img, return_dict=False)
    processed_image = process_image_base(
        test_obj, model_results, rgb_xyz_matrix=rgb_xyz_matrix
    )
    save_image(processed_image, out_img_fpath, src_fpath=in_img_fpath)


def bayer_to_prgb(image, rgb_xyz_matrix):
    """Convert Bayer pattern image to profiled RGB for pRGB model compatibility.

    This utility function handles automatic color space conversion when a Bayer pattern
    image needs to be processed by a model that expects profiled RGB input. If the
    input is already in RGB format, it passes through unchanged.

    Args:
        image: Input image tensor, either Bayer (4-channel) or RGB (3-channel).
        rgb_xyz_matrix: Color transformation matrix for camera RGB to linear Rec.2020.

    Returns:
        Image tensor in profiled RGB color space, suitable for pRGB model input.
    """
    if image.shape[-3] == 3:
        return image
    image = rawproc.demosaic(image).unsqueeze(0)
    image = rawproc.camRGB_to_lin_rec2020_images(image, rgb_xyz_matrix)
    return image


def denoise_image_compute_metrics(
        in_img,
        test_obj: ImageToImageNN,
        rgb_xyz_matrix: Optional[torch.Tensor] = None,
        gt_img: Optional[torch.Tensor] = None,
        metrics: list[str] = [],
        nonlinearities: list[str] = [],
) -> tuple[torch.Tensor, dict]:
    """Denoise an image tensor and compute evaluation metrics against ground truth.

    This function handles the complete denoising pipeline including automatic color space
    conversion, model inference, output processing, and comprehensive metric computation
    with optional perceptual transforms.

    Args:
        in_img: Input image tensor to denoise.
        test_obj: Trained model inference object.
        rgb_xyz_matrix: Optional color transformation matrix for Bayer to pRGB conversion.
        gt_img: Optional ground truth image for metric computation.
        metrics: List of metric names to compute (e.g., 'mse', 'msssim_loss').
        nonlinearities: List of perceptual transforms to apply before metrics.

    Returns:
        Tuple of (processed_denoised_image, metrics_dict) where metrics_dict
        includes both standard metrics and optional compression bitrate.
    """
    # if model is pRGB and img is bayer, debayer
    if test_obj.in_channels == 3:
        in_img = bayer_to_prgb(in_img, rgb_xyz_matrix)
    # denoise and proc
    model_results = test_obj.infer(in_img, return_dict=True)
    processed_image = process_image_base(
        test_obj,
        model_results["reconstructed_image"],
        gt_img,
        in_img,
        rgb_xyz_matrix=rgb_xyz_matrix,
    )
    # init metrics
    metrics_results = {}
    if "bpp" in model_results:
        metrics_results["bpp"] = float(model_results["bpp"])
    # compute metrics
    if gt_img and metrics:
        metrics_results.update(compute_metrics(processed_image, gt_img, metrics))
        for nonlinearity in nonlinearities:
            if str(nonlinearity) == "None":
                continue
            nl_gt = apply_nonlinearity(gt_img, nonlinearity)
            nl_out = apply_nonlinearity(processed_image, nonlinearity)
            nl_out = rawproc.match_gain(nl_gt, nl_out)
            metrics_results.update(
                compute_metrics(nl_out, nl_gt, metrics, nonlinearity)
            )
    return processed_image, metrics_results


def denoise_image_from_fpath_compute_metrics_and_export(
        in_img_fpath: str,
        test_obj: Optional[ImageToImageNN] = None,
        gt_img_fpath: Optional[str] = None,
        metrics: list[str] = [],
        nonlinearities: list[str] = [],
        out_img_fpath=None,
):
    """Convenience wrapper: load, denoise, compute metrics, and export.

    Args:
        in_img_fpath: Input image file path.
        test_obj: If None, will be created via abstract_trainer.get_and_load_test_object().
        gt_img_fpath: Optional ground-truth image path.
        metrics: Metrics to compute.
        nonlinearities: Perceptual non-linearities to apply before metrics.
        out_img_fpath: Optional explicit output path for the denoised image.
    """
    if test_obj is None:
        test_obj = get_and_load_test_object()
    in_img, rgb_xyz_matrix = load_image(in_img_fpath, test_obj.device)
    if gt_img_fpath:
        gt_img = load_image(gt_img_fpath, test_obj.device)
    else:
        gt_img = None
    processed_image, metrics_results = denoise_image_compute_metrics(
        in_img=in_img,
        test_obj=test_obj,
        gt_img=gt_img,
        metrics=metrics,
        nonlinearities=nonlinearities,
        rgb_xyz_matrix=rgb_xyz_matrix,
    )
    # output
    model_fn = os.path.basename(test_obj.load_path)
    if out_img_fpath is None:
        out_img_dpath = os.path.join(test_obj.save_dpath, DENOISED_DN + "_" + model_fn)
        os.makedirs(out_img_dpath, exist_ok=True)
        out_img_fpath = os.path.join(
            out_img_dpath, os.path.basename(in_img_fpath) + ".denoised.tif"
        )
    save_image(processed_image, out_img_fpath, src_fpath=in_img_fpath)
    if gt_img_fpath or metrics_results:
        metrics_dpath = os.path.join(test_obj.save_dpath, METRICS_DN + "_" + model_fn)
        os.makedirs(metrics_dpath, exist_ok=True)
        if gt_img_fpath:
            metrics_fn = f"{os.path.basename(in_img_fpath)}--{os.path.basename(gt_img_fpath)}.metrics.yaml"
        else:
            metrics_fn = f"{os.path.basename(in_img_fpath)}.metrics.yaml"
        save_metrics(metrics_results, os.path.join(metrics_dpath, metrics_fn))
        print(
            f"{metrics_results=} written to {os.path.join(metrics_dpath, metrics_fn)}"
        )


# CLI interface removed - use clean API functions instead:
# from rawnind.inference import create_rgb_denoiser, load_model_from_checkpoint, compute_image_metrics