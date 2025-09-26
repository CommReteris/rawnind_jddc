"""PyTorch helper functions and utilities.

This module contains PyTorch-specific utilities extracted from
the original pt_helpers.py file.

Extracted from libs/pt_helpers.py as part of the codebase refactoring.
"""

import logging
import time

import cv2
import numpy as np
import torch
from PIL import Image

# Import from dependencies package (will be moved later)
<<<<<<< HEAD
from .utilities import noop
=======
def noop(*args, **kwargs):
    """No-operation function that accepts any arguments and does nothing."""
    pass
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c


def get_device(device_n=None):
    """Get device given index (-1 = CPU).

    Args:
        device_n: Device number or device object

    Returns:
        torch.device: The appropriate device
    """
    if isinstance(device_n, torch.device):
        return device_n
    elif isinstance(device_n, str):
        if device_n == "cpu":
            return torch.device("cpu")
        device_n = int(device_n)

    if device_n is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("get_device: cuda not available; defaulting to cpu")
            return torch.device("cpu")
    elif torch.cuda.is_available() and device_n >= 0:
        return torch.device(f"cuda:{device_n}")
    elif device_n >= 0:
        print("get_device: cuda not available")
    return torch.device("cpu")


def fpath_to_tensor(
        img_fpath,
        device=torch.device("cpu"),
        batch=False,
        incl_metadata=False,
        crop_to_multiple: int = False,
):
    """Convert image file path to PyTorch tensor.

    Args:
        img_fpath: Path to image file
        device: Target device for tensor
        batch: Whether to add batch dimension
        incl_metadata: Whether to include metadata
        crop_to_multiple: Crop to multiple of this number

    Returns:
        torch.Tensor or tuple: Image tensor (and metadata if requested)
    """
    # Import from dependencies (will be moved later)
    from .numpy_operations import img_fpath_to_np_flt

    try:
        tensor = img_fpath_to_np_flt(img_fpath, incl_metadata=incl_metadata)
    except ValueError as e:
        try:
            logging.error(f"fpath_to_tensor error {e} with {img_fpath=}. Trying again.")
            tensor = img_fpath_to_np_flt(img_fpath, incl_metadata=incl_metadata)
        except ValueError as e:
            logging.error(f"fpath_to_tensor failed again ({e}). Trying one last time after 5 seconds.")
            time.sleep(5)
            tensor = img_fpath_to_np_flt(img_fpath, incl_metadata=incl_metadata)

    if incl_metadata:
        tensor, metadata = tensor

    tensor = torch.tensor(tensor, device=device)

    if crop_to_multiple:
<<<<<<< HEAD
        tensor = crop_to_multiple(tensor, crop_to_multiple)
=======
        from .pytorch_operations import crop_to_multiple as crop_fn
        tensor = crop_fn(tensor, crop_to_multiple)
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

    if batch:
        tensor = tensor.unsqueeze(0)

    if incl_metadata:
        return tensor, metadata
    else:
        return tensor


def sdr_pttensor_to_file(tensor: torch.Tensor, fpath: str):
    """Save PyTorch tensor to SDR image file.

    Args:
        tensor: PyTorch tensor to save
        fpath: Output file path
    """
    if tensor.dim() == 4:
        assert tensor.size(0) == 1, "sdr_pttensor_to_file: batch size > 1 is not supported"
        tensor = tensor.squeeze(0)

    if tensor.dtype == torch.float32 or tensor.dtype == torch.float16:
        if fpath[-4:].lower() in [".jpg", "jpeg"]:
            import torchvision
            return torchvision.utils.save_image(tensor.clip(0, 1), fpath)
        elif fpath[-4:].lower() in [".png", ".tif", "tiff"]:
            nptensor = (tensor.clip(0, 1) * 65535).round().cpu().numpy().astype(np.uint16).transpose(1, 2, 0)
            nptensor = cv2.cvtColor(nptensor, cv2.COLOR_RGB2BGR)
            outflags = None
            if fpath.endswith("tif") or fpath.endswith("tiff"):
                outflags = (cv2.IMWRITE_TIFF_COMPRESSION, 34925)  # lzma2
            cv2.imwrite(fpath, nptensor, outflags)
        else:
            raise NotImplementedError(f"Extension in {fpath}")
    elif tensor.dtype == torch.uint8:
        tensor = tensor.permute(1, 2, 0).to(torch.uint8).numpy()
        pilimg = Image.fromarray(tensor)
        pilimg.save(fpath)
    else:
        raise NotImplementedError(tensor.dtype)


def freeze_model(net):
    """Freeze model parameters for inference.

    Args:
        net: PyTorch model to freeze

    Returns:
        Frozen model
    """
    net = net.eval()
    for p in net.parameters():
        p.requires_grad = False
    return net


<<<<<<< HEAD
def get_losses(img1_fpath, img2_fpath):
    """Compute various losses between two images.

    Args:
        img1_fpath: Path to first image
        img2_fpath: Path to second image

    Returns:
        dict: Dictionary of loss metrics
    """
    # Import from dependencies (will be moved later)
    from .pytorch_losses import SSIM_loss, MS_SSIM_loss

    img1 = fpath_to_tensor(img1_fpath).unsqueeze(0)
    img2 = fpath_to_tensor(img2_fpath).unsqueeze(0)
    assert img1.shape == img2.shape, f"img1.shape={img1.shape}, img2.shape={img2.shape}"

    res = dict()
    res["mse"] = torch.nn.functional.mse_loss(img1, img2).item()
    res["ssim"] = SSIM_loss()(img1, img2).item()
    res["msssim"] = MS_SSIM_loss()(img1, img2).item()
    return res


=======
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
# CUDA synchronization utility
torch_cuda_synchronize = (
    torch.cuda.synchronize if torch.cuda.is_available() else noop
)
<<<<<<< HEAD
=======

from torch.optim.lr_scheduler import LambdaLR

def get_basic_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
