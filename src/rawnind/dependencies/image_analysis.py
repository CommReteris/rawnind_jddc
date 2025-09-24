# -*- coding: utf-8 -*-
"""
Image analysis on file paths
"""

import subprocess
from typing import Union
from PIL import Image
import os
import shutil
import numpy as np

if shutil.which("exiftool") is None:
    print("libimganalysis warning: exiftool binary is missing. get_iso is impacted.")
try:
    import piexif
except ModuleNotFoundError:
    print(
        "filter_dataset_by_iso.py: warning: piexif library not found, using exiftool instead"
    )
from typing import Optional
import sys

import piqa

from . import pytorch_helpers as pt_helpers
from . import raw_processing as rawproc

VALID_IMG_EXT = ["png", "jpg", "jpeg", "bmp", "gif", "tiff", "ppm", "j2k", "webp"]


def get_iso(fpath):
    def piexif_get_iso(fpath):
        """
        supports jpeg and maybe tiff. should be slightly faster than calling exiftool.
        """
        try:
            exifdata = piexif.load(fpath)["Exif"]
        except Exception as e:
            print(f"piexif_get_iso: {e} on {fpath}; reverting to exiftool_get_iso")
            return
        if 34855 in exifdata:
            isoval = exifdata[34855]
            if not isinstance(isoval, int):
                print(
                    f"piexif_get_iso: invalid non-int format for {fpath} ({isoval}), skipping."
                )
                isoval = None
            return isoval

    def exiftool_get_iso(fpath):
        cmd = "exiftool", "-S", "-ISO", fpath
        try:
            res = subprocess.run(cmd, text=True, capture_output=True, timeout=30).stdout
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"exiftool_get_iso error: exiftool binary not present ({e}"
            )
        if res == "":
            return None
        else:
            try:
                return int(res.split(": ")[-1])
            except ValueError as e:
                print(f"exiftool_get_iso: got {e} on {fpath}, skipping.")

    exiftool_installed = True if shutil.which("exiftool") else False
    ext = fpath[-4:].lower()
    isoval = False
    if (
            ext.endswith("jpg")
            or ext.endswith("jpeg")
            or ext.endswith("tif")
            or ext.endswith("tiff")
    ) and "piexif" in sys.modules:
        isoval = piexif_get_iso(fpath)
    if exiftool_installed and not isoval:
        isoval = exiftool_get_iso(fpath)
    if isoval is False:
        isoval = None
        print(
            "get_iso: no suitable tools found. Install piexif library and/or exiftool executable."
        )
    assert isoval is None or isinstance(isoval, int), f"fpath={fpath}, isoval={isoval}"
    return isoval


def is_raw(fpath: str) -> bool:
    if fpath.split(".")[-1].lower() in (
            "3fr",
            "ari",
            "arw",
            "bay",
            "braw",
            "crw",
            "cr2",
            "cr3",
            "cap",
            "data",
            "dcs",
            "dcr",
            "dng",
            "drf",
            "eip",
            "erf",
            "fff",
            "gpr",
            "iiq",
            "k25",
            "kdc",
            "mdc",
            "mef",
            "mos",
            "mrw",
            "nef",
            "nrw",
            "obm",
            "orf",
            "pef",
            "ptx",
            "pxn",
            "r3d",
            "raf",
            "raw",
            "rwl",
            "rw2",
            "rwz",
            "sr2",
            "srf",
            "srw",
            # "tif",
            "x3f",
            "cri",
            "jxs",
            "tco",
    ):
        return True
    return False


def piqa_msssim(img1path: str, img2path: str):
    img1 = pt_helpers.fpath_to_tensor(img1path, batch=True)
    img2 = pt_helpers.fpath_to_tensor(img2path, batch=True)
    return piqa.MS_SSIM()(img1, img2).item()

msssim = piqa_msssim

def pil_get_resolution(imgpath):
    return Image.open(imgpath).size


def is_valid_img(img_fpath, open_img=False, save_img=False, clean=False):
    """
    Check if an image is valid.
    open_img = True: use PIL's verify function
    save_img = True: use PIL to resize the image and save as png (slower but more effective)
    open_img = save_img = False: just check the extension (default)
    clean = True: remove deffective images, otherwise just return False
    """
    if is_raw(img_fpath) or img_fpath.split(".")[-1].lower() == "exr":
        try:
            img, metadata = rawproc.raw_fpath_to_rggb_img_and_metadata(img_fpath)
            return True
        except AssertionError:
            if img_fpath.lower().endswith(".raf"):  # cannot check RAF atm
                return True
            return False
        except ValueError:
            return False
    if img_fpath.lower().endswith(".npy"):
        try:
            np.load(img_fpath)
            return True
        except ValueError:
            return False
    Image.MAX_IMAGE_PIXELS = 15000 ** 2
    ext_is_valid = img_fpath.split(".")[-1].lower() in VALID_IMG_EXT
    if not open_img and not save_img:
        return ext_is_valid
    try:
        img = Image.open(img_fpath)
        if save_img:
            img = img.resize((128, 128))
            img.save("tmp.png")
        else:
            img.verify()
        return True
    except OSError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print("rm {}".format(img_fpath))
        return False
    except Image.DecompressionBombError as e:
        print(e)
        if clean:
            os.remove(img_fpath)
            print("rm {}".format(img_fpath))
        return False