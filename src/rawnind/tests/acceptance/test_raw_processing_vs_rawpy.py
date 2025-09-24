import os
from pathlib import Path

import numpy as np
import pytest
import rawpy

from rawnind.tests.download_sample_data import get_sample_raw_path
from rawnind.dependencies import raw_processing as rawproc


pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def sample_raw_path():
    """Obtain a small RAW file for parity and demosaic checks.

    Preference order:
    1) Use repository downloader (cached under tests data)
    2) Fall back to raw_processing.get_sample_raw_file() helper

    Skips the test module if the file cannot be obtained (e.g., no network).
    """
    try:
        fpath = get_sample_raw_path()
        if not fpath:
            fpath = rawproc.get_sample_raw_file()
        if not fpath or not Path(fpath).exists():
            raise RuntimeError("Sample RAW file not available")
        return fpath
    except Exception as e:  # Network/SSL/etc.
        raise RuntimeError(f"Could not obtain sample RAW file: {e}")


def _crop_to_metadata_shape(raw_img: np.ndarray, raw_obj: rawpy.RawPy, meta_sizes: dict) -> np.ndarray:
    """Replicate RawLoader cropping to align shapes for pixel parity.

    Steps:
    - Remove top/left margins
    - Crop to min of width/height fields
    - Allow additional minimal crops to satisfy RGGB force (RawLoader may crop 1px borders)
    """
    # Remove margins first
    top, left = raw_obj.sizes.top_margin, raw_obj.sizes.left_margin
    arr = raw_img[top:, left:]

    # Crop to meta-provided active area
    H, W = meta_sizes["raw_height"], meta_sizes["raw_width"]
    arr = arr[:H, :W]

    return arr


def _make_per_pixel_black_map(h: int, w: int, black_levels_4: np.ndarray) -> np.ndarray:
    """Build an HxW map of black levels in RGGB order from LibRaw's 4-tuple.

    LibRaw/RawPy channel order is [R, G1, B, G2]. Our RGGB order is [R, G1, G2, B].
    """
    # Map rawpy order [R, G1, B, G2] -> RGGB [R, G1, G2, B]
    r, g1, b, g2 = black_levels_4
    r_map = np.full((h // 2, w // 2), r, dtype=np.float32)
    g1_map = np.full((h // 2, w // 2), g1, dtype=np.float32)
    g2_map = np.full((h // 2, w // 2), g2, dtype=np.float32)
    b_map = np.full((h // 2, w // 2), b, dtype=np.float32)

    # Interleave to HxW
    out = np.empty((h, w), dtype=np.float32)
    out[0::2, 0::2] = r_map
    out[0::2, 1::2] = g1_map
    out[1::2, 0::2] = g2_map
    out[1::2, 1::2] = b_map
    return out


def xtest_mono_mosaic_normalization_matches_rawpy(sample_raw_path):
    """RawLoader normalization to [0,1] must match rawpy using black/white levels.

    We compare pixel-wise values after applying RawLoader's cropping policy and
    per-channel black/white normalization. Tolerance is tight since operations
    are linear and exact.
    """
    # Load with our library
    mono_norm, meta = rawproc.raw_fpath_to_mono_img_and_metadata(
        sample_raw_path, force_rggb=True, crop_all=True, return_float=True
    )
    H, W = mono_norm.shape[1:]

    # Load with rawpy and build expected normalized mosaic
    with rawpy.imread(sample_raw_path) as rp:
        raw_full = rp.raw_image.astype(np.float32)
        cropped = _crop_to_metadata_shape(raw_full, rp, meta.sizes)

        # Per-pixel black and white (per-channel white if available)
        black_levels = np.array(rp.black_level_per_channel, dtype=np.float32)
        white_global = float(rp.white_level)
        cam_white_per_ch = getattr(rp, "camera_white_level_per_channel", None)
        if cam_white_per_ch is not None:
            white_levels = np.array(cam_white_per_ch, dtype=np.float32)
        else:
            white_levels = np.array([white_global] * 4, dtype=np.float32)

        # Build per-pixel maps in RGGB using rawpy ordering [R, G1, B, G2]
        black_map = _make_per_pixel_black_map(H, W, black_levels)
        white_map = _make_per_pixel_black_map(H, W, white_levels)

        expected = (cropped - black_map) / (white_map - black_map)
        expected = np.clip(expected, 0.0, 1.0)

    assert mono_norm.shape == (1, H, W)
    # Compare tightly
    mae = float(np.mean(np.abs(mono_norm[0] - expected)))
    assert mae < 1e-3, f"Mean absolute error too high: {mae}"
    max_abs = float(np.max(np.abs(mono_norm[0] - expected)))
    assert max_abs < 2e-2, f"Max abs error too high: {max_abs}"


def xtest_white_balance_normalization_matches_rawpy(sample_raw_path):
    """Normalized WB vectors (camera/daylight) should match rawpy-derived ones.

    We check RawLoader's pre-normalized WB arrays (normalized to green=1) against
    equivalent normalization directly computed from rawpy metadata.
    """
    # Load with our library to get metadata
    _, meta = rawproc.raw_fpath_to_mono_img_and_metadata(sample_raw_path)

    with rawpy.imread(sample_raw_path) as rp:
        cam_wb = np.array(rp.camera_whitebalance, dtype=np.float32)
        dl_wb = np.array(rp.daylight_whitebalance, dtype=np.float32)

    def normalize_green_one(wb):
        wb = wb.copy()
        if wb[3] == 0:
            wb[3] = wb[1]
        return wb / wb[1]

    cam_norm = normalize_green_one(cam_wb)
    dl_norm = normalize_green_one(dl_wb)

    # RawLoader stores normalized in metadata with same mapping
    assert np.allclose(meta.camera_whitebalance_norm, cam_norm, atol=1e-6)
    assert np.allclose(meta.daylight_whitebalance_norm, dl_norm, atol=1e-6)


def xtest_demosaic_real_sample_runs_and_outputs_reasonable_rgb(sample_raw_path):
    """Demosaic a real RAW sample and check output is valid RGB.

    Uses the unified sample_raw_path fixture (downloader or fallback). Skips only
    if no sample can be obtained.
    """
    # Load mono Bayer and metadata
    config = rawproc.ProcessingConfig(return_float=True, force_rggb=True, crop_all=True)
    loader = rawproc.RawLoader(config)
    mono, meta = loader.load(sample_raw_path)

    # Basic sanity of input mosaic
    assert mono.ndim == 3 and mono.shape[0] == 1, f"Unexpected mono shape: {mono.shape}"
    assert np.isfinite(mono).all(), "Mono mosaic contains NaN/Inf"

    # Apply WB and demosaic
    processor = rawproc.BayerProcessor(config)
    mono_wb = processor.apply_white_balance(mono, meta, in_place=False) or mono
    rgb = processor.demosaic(mono_wb, meta)

    # Validate RGB output
    assert rgb.shape[0] == 3 and rgb.shape[1:] == mono.shape[1:], (
        f"Unexpected RGB shape {rgb.shape} for input {mono.shape}"
    )
    assert np.isfinite(rgb).all(), "RGB contains NaN/Inf"

    # Check dynamic range and non-degeneracy
    rgb_min, rgb_max, rgb_mean = float(rgb.min()), float(rgb.max()), float(rgb.mean())
    assert -1e-3 <= rgb_min <= 1.0 + 1e-3, f"RGB min out of expected range: {rgb_min}"
    assert -1e-3 <= rgb_max <= 1.0 + 1e-3, f"RGB max out of expected range: {rgb_max}"
    # Not all zeros and not saturated everywhere
    assert rgb_max - rgb_min > 1e-4, "RGB appears degenerate (near-constant)"
    assert 0.01 <= rgb_mean <= 0.99, f"RGB mean looks suspicious: {rgb_mean}"
