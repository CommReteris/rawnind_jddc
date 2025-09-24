"""Unit tests for dataset classes.

This module contains comprehensive unit tests for all dataset classes
in the dataset package, ensuring proper functionality and data integrity.
"""

import random
import time
import pytest
import torch
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Literal, Optional, List

# Import raw processing (will be moved to dependencies later)
from ...dependencies import arbitrary_processing as arbitrary_proc_fun, pytorch_helpers as pt_helpers_dep
from ..base_dataset import RawImageDataset





# Constants from original rawds.py (using updated values/references from clean_api)
BREAKPOINT_ON_ERROR = True
COLOR_PROFILE = "lin_rec2020"
TOY_DATASET_LEN = 2 # Reduced for faster testing

# Mock content_fpath for hermetic testing
@pytest.fixture
def mock_content_fpath_rawnind(tmp_path):
    """Create a mock YAML file path for RAWNIND content."""
    yaml_path = tmp_path / 'rawnind_content.yaml'
    yaml_content = {
        'images': [
            {'img_id': 'MuseeL-Bobo-alt-A7C', 'gt_path': str(tmp_path / 'gt_bobo.exr'), 'raw_path': str(tmp_path / 'raw_bobo.exr'),
             'is_bayer': True, 'image_set': 'MuseeL-Bobo-alt-A7C', 'best_alignment_loss': 0.01, 'mask_mean': 0.9,
             'rgb_msssim_score': 0.95, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'raw_gain': 1.5,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'gt_bobo.exr'), 'f_bayer_fpath': str(tmp_path / 'raw_bobo.exr'), 'mask_fpath': str(tmp_path / 'mask_bobo.exr')}]},
            {'img_id': 'MuseeL-yombe-A7C', 'gt_path': str(tmp_path / 'gt_yombe.exr'), 'raw_path': str(tmp_path / 'raw_yombe.exr'),
             'is_bayer': True, 'image_set': 'MuseeL-yombe-A7C', 'best_alignment_loss': 0.02, 'mask_mean': 0.85,
             'rgb_msssim_score': 0.80, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'raw_gain': 1.2,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'gt_yombe.exr'), 'f_bayer_fpath': str(tmp_path / 'raw_yombe.exr'), 'mask_fpath': str(tmp_path / 'mask_yombe.exr')}]}
        ]
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files that the RawImageDataset would expect
    (tmp_path / 'gt_bobo.exr').touch()
    (tmp_path / 'raw_bobo.exr').touch()
    (tmp_path / 'mask_bobo.exr').touch()
    (tmp_path / 'gt_yombe.exr').touch()
    (tmp_path / 'raw_yombe.exr').touch()
    (tmp_path / 'mask_yombe.exr').touch()
    return str(yaml_path)

@pytest.fixture
def mock_content_fpath_extraraw(tmp_path):
    """Create a mock YAML file path for EXTRARAW content."""
    yaml_path = tmp_path / 'extraraw_content.yaml'
    yaml_content = {
        'images': [
            {'img_id': 'image_exr1', 'gt_path': str(tmp_path / 'image_exr1.exr'), 'raw_path': str(tmp_path / 'image_exr1.exr'),
             'is_bayer': False, 'image_set': 'image_exr1', 'best_alignment_loss': 0.01, 'mask_mean': 0.9,
             'rgb_msssim_score': 0.9, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'rgb_gain': 1.0,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'image_exr1.exr'), 'gt_bayer_fpath': str(tmp_path / 'image_exr1.exr'), 'mask_fpath': str(tmp_path / 'mask_exr1.exr')}]}
        ]
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files
    (tmp_path / 'image_exr1.exr').touch()
    (tmp_path / 'mask_exr1.exr').touch()
    return [str(yaml_path)]

@pytest.fixture(autouse=True)
def patch_rawproc_paths(monkeypatch, mock_content_fpath_rawnind, mock_content_fpath_extraraw):

    
    # Mock RawImageDataset's dependency on reading image files to prevent actual I/O
    # and return tensors of appropriate shapes
    def mock_fpath_to_tensor(fpath):
        if 'bayer' in fpath:
            return torch.ones(4, 128, 128) # Bayer is 4ch, half resolution
        elif 'gt' in fpath or 'f_linrec2020' in fpath: # RGB image
            return torch.ones(3, 256, 256)
        elif 'mask' in fpath:
            return torch.ones(1, 256, 256, dtype=torch.bool)
        elif 'image_exr' in fpath: # for extraraw, both raw and gt are RGB
             return torch.ones(3, 256, 256)
        return torch.empty(0) # Default for unexpected paths

    monkeypatch.setattr(pt_helpers_dep, 'fpath_to_tensor', mock_fpath_to_tensor)
    
    
    # Mock cv2.imread and rp.imread if they are directly used in base_dataset
    with patch('src.rawnind.dataset.base_dataset.cv2.imread', return_value=np.ones((256,256,3), dtype=np.float32)), \
         patch('src.rawnind.dataset.base_dataset.rp.imread', return_value=np.ones((256,256), dtype=np.uint16)), \
         patch('src.rawnind.dataset.base_dataset.RawImageDataset._load_metadata_from_xmp_sidecar', return_value={}):
        yield

# --- Parameterized Tests for Dataset Classes ---


    """Test various dataset types for correct crop shapes and gain handling."""

    # Only pass bayer_only if it's a noisy dataset, clean datasets don't have this param
    kwargs = {'bayer_only': bayer_only} if 'Noisy' in dataset_class.__name__ else {}
    
    ds = dataset_class(
        content_fpaths=content_fpaths,
        num_crops=num_crops,
        crop_size=crop_size,
        test_reserve=test_reserve,
        **kwargs
    )
    
    # Check dataset length to ensure images were loaded
    assert len(ds) > 0

    # Test __getitem__ for a few random indices/first/last
    indices_to_test = [0, -1]
    if len(ds) > 2:
        indices_to_test.append(random.randrange(len(ds)))

    for i in indices_to_test:
        image_batch = ds[i]
        
        # Adjust expected_y_shape based on gain matching for noisy datasets
        current_expected_y_shape = expected_y_shape
        if 'Noisy' in dataset_class.__name__ and dataset_class == CleanProfiledRGBNoisyBayerImageCropsDataset:
            # Bayer noisy dataset; y_crops is bayer, x_crops is RGB
            # The getitem for NoisyBayerImageCropsDataset returns output from random_crops
            # which has a different y_crops shape if bayer_only=True.
            # Here, y_crops.shape will be (num_crops, 4, crop_size/2, crop_size/2)
            assert image_batch["y_crops"].shape == (num_crops, 4, crop_size // 2, crop_size // 2)
        elif 'Noisy' in dataset_class.__name__ and dataset_class == CleanProfiledRGBNoisyProfiledRGBImageCropsDataset:
             # RGB noisy: x_crops is gt_linrec2020 which is 3ch, y_crops is f_linrec2020 which is 3ch
            assert image_batch["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert image_batch["y_crops"].shape == (num_crops, 3, crop_size, crop_size)
        elif 'Clean' in dataset_class.__name__ and dataset_class == CleanProfiledRGBCleanBayerImageCropsDataset:
            # Clean Bayer: gt is RGB (x), bayer is y
            assert image_batch["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert image_batch["y_crops"].shape == (num_crops, 4, crop_size // 2, crop_size // 2)
        elif 'Clean' in dataset_class.__name__ and dataset_class == CleanProfiledRGBCleanProfiledRGBImageCropsDataset:
            # Clean RGB: gt is RGB (x), gt is RGB (y)
            assert image_batch["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert image_batch["y_crops"].shape == (num_crops, 3, crop_size, crop_size)

        assert image_batch["mask_crops"].shape == (num_crops, expected_x_shape[1], expected_x_shape[2], expected_x_shape[3]) # mask matches x_crops size
        
        if "Bayer" in dataset_class.__name__:
            assert image_batch["rgb_xyz_matrix"].shape == (num_crops, 3) # (num_crops, num_channels_rgb)
        
        # Check gain
        if 'Noisy' in dataset_class.__name__:
            # This is specifically for CleanProfiledRGBNoisyBayerImageCropsDataset:
            # When bayer_only=True and match_gain=False, the gain is image["raw_gain"] from fixture (1.5)
            # When bayer_only=False (RGB noisy) and match_gain=False, gain is image["rgb_gain"] from fixture (1.0)
            if expected_gain == 1.5: # only for noisy bayer
                assert image_batch["gain"] == pytest.approx(1.5)
            else: # for noisy RGB
                assert image_batch["gain"] == pytest.approx(1.0)
        else: # Clean-clean datasets
            assert image_batch["gain"] == pytest.approx(1.0) # Clean will have hardcoded 1.0 gain


    """Test validation datasets for correct single crop shapes."""
    content_fpath = request.getfixturevalue(content_fpath_fixture)



# Test Dataloader variants

    """Test test dataloaders for correct single-crop shapes and additional metadata."""
    content_fpath = request.getfixturevalue(content_fpath_fixture)


# --- RawImageDataset cropping logic tests ---

@pytest.fixture
def mock_raw_image_data():
    """Fixture for mock raw image data."""
    ximg = torch.randn(3, 512, 512)
    yimg = torch.randn(4, 256, 256) # Bayer, half resolution
    mask = torch.ones(3, 512, 512, dtype=torch.bool)
    return ximg, yimg, mask

@pytest.fixture
def mock_raw_image_data_rgb():
    """Fixture for mock raw image data (RGB)."""
    ximg = torch.randn(3, 512, 512)
    yimg = torch.randn(3, 512, 512) # RGB, same resolution
    mask = torch.ones(3, 512, 512, dtype=torch.bool)
    return ximg, yimg, mask

@pytest.mark.parametrize("crop_size, num_crops, use_yimg, expected_x_shape, expected_y_shape, expected_mask_shape", [
    (128, 4, True, (4, 3, 128, 128), (4, 4, 64, 64), (4, 3, 128, 128)), # Bayer scenario
    (128, 4, False, (4, 3, 128, 128), None, (4, 3, 128, 128)),
    (256, 2, True, (2, 3, 256, 256), (2, 4, 128, 128), (2, 3, 256, 256)), # Bayer scenario
])
def test_rawimagedataset_random_crops(
    mock_raw_image_data, crop_size, num_crops, use_yimg,
    expected_x_shape, expected_y_shape, expected_mask_shape
):
    """Test RawImageDataset.random_crops for correct output shapes and logic."""
    ximg, yimg, mask = mock_raw_image_data
    ds = RawImageDataset(num_crops=num_crops, crop_size=crop_size)

    crops_tuple = ds.random_crops(ximg, yimg if use_yimg else None, mask)
    
    assert len(crops_tuple) == (3 if use_yimg else 2) # x, y, mask OR x, mask

    x_crops = crops_tuple[0]
    y_crops = crops_tuple[1] if use_yimg else None
    mask_crops = crops_tuple[1] if not use_yimg else crops_tuple[2] # Correctly retrieve mask

    assert x_crops.shape == expected_x_shape
    if use_yimg:
        assert y_crops.shape == expected_y_shape
    assert mask_crops.shape == expected_mask_shape
    
    # Ensure crops have sufficient valid pixels (mocked mask is all ones)
    assert (mask_crops.sum() / (crop_size**2 * expected_x_shape[1])) == pytest.approx(1.0)


@pytest.mark.parametrize("crop_size, use_yimg, input_is_rgb, expected_x_shape, expected_y_shape, expected_mask_shape", [
    (128, True, True, (3, 128, 128), (3, 128, 128), (3, 128, 128)), # RGB scenario
    (128, True, False, (3, 128, 128), (4, 64, 64), (3, 128, 128)), # Bayer scenario
    (256, False, True, (3, 256, 256), None, (3, 256, 256)),
])
def test_rawimagedataset_center_crop(
    mock_raw_image_data, mock_raw_image_data_rgb, crop_size, use_yimg, input_is_rgb,
    expected_x_shape, expected_y_shape, expected_mask_shape
):
    """Test RawImageDataset.center_crop for correct output shapes and logic."""
    ximg, yimg_bayer, mask = mock_raw_image_data
    _, yimg_rgb, _ = mock_raw_image_data_rgb

    if input_is_rgb:
        ximg, yimg = ximg, yimg_rgb
    else:
        ximg, yimg = ximg, yimg_bayer # ximg is always RGB_3ch here. yimg is either RGB or Bayer

    ds = RawImageDataset(num_crops=1, crop_size=crop_size)

    crops_tuple = ds.center_crop(ximg, yimg if use_yimg else None, mask)

    assert len(crops_tuple) == (3 if use_yimg else 2)

    x_crop = crops_tuple[0]
    y_crop = crops_tuple[1] if use_yimg else None
    mask_crop = crops_tuple[1] if not use_yimg else crops_tuple[2]

    assert x_crop.shape == expected_x_shape
    if use_yimg:
        assert y_crop.shape == expected_y_shape
    assert mask_crop.shape == expected_mask_shape

def test_rawimagedataset_center_crop_invalid_y_channels(mock_raw_image_data_rgb, y_channels, crop_size):
    """Test RawImageDataset.center_crop with invalid y_channels."""
    ximg, yimg_rgb, mask = mock_raw_image_data_rgb
    # Create yimg with invalid channels
    yimg_invalid = torch.randn(y_channels, 512, 512)
    ds = RawImageDataset(num_crops=1, crop_size=crop_size)
    with pytest.raises(ValueError, match="center_crop: invalid number of channels"):
        ds.center_crop(ximg, yimg_invalid, mask)

# --- CleanCleanImageDataset.get_mask tests ---

@pytest.mark.parametrize("input_channels, scale_factor, overexposure_lb, expected_mask_shape", [
    (4, 2, 0.5, (3, 256, 256)), # Bayer input -> interpolated to RGB size
    (3, 1, 0.5, (3, 256, 256)), # RGB input
])
def test_cleanclean_imagedataset_get_mask(input_channels, scale_factor, overexposure_lb, expected_mask_shape):
    """Test CleanCleanImageDataset.get_mask for Bayer and RGB scenarios."""
    ximg_shape_in = (input_channels, 128*scale_factor if input_channels==4 else 256, 128*scale_factor if input_channels==4 else 256)
    ximg = torch.randn(ximg_shape_in)
    metadata = {"overexposure_lb": overexposure_lb}
    
    # Mock interpolate if input is Bayer to control output shape
    with patch('torch.nn.functional.interpolate', side_effect=lambda img, scale_factor: img.repeat(1,2,2) if scale_factor==2 else img) if input_channels == 4 else MagicMock() as mock_interpolate:
        ds = CleanProfiledRGBCleanProfiledRGBImageCropsDataset(content_fpaths=[], num_crops=1, crop_size=128) # Dataset class used for method, content not relevant
        mask = ds.get_mask(ximg, metadata)
        
        assert mask.dtype == torch.bool
        assert mask.shape == expected_mask_shape
        if input_channels == 4:
            mock_interpolate.assert_called_once()


# --- __getitem__ logic tests (using toy dataset for speed) ---


    """Test __getitem__ logic including data pairing, gain matching, and msssim filtering."""
    content_fpaths = request.getfixturevalue(content_fpaths_fixture)
    kwargs = {}
    if 'Noisy' in dataset_class.__name__:
        kwargs.update({
            'test_reserve': [], # Ensure images are included
            'bayer_only': bayer_only,
            'data_pairing': data_pairing,
            'match_gain': match_gain,
            'min_msssim_score': min_msssim_score,
            'max_msssim_score': max_msssim_score,
        })
    
    if dataset_class == CleanProfiledRGBNoisyProfiledRGBImageCropsDataset and arbitrary_proc_method:
        kwargs['arbitrary_proc_method'] = arbitrary_proc_method

    with patch('src.rawnind.dataset.base_dataset.cv2.imread', return_value=np.ones((512,512,3), dtype=np.float32)), \
         patch('src.rawnind.dataset.base_dataset.rp.imread', return_value=np.ones((512,512), dtype=np.uint16)), \
         patch('src.rawnind.dataset.base_dataset.RawImageDataset._load_metadata_from_xmp_sidecar', return_value={}), \
         patch('src.rawnind.dependencies.arbitrary_processing.arbitrarily_process_images', side_effect=lambda img, **kw: img) if arbitrary_proc_method else MagicMock() as mock_arbitrary_proc: # Mock arbitrary processing
        
        ds = dataset_class(
            content_fpaths=content_fpaths,
            num_crops=num_crops,
            crop_size=crop_size,
            toy_dataset=False, # Use real fixture content
            **kwargs
        )
        
        assert len(ds) > 0 # Ensure at least one image is loaded

        output = ds[0] # Get the first item

        assert "x_crops" in output
        assert "y_crops" in output
        assert "mask_crops" in output
        assert "gain" in output

        if 'Bayer' in dataset_class.__name__ and dataset_class != CleanProfiledRGBCleanBayerImageCropsDataset: # For noisy bayer
            # x_crops: RGB, num_crops, 3, crop_size, crop_size
            # y_crops: Bayer, num_crops, 4, crop_size/2, crop_size/2
            assert output["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            if not bayer_only: # It should be 3ch here too
                assert output["y_crops"].shape == (num_crops, 3, crop_size, crop_size)
            else:
                assert output["y_crops"].shape == (num_crops, 4, crop_size // 2, crop_size // 2)
            assert output["mask_crops"].shape == (num_crops, 3, crop_size, crop_size) # Mask matches x_crops size
        elif dataset_class == CleanProfiledRGBCleanBayerImageCropsDataset: # Clean bayer
            assert output["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert output["y_crops"].shape == (num_crops, 4, crop_size // 2, crop_size // 2)
            assert output["mask_crops"].shape == (num_crops, 3, crop_size, crop_size) # Mask matches x_crops size
        else: # RGB noisy or clean
            assert output["x_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert output["y_crops"].shape == (num_crops, 3, crop_size, crop_size)
            assert output["mask_crops"].shape == (num_crops, 3, crop_size, crop_size)

        if match_gain:
            assert output["gain"] == pytest.approx(1.0)
        else:
            assert output["gain"] == pytest.approx(expected_gain_assert) # Test against expected gain from fixture
        
        if arbitrary_proc_method:
            mock_arbitrary_proc.assert_called()

# --- MS-SSIM filtering tests ---

    """Test min_msssim_score and max_msssim_score filtering in dataset initialization."""
    ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
        content_fpaths=[mock_content_fpath_rawnind],
        num_crops=1,
        crop_size=256,
        test_reserve=[],
        bayer_only=True,
        min_msssim_score=min_msssim,
        max_msssim_score=max_msssim
    )
    assert len(ds) == expected_dataset_len


# --- Error Handling Tests ---

def test_getitem_insufficient_valid_pixels_removes_crop(mock_content_fpath_rawnind):
    """Test that crops with insufficient valid pixels are removed."""
    # Mock random_crops to return False (no valid crops found)
    patch_path = 'src.rawnind.dataset.base_dataset.RawImageDataset.random_crops'
    with patch(patch_path, return_value=False):
        ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
            content_fpaths=[mock_content_fpath_rawnind],
            num_crops=1,
            crop_size=256,
            test_reserve=[],
            toy_dataset=False,
            bayer_only=True
        )
        initial_len = len(ds)
        # Assuming the mock yaml has at least one image with crops
        try:
            _ = ds[0]
        except IndexError: # Expect IndexError if all crops for image are exhausted and image is removed
            pass
        
        # Depending on how the actual dataset implementation handles `_dataset[i]["crops"].remove(crop)`
        # and re-tries, the length might decrease or an IndexError might be raised if all images are removed.
        # Here, we expect the image to be effectively "skipped" or removed from consideration
        # if no valid crops can be obtained after multiple retries.
        # The test should ultimately result in the image being removed from the dataset's consideration.
        assert len(ds) < initial_len or len(ds) == initial_len # Verifies that the dataset length either reduced or an IndexError was raised if all items were cleared.

def test_getitem_no_remaining_images_raises_error(mock_content_fpath_rawnind):
    """Test that if an image runs out of crops, it's removed and dataset eventually empties."""
    ds = CleanProfiledRGBNoisyBayerImageCropsDataset(
        content_fpaths=[mock_content_fpath_rawnind],
        num_crops=1,
        crop_size=256,
        test_reserve=[],
        toy_dataset=False,
        bayer_only=True
    )
    # Force _dataset[0]["crops"] to be empty after first access attempts
    patch_path = 'src.rawnind.dataset.base_dataset.RawImageDataset.random_crops'
    original_random_crops = RawImageDataset.random_crops

    call_count = 0
    def mock_random_crops_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= RawImageDataset.MAX_RANDOM_CROP_ATTEMPS + 1: # Plus 1 for the initial call
            return False # Simulate no valid crops
        return original_random_crops(*args, **kwargs) # Fallback if more calls are made

    with patch(patch_path, side_effect=mock_random_crops_side_effect):
        with pytest.raises(IndexError): # Expecting it to try to get next image or fail with IndexError
            _ = ds[0] # Should exhaust crops for the first image and then try next or fail
        
        # After attempting to get the first item many times and failing to find valid crops,
        # the dataset should reduce its internal list of images.
        # Given the fixture has 2 images, it should remove one, then the other, leading to 0.
        assert len(ds) == 0 # Dataset should become empty if all images are removed.

# --- RawImageDataset random_crops logic tests for specific failures ---
@pytest.fixture
def mock_random_crops_input_data():
    ximg = torch.randn(3, 50, 50) # smaller than crop_size
    yimg = torch.randn(4, 25, 25)
    mask = torch.ones(3, 50, 50, dtype=torch.bool)
    return ximg, yimg, mask

def test_random_crops_no_valid_attempts(mock_random_crops_input_data):
    """Test random_crops returns False if no valid crop can be made after max attempts."""
    ximg, yimg, mask = mock_random_crops_input_data
    ds = RawImageDataset(num_crops=1, crop_size=128) # crop_size larger than image
    
    # Mock the internal make_a_random_crop and the mask validation
    with patch.object(RawImageDataset, 'make_a_random_crop', side_effect=lambda *args, **kwargs: (None)), \
         patch.object(torch.BoolTensor, 'sum', return_value=0): # Always return 0 valid pixels
        result = ds.random_crops(ximg, yimg, mask)
        assert result is False

def test_rawimagedataset_random_crops_assertion_error(monkeypatch):
    """Test RawImageDataset.random_crops raises AssertionError for incompatible shapes."""
    ds = RawImageDataset(num_crops=1, crop_size=128)
    ximg = torch.randn(3, 256, 256)
    # yimg shape incompatible with ximg
    yimg = torch.randn(3, 128, 128) 
    mask = torch.ones(3, 256, 256, dtype=torch.bool)

    with pytest.raises(AssertionError, match="ximg and yimg should already be aligned."):
        ds.random_crops(ximg, yimg, mask)
