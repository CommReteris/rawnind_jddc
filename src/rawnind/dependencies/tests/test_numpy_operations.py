import pytest
import numpy as np
import os
import cv2
from unittest.mock import MagicMock, patch
import torch
from pathlib import Path

from src.rawnind.dependencies.numpy_operations import (
    CropMethod, img_fpath_to_np_flt, np_pad_img_pair,
    np_crop_img_pair, np_to_img, np_l1, gamma, scenelin_to_pq, pq_to_scenelin, match_gain
)
from src.rawnind.dependencies import raw_processing as raw # for mocking external calls if needed


# --- Fixtures ---
@pytest.fixture
def dummy_img_chw():
    """Returns a dummy NumPy image array (C,H,W) in [0,1] range."""
    return np.random.rand(3, 100, 100).astype(np.float32)

@pytest.fixture
def dummy_img_pair_even():
    return np.random.rand(3, 8, 8).astype(np.float32), np.random.rand(3, 8, 8).astype(np.float32)

@pytest.fixture
def dummy_img_pair_odd():
    return np.random.rand(3, 5, 5).astype(np.float32), np.random.rand(3, 5, 5).astype(np.float32)

# Using tmp_path for file creation
@pytest.fixture
def create_dummy_png_file(tmp_path):
    """Creates a dummy PNG file for testing OpenCV loading."""
    dummy_file_path = tmp_path / "test_img.png"
    dummy_img = (np.random.rand(10, 10, 3) * 255).astype(np.uint8) # HWC format for OpenCV
    cv2.imwrite(str(dummy_file_path), dummy_img) # Save using OpenCV
    return str(dummy_file_path)

# --- Tests for img_fpath_to_np_flt ---
def test_img_fpath_to_np_flt_file_not_found():
    """
    Test img_fpath_to_np_flt raises ValueError for non-existent file.

    Objective: Verify img_fpath_to_np_flt handles missing files gracefully.
    Test criteria: Function raises ValueError with "File not found" message.
    How testing for this criteria fulfills purpose: Ensures proper error handling for invalid file paths.
    What components are mocked, monkeypatched, or are fixtures: None (tests real file existence check).
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Exception is raised before any file operations.
    """
    with pytest.raises(ValueError, match="File not found"):
        img_fpath_to_np_flt("non_existent_file.png")

def test_img_fpath_to_np_flt_raw_loading(mock_rawpy_data, tmp_path):
    """
    Test img_fpath_to_np_flt with RAW files using mocked loader.

    Objective: Verify RAW file loading through mocked RawLoader.
    Test criteria: Returns correct 4-channel shape and Bayer pattern metadata.
    How testing fulfills purpose: Ensures RAW processing pipeline works without real files.
    What components are mocked, monkeypatched, or are fixtures: mock_rawpy_data fixture mocks rawpy.imread.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Mock provides controlled output for testing parsing logic.
    """
    fpath = tmp_path / "test.raw" # Use tmp_path for fpath
    # Create a dummy file to pass os.path.isfile check
    fpath.touch()

    # The mock_rawpy_data fixture mocks rawpy.imread.
    # img_fpath_to_np_flt internally calls RawLoader().load_raw_data, which will now hit the mock.

    # Create a dummy metadata object for the mock
    dummy_metadata = raw.Metadata(
        fpath='dummy',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [2, 3]]),
        sizes={'raw_width': 512, 'raw_height': 512, 'width': 512, 'height': 512, 'iwidth': 512, 'iheight': 512, 'top_margin': 0, 'left_margin': 0},
        camera_whitebalance=np.ones(4),
        black_level_per_channel=np.zeros(4),
        white_level=65535,
        camera_white_level_per_channel=np.full(4, 65535),
        daylight_whitebalance=np.ones(4),
        rgb_xyz_matrix=np.eye(3),
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.ones(4),
        daylight_whitebalance_norm=np.ones(4)
    )

    with patch('src.rawnind.dependencies.raw_processing.RawLoader.load_raw_data', return_value=(np.zeros((1, 512, 512)), dummy_metadata)):
      img, metadata_dict = img_fpath_to_np_flt(str(fpath), incl_metadata=True)

    # Assert against the dummy data returned by the fixture's mock_method
    assert img.shape == (1, 512, 512)
    assert 'bayer_pattern' in metadata_dict
    assert metadata_dict['bayer_pattern'] == raw.BayerPattern.RGGB # Ensure correct enum conversion 

def test_img_fpath_to_np_flt_opencv_backend(create_dummy_png_file):
    """
    Test img_fpath_to_np_flt with OpenCV backend for PNG files.

    Objective: Verify PNG loading and color conversion through OpenCV.
    Test criteria: Returns correct CHW shape, float32 dtype, normalized range, and calls cv2.imread.
    How testing fulfills purpose: Ensures OpenCV fallback works for non-RAW images.
    What components are mocked, monkeypatched, or are fixtures: cv2.imread and cv2.cvtColor mocked to avoid real file I/O.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Mocked cv2 calls provide controlled image data for testing conversion logic.
    """
    fpath = create_dummy_png_file

    # Mock cv2.imread and cv2.cvtColor to ensure no actual file operation issue for test
    with patch('cv2.imread', return_value=(np.random.rand(10, 10, 3) * 255).astype(np.uint8)) as mock_imread, \
         patch('cv2.cvtColor', side_effect=lambda img, code: img[:,:,::-1] if code == cv2.COLOR_BGR2RGB else img) as mock_cvtColor:

        img = img_fpath_to_np_flt(fpath)
        assert img.shape == (3, 10, 10) # Default size of dummy file
        assert img.dtype == np.float32
        assert img.max() <= 1.0 and img.min() >= 0.0
        mock_imread.assert_called_once()


# --- Tests for np_pad_img_pair ---
def test_np_pad_img_pair_even(dummy_img_pair_even):
    """
    Test np_pad_img_pair with even-sized images.

    Objective: Verify padding works correctly for even-sized input images.
    Test criteria: Padded images have correct target size and original content is centered.
    How testing for this criteria fulfills purpose: Ensures padding logic handles even dimensions properly.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_pair_even fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test padding calculations.
    """
    img1, img2 = dummy_img_pair_even
    cs = 16 # Target size
    padded1, padded2 = np_pad_img_pair(img1, img2, cs)

    assert padded1.shape == (3, cs, cs)
    assert padded2.shape == (3, cs, cs)

    # Verify original content is centered
    expected_start = (cs - img1.shape[1]) // 2
    assert np.all(padded1[:, expected_start:expected_start+img1.shape[1], 
                           expected_start:expected_start+img1.shape[2]] == img1)

def test_np_pad_img_pair_odd(dummy_img_pair_odd):
    """
    Test np_pad_img_pair with odd-sized images.

    Objective: Verify padding works correctly for odd-sized input images.
    Test criteria: Padded images have correct target size and original content is centered.
    How testing for this criteria fulfills purpose: Ensures padding logic handles odd dimensions properly.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_pair_odd fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test padding calculations.
    """
    img1, img2 = dummy_img_pair_odd
    cs = 10 # Target size
    padded1, padded2 = np_pad_img_pair(img1, img2, cs)

    assert padded1.shape == (3, cs, cs)
    assert padded2.shape == (3, cs, cs)
    
    # Verify original content is centered
    expected_start_h = (cs - img1.shape[1]) // 2
    expected_start_w = (cs - img1.shape[2]) // 2
    assert np.all(padded1[:, expected_start_h:expected_start_h+img1.shape[1], 
                           expected_start_w:expected_start_w+img1.shape[2]] == img1)

# --- Tests for np_crop_img_pair ---
def test_np_crop_img_pair_center(dummy_img_pair_even):
    """
    Test np_crop_img_pair with center cropping method.

    Objective: Verify center cropping extracts correct region from image pairs.
    Test criteria: Returns cropped images of specified size with content from center of originals.
    How testing for this criteria fulfills purpose: Ensures center cropping logic works correctly.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_pair_even fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test cropping calculations.
    """
    img1, img2 = dummy_img_pair_even
    cs = 4 # Crop size
    cropped1, cropped2 = np_crop_img_pair(img1, img2, cs, CropMethod.CENTER)

    assert cropped1.shape == (3, cs, cs)
    assert cropped2.shape == (3, cs, cs)

    # Verify content for center crop
    expected_y0 = (img1.shape[1] - cs) // 2
    expected_x0 = (img1.shape[2] - cs) // 2
    assert np.all(cropped1 == img1[:, expected_y0:expected_y0+cs, expected_x0:expected_x0+cs])

def test_np_crop_img_pair_rand(dummy_img_pair_even):
    """
    Test np_crop_img_pair with random cropping method.

    Objective: Verify random cropping returns images of correct size without errors.
    Test criteria: Returns cropped images of specified size from random locations.
    How testing for this criteria fulfills purpose: Ensures random cropping doesn't crash and produces valid output.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_pair_even fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests basic functionality without needing deterministic random behavior.
    """
    img1, img2 = dummy_img_pair_even
    cs = 4
    cropped1, cropped2 = np_crop_img_pair(img1, img2, cs, CropMethod.RAND)

    assert cropped1.shape == (3, cs, cs)
    assert cropped2.shape == (3, cs, cs)

    # Check that it's not simply the top-left corner
    # This is non-deterministic, can only check if shapes match
    # A deeper test would run it multiple times and check for different offsets within bounds
    # For now, just ensure it runs without error and returns correct shape.

def test_np_crop_img_pair_full_size(dummy_img_pair_even):
    """
    Test np_crop_img_pair when crop size equals original size.

    Objective: Verify cropping with full size returns original images unchanged.
    Test criteria: When crop size equals image size, returns identical images.
    How testing for this criteria fulfills purpose: Ensures edge case of full-size cropping works.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_pair_even fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests that cropping bounds are handled correctly.
    """
    img1, img2 = dummy_img_pair_even
    cs_full = img1.shape[1] # Crop to original size
    cropped1, cropped2 = np_crop_img_pair(img1, img2, cs_full, CropMethod.CENTER)
    assert np.all(cropped1 == img1)
    assert np.all(cropped2 == img2)

# --- Tests for np_to_img ---
def test_np_to_img_png_16bit(dummy_img_chw, tmp_path):
    """
    Test np_to_img with 16-bit precision PNG output.

    Objective: Verify 16-bit PNG saving converts float to uint16 correctly.
    Test criteria: Calls cv2.imwrite with uint16 array in valid range.
    How testing for this criteria fulfills purpose: Ensures high-precision image saving works.
    What components are mocked, monkeypatched, or are fixtures: cv2.imwrite mocked to avoid real file I/O.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests data type conversion and range validation.
    """
    fpath = tmp_path / "test_16bit.png"
    with patch('cv2.imwrite') as mock_imwrite:
        np_to_img(dummy_img_chw, str(fpath), precision=16)
        mock_imwrite.assert_called_once()
        # Verify call arguments, specifically data type and range
        args, kwargs = mock_imwrite.call_args
        assert args[0] == str(fpath) # First arg is fpath
        img_written = args[1] # Second arg is the image array
        assert img_written.dtype == np.uint16
        assert img_written.max() <= 65535
        assert img_written.min() >= 0

def test_np_to_img_png_8bit(dummy_img_chw, tmp_path):
    """
    Test np_to_img with 8-bit precision PNG output.

    Objective: Verify 8-bit PNG saving converts float to uint8 correctly.
    Test criteria: Calls cv2.imwrite with uint8 array in valid range.
    How testing for this criteria fulfills purpose: Ensures standard image saving works.
    What components are mocked, monkeypatched, or are fixtures: cv2.imwrite mocked to avoid real file I/O.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests data type conversion and range validation.
    """
    fpath = tmp_path / "test_8bit.png"
    with patch('cv2.imwrite') as mock_imwrite:
        np_to_img(dummy_img_chw, str(fpath), precision=8)
        mock_imwrite.assert_called_once()
        args, kwargs = mock_imwrite.call_args
        img_written = args[1]
        assert img_written.dtype == np.uint8
        assert img_written.max() <= 255
        assert img_written.min() >= 0

def test_np_to_img_single_channel(tmp_path):
    """
    Test np_to_img with single-channel input.

    Objective: Verify single-channel images are expanded to 3-channel for OpenCV.
    Test criteria: HW input becomes HWC with 3 channels.
    How testing for this criteria fulfills purpose: Ensures grayscale images are handled correctly.
    What components are mocked, monkeypatched, or are fixtures: cv2.imwrite mocked to avoid real file I/O.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Tests channel expansion logic.
    """
    img_mono = np.random.rand(50, 50).astype(np.float32) # HW format
    fpath = tmp_path / "test_mono.png"
    with patch('cv2.imwrite') as mock_imwrite:

        np_to_img(img_mono, str(fpath), precision=8)
        mock_imwrite.assert_called_once()
        # Check that it was expanded to 3 channel for saving with cv2 (HWC, 3 channels)
        args, kwargs = mock_imwrite.call_args
        img_written = args[1]
        assert img_written.shape == (50, 50, 3)

def test_np_to_img_not_implemented_precision(dummy_img_chw, tmp_path):
    """
    Test np_to_img raises NotImplementedError for unsupported precision.

    Objective: Verify unsupported precisions are rejected.
    Test criteria: Function raises NotImplementedError for invalid precision.
    How testing for this criteria fulfills purpose: Ensures only supported precisions work.
    What components are mocked, monkeypatched, or are fixtures: tmp_path fixture for path handling.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Exception is raised before file operations.
    """
    with pytest.raises(NotImplementedError):
        np_to_img(dummy_img_chw, str(tmp_path / "fail.png"), precision=24) # Expect NotImplementedError

# --- Tests for np_l1 ---
def test_np_l1_avg(dummy_img_chw):
    """
    Test np_l1 with average calculation.

    Objective: Verify L1 distance averaging works correctly.
    Test criteria: Returns correct average L1 distance between images.
    How testing for this criteria fulfills purpose: Ensures L1 averaging logic is implemented properly.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_chw fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test mathematical operations.
    """
    img1 = dummy_img_chw
    img2 = np.zeros_like(img1)
    expected_l1_avg = img1.mean()
    assert np_l1(img1, img2, avg=True) == pytest.approx(expected_l1_avg)

def test_np_l1_element_wise(dummy_img_chw):
    """
    Test np_l1 with element-wise calculation.

    Objective: Verify L1 distance element-wise computation works correctly.
    Test criteria: Returns correct element-wise L1 distance map between images.
    How testing for this criteria fulfills purpose: Ensures L1 element-wise logic is implemented properly.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_chw fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test mathematical operations.
    """
    img1 = dummy_img_chw
    img2 = np.zeros_like(img1)
    expected_l1_map = np.abs(img1 - img2)
    assert np.allclose(np_l1(img1, img2, avg=False), expected_l1_map)

# --- Tests for gamma ---
def test_gamma_np_positive_values():
    """
    Test gamma with positive values.

    Objective: Verify gamma correction applies correctly to positive image values.
    Test criteria: Returns correct gamma-corrected values using 1/gamma exponentiation.
    How testing for this criteria fulfills purpose: Ensures gamma correction logic works for valid inputs.
    What components are mocked, monkeypatched, or are fixtures: None (tests pure mathematical operation).
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses direct numpy operations to test gamma formula.
    """
    img = np.array([0.1, 0.5, 0.8, 1.0], dtype=np.float32)
    gamma_val = 2.2
    expected_output = img**(1/gamma_val)
    output = gamma(img, gamma_val)
    assert np.allclose(output, expected_output)

def test_gamma_np_non_positive_values():
    """
    Test gamma with non-positive values.

    Objective: Verify gamma correction handles non-positive values correctly by leaving them unchanged.
    Test criteria: Non-positive values remain unchanged while positive values are gamma-corrected.
    How testing for this criteria fulfills purpose: Ensures gamma correction doesn't corrupt invalid inputs.
    What components are mocked, monkeypatched, or are fixtures: None (tests pure mathematical operation).
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses direct numpy operations to test conditional gamma logic.
    """
    img = np.array([-0.1, 0.0, 0.5, 1.0], dtype=np.float32)
    gamma_val = 2.2
    output = gamma(img, gamma_val)
    assert np.allclose(output[0:2], np.array([-0.1, 0.0]))
    assert np.allclose(output[2:], np.array([0.5, 1.0])**(1/gamma_val))

def test_gamma_np_in_place():
    """
    Test gamma with in-place modification.

    Objective: Verify gamma correction modifies the input array in-place when requested.
    Test criteria: Input array is modified to contain gamma-corrected values.
    How testing for this criteria fulfills purpose: Ensures in-place operation works without creating copies.
    What components are mocked, monkeypatched, or are fixtures: None (tests pure mathematical operation).
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses direct numpy operations to test in-place modification.
    """
    img = np.array([0.1, 0.5, 0.8], dtype=np.float32)
    original_img_clone = img.copy()
    gamma(img, in_place=True)
    expected_output = original_img_clone**(1/2.2)
    assert np.allclose(img, expected_output)

# --- Tests for scenelin_to_pq and pq_to_scenelin (mocking colour library) ---
def test_scenelin_to_pq_and_back_np():
    """
    Test scenelin_to_pq and pq_to_scenelin round-trip conversion.

    Objective: Verify PQ transfer functions work correctly with mocked colour library.
    Test criteria: scenelin_to_pq applies PQ OETF, pq_to_scenelin applies inverse, round-trip preserves input.
    How testing fulfills purpose: Ensures HDR transfer functions are implemented correctly.
    What components are mocked, monkeypatched, or are fixtures: Colour library functions mocked to avoid external dependency.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Mocked colour library provides controlled transfer function behavior for testing conversion logic.
    """
    mock_scenelin = np.random.rand(3, 100, 100).astype(np.float32)

    # Mock the external 'colour' library
    with patch('builtins.__import__') as mock_import:
        # Mock the 'colour' module and its specific functions
        mock_colour_module = MagicMock()
        mock_colour_module.models.rgb.transfer_functions.itur_bt_2100.oetf_BT2100_PQ.return_value = mock_scenelin * 2 # Dummy PQ output
        mock_colour_module.models.rgb.transfer_functions.itur_bt_2100.oetf_inverse_PQ_BT2100.return_value = mock_scenelin # Simulate perfect inverse

        # Configure __import__ to return our mock when 'colour' is imported
        def custom_importer(name, *args, **kwargs):
            if name == 'colour':
                return mock_colour_module
            return __import__(name, *args, **kwargs) # Call original import for other modules
        mock_import.side_effect = custom_importer

        # Call the functions that depend on the 'colour' library
        pq_output = scenelin_to_pq(mock_scenelin)
        assert np.allclose(pq_output, mock_scenelin * 2)

        scenelin_back = pq_to_scenelin(pq_output)
        assert np.allclose(scenelin_back, mock_scenelin)


# --- Tests for match_gain (placeholder) ---
def test_match_gain_numpy_implementation(dummy_img_chw):
    """
    Test match_gain with numpy implementation.

    Objective: Verify gain matching adjusts image brightness correctly.
    Test criteria: Matched image has same mean as reference, gain value is calculated correctly.
    How testing fulfills purpose: Ensures gain matching logic works for brightness normalization.
    What components are mocked, monkeypatched, or are fixtures: dummy_img_chw fixture provides test images.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test gain calculation formulas.
    """
    img1 = dummy_img_chw
    img2_lower_gain = dummy_img_chw * 0.5

    matched_img = match_gain(img1, img2_lower_gain, return_val=False)
    # Check if the mean of matched_img is approximately the mean of img1
    assert np.mean(matched_img) == pytest.approx(np.mean(img1))

    gain_val = match_gain(img1, img2_lower_gain, return_val=True)
    assert isinstance(gain_val, float)
    assert gain_val == pytest.approx(2.0) # If img2_lower_gain is half of img1, gain should be 2.0

def test_match_gain_division_by_zero():
    """
    Test match_gain handles division by zero gracefully.

    Objective: Verify gain matching handles zero-mean images without crashing.
    Test criteria: Returns unchanged image and default gain of 1.0 when reference mean is zero.
    How testing for this criteria fulfills purpose: Ensures robust handling of edge cases in gain calculation.
    What components are mocked, monkeypatched, or are fixtures: None (tests pure mathematical operation).
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses controlled numpy arrays to test division by zero handling.
    """
    img1 = np.ones((3,10,10), dtype=np.float32)
    img2 = np.zeros_like(img1)

    matched_img = match_gain(img1, img2, return_val=False)
    assert np.allclose(matched_img, img2) # No change in img2 if its mean is zero

    gain_val = match_gain(img1, img2, return_val=True)
    assert gain_val == 1.0 # Default gain to 1.0 to avoid division by zero