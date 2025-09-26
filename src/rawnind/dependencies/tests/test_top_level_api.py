import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path
from rawnind.dependencies import raw_processing as raw
import OpenImageIO as oiio

def test_raw_fpath_to_rggb_img_and_metadata():
    """
    Test raw_fpath_to_rggb_img_and_metadata function.

    Objective: Verify complete RAW to RGGB conversion pipeline.
    Test criteria: Returns RGGB image and metadata, handles Bayer pattern conversion.
    How testing fulfills purpose: Ensures end-to-end RAW loading matches domain knowledge.
    Components mocked: rawpy.imread to avoid real file I/O.
    Reason for hermeticity: RAW file processing requires external files/libraries.
    """
    # Mock rawpy image
    mock_raw = MagicMock()
    mock_raw.raw_image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
    mock_sizes_dict = {'raw_width': 512, 'raw_height': 512, 'width': 512, 'height': 512, 'iwidth': 512, 'iheight': 512, 'top_margin': 0, 'left_margin': 0}
    mock_raw.sizes = MagicMock()
    mock_raw.sizes._asdict.return_value = mock_sizes_dict
    mock_raw.color_desc = b'RGBG'
    mock_raw.camera_whitebalance = np.array([1.0, 1.0, 1.0, 1.0])
    mock_raw.camera_white_level_per_channel = np.array([65535, 65535, 65535, 65535])
    mock_raw.daylight_whitebalance = np.array([2.0, 1.0, 1.5, 1.2])
    mock_raw.black_level_per_channel = np.array([0, 0, 0, 0])
    mock_raw.white_level = 65535
    mock_raw.raw_pattern = [[0, 1], [2, 3]]  # RGGB pattern
    mock_raw.raw_colors = np.array([[0, 1], [2, 3]])  # RGGB pattern
    mock_raw.rgb_xyz_matrix = np.eye(3)

    with patch('rawnind.dependencies.raw_processing.rawpy.imread') as mock_imread:
        mock_imread.return_value = mock_raw

        rggb_img, metadata = raw.raw_fpath_to_rggb_img_and_metadata('dummy.cr2')

        assert isinstance(rggb_img, np.ndarray)
        assert rggb_img.shape[0] == 4  # RGGB channels
        assert isinstance(metadata, raw.Metadata)
        assert metadata.bayer_pattern == raw.BayerPattern.RGGB


def test_is_exposure_ok():
    """
    Test exposure checking functionality.

    Objective: Verify exposure validation logic works correctly.
    Test criteria: Detects over/under exposed images based on thresholds.
    How testing fulfills purpose: Ensures exposure checking matches domain knowledge.
    Components mocked: None - pure numpy operations.
    Reason for hermeticity: Deterministic threshold comparisons.
    """
    # Create test mono image and RGGB conversion
    mono_img = np.ones((1, 4, 4), dtype=np.float32) * 0.5  # Normal exposure

    metadata = raw.Metadata(
        fpath='test.cr2',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [2, 3]]),
        sizes={'raw_width': 4, 'raw_height': 4},
        camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        black_level_per_channel=np.array([0, 0, 0, 0]),
        white_level=1,
        camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        rgb_xyz_matrix=np.eye(3),
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
    )

    # Test normal exposure
    assert raw.is_exposure_ok(mono_img, metadata)

    # Test overexposed image
    overexposed_img = np.ones((1, 4, 4), dtype=np.float32) * 0.99  # Near clipping
    assert not raw.is_exposure_ok(overexposed_img, metadata, oe_threshold=0.98)

    # Test underexposed image
    underexposed_img = np.zeros((1, 4, 4), dtype=np.float32)
    assert not raw.is_exposure_ok(underexposed_img, metadata, ue_threshold=0.01)


def test_is_xtrans():
    """
    Test X-Trans file type detection.

    Objective: Verify X-Trans file detection works correctly.
    Test criteria: Correctly identifies .raf files as X-Trans.
    How testing fulfills purpose: Ensures file type detection matches domain knowledge.
    Components mocked: None - pure string operations.
    Reason for hermeticity: Deterministic file extension checking.
    """
    assert raw.is_xtrans("image.raf")
    assert raw.is_xtrans("IMAGE.RAF")  # Case insensitive
    assert raw.is_xtrans("/path/to/image.raf")

    assert not raw.is_xtrans("image.cr2")
    assert not raw.is_xtrans("image.arw")
    assert not raw.is_xtrans("image.jpg")
    assert not raw.is_xtrans("raf")  # No extension
    assert not raw.is_xtrans("")  # Empty string


@patch('subprocess.call')
@patch('shutil.which')
def test_xtrans_fpath_to_OpenEXR(mock_which, mock_subprocess):
    """
    Test X-Trans to OpenEXR conversion.

    Objective: Verify X-Trans conversion calls correct external tools.
    Test criteria: Calls darktable-cli with correct parameters for X-Trans files.
    How testing fulfills purpose: Ensures X-Trans processing matches domain knowledge.
    Components mocked: subprocess.call and shutil.which for external tool testing.
    Reason for hermeticity: Avoids requiring darktable-cli installation.
    """
    mock_which.return_value = "/usr/bin/darktable-cli"

    src_path = "/path/to/image.raf"
    dest_path = "/path/to/output.exr"

    raw.xtrans_fpath_to_OpenEXR(src_path, dest_path)

    mock_subprocess.assert_called_once_with((
        "darktable-cli",
        src_path,
        str(raw.Path("src/rawnind/dependencies/configs") / "dt4_xtrans_to_linrec2020.xmp"),
        dest_path,
        "--core",
        "--conf",
        "plugins/imageio/format/exr/bpp=16",
    ))


@patch('rawnind.dependencies.raw_processing.is_exposure_ok', return_value=True) # Mock this to prevent second BayerProcessor call
@patch('rawnind.dependencies.raw_processing.xtrans_fpath_to_OpenEXR')
@patch('rawnind.dependencies.raw_processing.RawLoader')
@patch('rawnind.dependencies.raw_processing.BayerProcessor')
@patch('rawnind.dependencies.raw_processing.ColorTransformer')
@patch('rawnind.dependencies.raw_processing.hdr_nparray_to_file')
def test_raw_fpath_to_hdr_img_file_success(mock_hdr_save, mock_color_transformer,
                                           mock_bayer_processor, mock_raw_loader, mock_xtrans,
                                           mock_is_exposure_ok): # Added mock_is_exposure_ok
    """
    Test full RAW to HDR conversion pipeline (success case).

    Objective: Verify complete RAW processing pipeline works end-to-end.
    Test criteria: Orchestrates all components correctly for successful conversion.
    How testing fulfills purpose: Ensures full pipeline integration matches domain knowledge.
    Components mocked: All external operations and file I/O.
    Reason for hermeticity: Avoids real file operations and external dependencies.
    """
    # Setup mocks
    mock_loader_instance = MagicMock()
    mock_loader_instance.load_raw_data.return_value = (
        np.random.rand(1, 256, 256).astype(np.float32),  # mono_img
        raw.Metadata(
            fpath='test.cr2',
            bayer_pattern=raw.BayerPattern.RGGB,
            rgbg_pattern=np.array([[0, 1], [2, 3]]),
            sizes={'raw_width': 256, 'raw_height': 256},
            camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            black_level_per_channel=np.array([0, 0, 0, 0]),
            white_level=1,
            camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            rgb_xyz_matrix=np.eye(3),
            overexposure_lb=1.0,
            camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
        )
    )
    mock_raw_loader.return_value = mock_loader_instance

    mock_processor_instance = MagicMock()
    mock_processor_instance.demosaic.return_value = np.random.rand(3, 256, 256).astype(np.float32)
    mock_processor_instance.mono_to_rggb.return_value = np.random.rand(4, 128, 128).astype(np.float32) # Mock mono_to_rggb
    mock_bayer_processor.return_value = mock_processor_instance

    mock_transformer_instance = MagicMock()
    mock_transformer_instance.camRGB_to_profiledRGB_img.return_value = np.random.rand(3, 256, 256).astype(np.float32)
    mock_color_transformer.return_value = mock_transformer_instance

    # Test successful conversion
    result = raw.raw_fpath_to_hdr_img_file(
        "test.cr2", "output.exr",
        output_color_profile="lin_rec2020",
        bit_depth=32
    )

    assert result[0] == "OK"
    assert result[1] == "test.cr2"
    assert result[2] == "output.exr"

    # Verify calls
    mock_raw_loader.assert_called_once()
    mock_bayer_processor.assert_called_once()
    mock_color_transformer.assert_called_once()
    mock_hdr_save.assert_called_once()


@patch('rawnind.dependencies.raw_processing.RawLoader')
def test_raw_fpath_to_hdr_img_file_exposure_failure(mock_raw_loader):
    """
    Test RAW to HDR conversion with exposure failure.

    Objective: Verify exposure checking prevents bad conversions.
    Test criteria: Returns BAD_EXPOSURE when image is over/under exposed.
    How testing fulfills purpose: Ensures quality control in pipeline.
    Components mocked: RawLoader to simulate exposure issues.
    Reason for hermeticity: Avoids real image processing.
    """
    mock_loader_instance = MagicMock()
    # Simulate overexposed image (values > 0.99)
    overexposed_img = np.ones((1, 256, 256), dtype=np.float32) * 0.995
    mock_loader_instance.load_raw_data.return_value = (
        overexposed_img,
        raw.Metadata(
            fpath='test.cr2',
            bayer_pattern=raw.BayerPattern.RGGB,
            rgbg_pattern=np.array([[0, 1], [2, 3]]),
            sizes={'raw_width': 256, 'raw_height': 256},
            camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            black_level_per_channel=np.array([0, 0, 0, 0]),
            white_level=1,
            camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            rgb_xyz_matrix=np.eye(3),
            overexposure_lb=1.0,
            camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
        )
    )
    mock_raw_loader.return_value = mock_loader_instance

    result = raw.raw_fpath_to_hdr_img_file("test.cr2", "output.exr", check_exposure=True)

    assert result[0] == "BAD_EXPOSURE"


def test_is_exposure_ok():
    """
    Test exposure checking functionality.

    Objective: Verify exposure validation logic works correctly.
    Test criteria: Detects over/under exposed images based on thresholds.
    How testing fulfills purpose: Ensures exposure checking matches domain knowledge.
    Components mocked: None - pure numpy operations.
    Reason for hermeticity: Deterministic threshold comparisons.
    """
    # Create test mono image and RGGB conversion
    mono_img = np.ones((1, 4, 4), dtype=np.float32) * 0.5  # Normal exposure

    metadata = raw.Metadata(
        fpath='test.cr2',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [2, 3]]),
        sizes={'raw_width': 4, 'raw_height': 4},
        camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        black_level_per_channel=np.array([0, 0, 0, 0]),
        white_level=1,
        camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        rgb_xyz_matrix=np.eye(3),
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
    )

    # Test normal exposure
    assert raw.is_exposure_ok(mono_img, metadata)

    # Test overexposed image
    overexposed_img = np.ones((1, 4, 4), dtype=np.float32) * 0.99  # Near clipping
    assert not raw.is_exposure_ok(overexposed_img, metadata, oe_threshold=0.98)

    # Test underexposed image
    underexposed_img = np.zeros((1, 4, 4), dtype=np.float32)
    assert not raw.is_exposure_ok(underexposed_img, metadata, ue_threshold=0.01)


def test_is_xtrans():
    """
    Test X-Trans file type detection.

    Objective: Verify X-Trans file detection works correctly.
    Test criteria: Correctly identifies .raf files as X-Trans.
    How testing fulfills purpose: Ensures file type detection matches domain knowledge.
    Components mocked: None - pure string operations.
    Reason for hermeticity: Deterministic file extension checking.
    """
    assert raw.is_xtrans("image.raf")
    assert raw.is_xtrans("IMAGE.RAF")  # Case insensitive
    assert raw.is_xtrans("/path/to/image.raf")

    assert not raw.is_xtrans("image.cr2")
    assert not raw.is_xtrans("image.arw")
    assert not raw.is_xtrans("image.jpg")
    assert not raw.is_xtrans("raf")  # No extension
    assert not raw.is_xtrans("")  # Empty string


@patch('subprocess.call')
@patch('shutil.which')
def test_xtrans_fpath_to_OpenEXR(mock_which, mock_subprocess):
    """
    Test X-Trans to OpenEXR conversion.

    Objective: Verify X-Trans conversion calls correct external tools.
    Test criteria: Calls darktable-cli with correct parameters for X-Trans files.
    How testing fulfills purpose: Ensures X-Trans processing matches domain knowledge.
    Components mocked: subprocess.call and shutil.which for external tool testing.
    Reason for hermeticity: Avoids requiring darktable-cli installation.
    """
    mock_which.return_value = "/usr/bin/darktable-cli"

    src_path = "/path/to/image.raf"
    dest_path = "/path/to/output.exr"

    raw.xtrans_fpath_to_OpenEXR(src_path, dest_path)

    mock_subprocess.assert_called_once_with((
        "darktable-cli",
        src_path,
        str(raw.Path("src/rawnind/dependencies/configs") / "dt4_xtrans_to_linrec2020.xmp"),
        dest_path,
        "--core",
        "--conf",
        "plugins/imageio/format/exr/bpp=16",
    ))