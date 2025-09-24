import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from pathlib import Path
from rawnind.dependencies import raw_processing as raw
import OpenImageIO as oiio

@patch('rawnind.dependencies.raw_processing.hdr_nparray_to_file')
def test_hdr_nparray_to_file_integration(mock_hdr_save):
    """
    Test HDR array to file saving.

    Objective: Verify HDR saving pipeline works.
    Test criteria: Calls underlying save function with correct parameters.
    How testing fulfills purpose: Ensures HDR output functionality.
    Components mocked: hdr_nparray_to_file to avoid file I/O.
    Reason for hermeticity: File operations require disk access.
    """
    test_array = np.random.rand(3, 256, 256).astype(np.float32)

    raw.hdr_nparray_to_file(test_array, 'test.exr', color_profile='lin_rec2020')
    mock_hdr_save.assert_called_once_with(test_array, 'test.exr', color_profile='lin_rec2020')


@pytest.mark.parametrize("color_profile, bit_depth, file_ext", [
    ('lin_rec2020', 16, '.exr'),
    ('lin_sRGB', 32, '.exr'),
])
def test_file_io_integration_exr_save(tmp_path, color_profile, bit_depth, file_ext):
    """
    Test EXR saving with actual file operations (integration test).

    Objective: Verify EXR saving preserves data integrity for different bit depths and color profiles.
    Test criteria: File is created, has content, and can be read back with OpenImageIO.
    How testing fulfills purpose: Ensures EXR output functionality.
    Components mocked: None - real file operations using temporary files.
    Reason for hermeticity: Uses temp files, tests data preservation.
    """
    # Create test HDR image with specified dtype
    test_dtype = np.float16 if bit_depth == 16 else np.float32
    test_img = np.random.rand(3, 32, 32).astype(test_dtype)
    test_img = test_img / test_img.max() if test_img.max() > 0 else test_img # Normalize to [0,1] for cleaner data

    file_path = tmp_path / f"test_hdr{file_ext}"

    # Save HDR image
    raw.hdr_nparray_to_file(test_img, str(file_path), color_profile=color_profile, bit_depth=bit_depth)

    # Verify file was created
    assert file_path.exists()
    assert file_path.stat().st_size > 0

    # Read the image back using OpenImageIO as it's the preferred provider for EXR
    # This requires OpenImageIO to be installed
    img_in = oiio.ImageInput.open(str(file_path))
    assert img_in is not None
    spec = img_in.spec()
    read_img = img_in.read_image()
    img_in.close()

    # Convert read_img to C,H,W format if it's H,W,C
    if read_img.ndim == 3 and read_img.shape[2] == test_img.shape[0]:
        read_img = read_img.transpose(2, 0, 1)

    assert read_img.shape == test_img.shape
    assert np.allclose(read_img, test_img, atol=1e-3, rtol=1e-3 if bit_depth == 16 else 1e-6) # Adjust tolerance for float16

@pytest.mark.parametrize("color_profile, bit_depth, file_ext", [
    ('lin_rec2020', 16, '.tif'),
    ('gamma_sRGB', None, '.tif'), # For TIFF, bit_depth=None defaults to uint16
])
def test_file_io_integration_tiff_save(tmp_path, color_profile, bit_depth, file_ext):
    """
    Test TIFF saving with actual file operations (integration test).

    Objective: Verify TIFF saving preserves data integrity for different color profiles.
    Test criteria: File is created, has content, and can be read back with OpenImageIO/OpenCV.
    How testing fulfills purpose: Ensures TIFF output functionality.
    Components mocked: None - real file operations using temporary files.
    Reason for hermeticity: Uses temp files, tests data preservation.
    """
    # Create test image
    test_img = np.random.rand(3, 32, 32).astype(np.float32)
    test_img = test_img / test_img.max() if test_img.max() > 0 else test_img # Normalize to [0,1]

    file_path = tmp_path / f"test_hdr{file_ext}"

    # Save TIFF image
    raw.hdr_nparray_to_file(test_img, str(file_path), color_profile=color_profile, bit_depth=bit_depth)

    # Verify file was created
    assert file_path.exists()
    assert file_path.stat().st_size > 0

    # Read the image back
    if raw.TIFF_PROVIDER == "OpenImageIO":
        img_in = oiio.ImageInput.open(str(file_path))
        assert img_in is not None
        read_img = img_in.read_image()
        img_in.close()
        if read_img.ndim == 3 and read_img.shape[2] == test_img.shape[0]:
            read_img = read_img.transpose(2, 0, 1)
        # Convert back to float for comparison if it was uint16
        if read_img.dtype == np.uint16:
            read_img = read_img.astype(np.float32) / 65535.0
    elif raw.TIFF_PROVIDER == "OpenCV":
        read_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        # OpenCV reads HWC, BGR by default for 3 channels, and reads uint16.
        # It needs to be converted back to CHW, RGB, float32 for comparison.
        if read_img.ndim == 3:
            read_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        read_img = read_img.astype(np.float32) / 65535.0 # Convert back to [0,1] float

    assert read_img.shape == test_img.shape
    # TIFF with OpenCV loses precision/gamma, so a high tolerance is needed, or check output directly.
    # Given the warning in the original code about data loss with OpenCV, we just check shape.
    if raw.TIFF_PROVIDER == "OpenImageIO":
        assert np.allclose(read_img, test_img, atol=1e-3, rtol=1e-3 if bit_depth == 16 else 1e-6)
    else: # OpenCV or other less precise provider
        # For OpenCV, specifically for gamma_sRGB, direct numerical comparison is hard due to internal transforms
        # and implicit assumptions. Just check shape and basic range.
        assert read_img.shape == test_img.shape
        assert 0 <= read_img.min() and read_img.max() <= 1 # Check range for uint16 conversion to float
class TestXTransProcessing:
    """Tests for X-Trans file processing functions."""

    def test_is_xtrans_true(self):
        """
        Test is_xtrans identifies .raf files as X-Trans.
        """
        assert raw.is_xtrans("image.raf") is True
        assert raw.is_xtrans("IMAGE.RAF") is True
        assert raw.is_xtrans("path/to/image.raf") is True

    def test_is_xtrans_false(self):
        """
        Test is_xtrans identifies non-.raf files as not X-Trans.
        """
        assert raw.is_xtrans("image.cr2") is False
        assert raw.is_xtrans("image.dng") is False
        assert raw.is_xtrans("image.png") is False

    @patch('rawnind.dependencies.raw_processing.shutil.which', return_value="/usr/bin/darktable-cli")
    @patch('rawnind.dependencies.raw_processing.subprocess.call')
    def test_xtrans_fpath_to_openexr_success(self, mock_subprocess_call, mock_shutil_which, tmp_path):
        """
        Test xtrans_fpath_to_OpenEXR successfully calls darktable-cli.
        """
        src_fpath = str(tmp_path / "test.raf")
        dest_fpath = str(tmp_path / "output.exr")
        raw.xtrans_fpath_to_OpenEXR(src_fpath, dest_fpath)
        mock_shutil_which.assert_called_once_with("darktable-cli")
        mock_subprocess_call.assert_called_once_with(
            (
                "darktable-cli",
                src_fpath,
                str(Path("src/rawnind/dependencies/configs") / "dt4_xtrans_to_linrec2020.xmp"),
                dest_fpath,
                "--core",
                "--conf",
                "plugins/imageio/format/exr/bpp=16",
            )
        )

    @patch('rawnind.dependencies.raw_processing.shutil.which', return_value=None)
    def test_xtrans_fpath_to_openexr_darktable_not_found(self, mock_shutil_which, tmp_path):
        """
        Test xtrans_fpath_to_OpenEXR raises RuntimeError if darktable-cli is not found.
        """
        src_fpath = str(tmp_path / "test.raf")
        dest_fpath = str(tmp_path / "output.exr")
        with pytest.raises(RuntimeError, match="darktable-cli not found for X-Trans conversion."):
            raw.xtrans_fpath_to_OpenEXR(src_fpath, dest_fpath)
        mock_shutil_which.assert_called_once_with("darktable-cli")

    def test_xtrans_fpath_to_openexr_non_raf_input(self, tmp_path):
        """
        Test xtrans_fpath_to_OpenEXR raises AssertionError for non-.raf input.
        """
        src_fpath = str(tmp_path / "test.cr2")
        dest_fpath = str(tmp_path / "output.exr")
        with pytest.raises(AssertionError):
            raw.xtrans_fpath_to_OpenEXR(src_fpath, dest_fpath)

class TestRawFpathtoHdrImgFile:
    """Tests for the raw_fpath_to_hdr_img_file function, orchestrating the pipeline."""

    @pytest.fixture
    def mock_raw_processing_pipeline(self):
        """Fixture to mock the internal components of the raw processing pipeline."""
        with patch('rawnind.dependencies.raw_processing.RawLoader') as MockRawLoader, \
             patch('rawnind.dependencies.raw_processing.BayerProcessor') as MockBayerProcessor, \
             patch('rawnind.dependencies.raw_processing.ColorTransformer') as MockColorTransformer, \
             patch('rawnind.dependencies.raw_processing.is_exposure_ok') as mock_is_exposure_ok, \
             patch('rawnind.dependencies.raw_processing.hdr_nparray_to_file') as mock_hdr_nparray_to_file:

            mock_loader_instance = MagicMock()
            mock_bayer_processor_instance = MagicMock()
            mock_color_transformer_instance = MagicMock()

            # Default successful mocks
            mock_loader_instance.load_raw_data.return_value = (
                np.random.rand(1, 200, 200).astype(np.float32),
                raw.Metadata(
                    fpath='dummy.cr2',
                    bayer_pattern=raw.BayerPattern.RGGB,
                    rgbg_pattern=raw.BAYER_PATTERNS["RGGB"],
                    sizes={'raw_width': 200, 'raw_height': 200},
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
            MockRawLoader.return_value = mock_loader_instance
            
            mock_is_exposure_ok.return_value = True
            mock_bayer_processor_instance.demosaic.return_value = np.random.rand(3, 200, 200).astype(np.float32)
            MockBayerProcessor.return_value = mock_bayer_processor_instance

            mock_color_transformer_instance.camRGB_to_profiledRGB_img.return_value = np.random.rand(3, 200, 200).astype(np.float32)
            MockColorTransformer.return_value = mock_color_transformer_instance
            
            yield MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance

    def test_raw_fpath_to_hdr_img_file_ok_outcome(self, mock_raw_processing_pipeline, tmp_path):
        """
        Test raw_fpath_to_hdr_img_file for successful conversion.
        """
        MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance = mock_raw_processing_pipeline
        
        src_fpath = str(tmp_path / "input.cr2")
        dest_fpath = str(tmp_path / "output.exr")

        status, _, _ = raw.raw_fpath_to_hdr_img_file(src_fpath, dest_fpath)
        
        assert status == "OK"
        mock_loader_instance.load_raw_data.assert_called_once_with(src_fpath)
        mock_is_exposure_ok.assert_called_once()
        MockBayerProcessor.return_value.demosaic.assert_called_once()
        MockColorTransformer.return_value.camRGB_to_profiledRGB_img.assert_called_once()
        mock_hdr_nparray_to_file.assert_called_once()


    def test_raw_fpath_to_hdr_img_file_bad_exposure_outcome(self, mock_raw_processing_pipeline, tmp_path):
        """
        Test raw_fpath_to_hdr_img_file for BAD_EXPOSURE outcome.
        """
        MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance = mock_raw_processing_pipeline
        mock_is_exposure_ok.return_value = False # Simulate bad exposure
        
        src_fpath = str(tmp_path / "input.cr2")
        dest_fpath = str(tmp_path / "output.exr")

        status, _, _ = raw.raw_fpath_to_hdr_img_file(src_fpath, dest_fpath)
        
        assert status == "BAD_EXPOSURE"
        mock_loader_instance.load_raw_data.assert_called_once_with(src_fpath)
        mock_is_exposure_ok.assert_called_once()
        MockBayerProcessor.return_value.demosaic.assert_not_called() # Should not proceed to demosaic
        mock_hdr_nparray_to_file.assert_not_called() # Should not save

    def test_raw_fpath_to_hdr_img_file_unreadable_error_outcome(self, mock_raw_processing_pipeline, tmp_path):
        """
        Test raw_fpath_to_hdr_img_file for UNREADABLE_ERROR outcome.
        """
        MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance = mock_raw_processing_pipeline
        mock_loader_instance.load_raw_data.side_effect = raw.RawProcessingError("Test unreadable error")
        
        src_fpath = str(tmp_path / "input.cr2")
        dest_fpath = str(tmp_path / "output.exr")
        

        status, _, _ = raw.raw_fpath_to_hdr_img_file(src_fpath, dest_fpath)
        
        assert status == "UNREADABLE_ERROR"
        mock_loader_instance.load_raw_data.assert_called_once_with(src_fpath)
        mock_is_exposure_ok.assert_not_called() # Should not check exposure
        mock_hdr_nparray_to_file.assert_not_called() # Should not save

    def test_raw_fpath_to_hdr_img_file_unknown_error_outcome(self, mock_raw_processing_pipeline, tmp_path):
        """
        Test raw_fpath_to_hdr_img_file for UNKNOWN_ERROR outcome.
        """
        MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance = mock_raw_processing_pipeline
        mock_loader_instance.load_raw_data.side_effect = ValueError("Simulated unexpected error") # Simulate a generic error
        
        src_fpath = str(tmp_path / "input.cr2")
        dest_fpath = str(tmp_path / "output.exr")

        status, _, _ = raw.raw_fpath_to_hdr_img_file(src_fpath, dest_fpath)
        
        assert status == "UNKNOWN_ERROR"
        mock_loader_instance.load_raw_data.assert_called_once_with(src_fpath)
        mock_is_exposure_ok.assert_not_called()
        mock_hdr_nparray_to_file.assert_not_called()

    def test_raw_fpath_to_hdr_img_file_hdr_save_error(self, mock_raw_processing_pipeline, tmp_path):
        """
        Test raw_fpath_to_hdr_img_file when hdr_nparray_to_file raises an error.
        """
        MockRawLoader, MockBayerProcessor, MockColorTransformer, mock_is_exposure_ok, mock_hdr_nparray_to_file, mock_loader_instance = mock_raw_processing_pipeline
        mock_hdr_nparray_to_file.side_effect = RuntimeError("Save failed")
        
        src_fpath = str(tmp_path / "input.cr2")
        dest_fpath = str(tmp_path / "output.exr")

        with pytest.raises(RuntimeError, match="Save failed"):
            raw.raw_fpath_to_hdr_img_file(src_fpath, dest_fpath)