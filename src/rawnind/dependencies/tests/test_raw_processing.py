<<<<<<< HEAD
import pytest
import torch

from rawnind.dependencies import raw_processing as raw
from rawnind.dependencies import raw_processing as rawproc


class TestRawproc:
    def test_scenelin_to_pq(self):
        """Test that scenelin_to_pq pytorch matches numpy colour_science library."""
        ptbatch = torch.rand(5, 3, 256, 256) * 1.1 - 0.1
        npbatch = ptbatch.numpy()
        nptransformed = rawproc.scenelin_to_pq(npbatch)
        pttransformed = rawproc.scenelin_to_pq(ptbatch)
        assert torch.allclose(
            pttransformed,
            torch.from_numpy(nptransformed).to(torch.float32),
            atol=1e-05,
        )

    def test_match_gain(self):
        """Check that the gain is matched for 0 and 1 and that the dimensions remain the same for single and batched images."""
        anchor_single_0 = torch.zeros(3, 256, 256, dtype=torch.float32)
        anchor_single_1 = torch.ones_like(anchor_single_0)
        anchor_batch_0 = torch.zeros(5, 3, 256, 256, dtype=torch.float32)
        anchor_batch_1 = torch.ones_like(anchor_batch_0)
        rand_single = torch.rand_like(anchor_single_0)
        rand_batch = torch.rand_like(anchor_batch_0)
        matched_single_0 = rawproc.match_gain(
            anchor_img=anchor_single_0, other_img=rand_single
        )
        matched_single_1 = rawproc.match_gain(
            anchor_img=anchor_single_1, other_img=rand_single
        )
        matched_batch_0 = rawproc.match_gain(
            anchor_img=anchor_batch_0, other_img=rand_batch
        )
        matched_batch_1 = rawproc.match_gain(
            anchor_img=anchor_batch_1, other_img=rand_batch
        )
        assert abs(matched_single_0.mean().item() - 0.0) < 1e-5
        assert abs(matched_single_1.mean().item() - 1.0) < 1e-5
        assert abs(matched_batch_0.mean().item() - 0.0) < 1e-5
        assert abs(matched_batch_1.mean().item() - 1.0) < 1e-5
        assert matched_single_0.shape == (3, 256, 256)
        assert matched_batch_1.shape == (5, 3, 256, 256)

    def test_camRGB_to_rec2020_batch_conversion(self):
        """Ensure the output of rawproc.camRGB_to_lin_rec2020_images (pytorch batched) is the same as that of raw.camRGB_to_profiledRGB_img (numpy single)."""
        rgb_xyz_matrix_1 = torch.tensor(
            [
                [0.9943, -0.3269, -0.0839],
                [-0.5323, 1.3269, 0.2259],
                [-0.1198, 0.2083, 0.7557],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_2 = torch.tensor(
            [
                [0.6988, -0.1384, -0.0714],
                [-0.5631, 1.3410, 0.2447],
                [-0.1485, 0.2204, 0.7318],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_3 = torch.tensor(
            [
                [0.7888, -0.1902, -0.1011],
                [-0.8106, 1.6085, 0.2099],
                [-0.2353, 0.2866, 0.7330],
                [0.0000, 0.0000, 0.0000],
            ]
        )
        rgb_xyz_matrix_4 = rgb_xyz_matrix_2
        rgb_xyz_matrices = torch.stack(
            (rgb_xyz_matrix_1, rgb_xyz_matrix_2, rgb_xyz_matrix_3, rgb_xyz_matrix_4)
        )
        img1_camRGB = torch.rand(3, 256, 256)
        img2_camRGB = torch.rand(3, 256, 256)
        img3_camRGB = torch.rand(3, 256, 256)
        img4_camRGB = torch.rand(3, 256, 256)
        images_camRGB = torch.stack(
            (img1_camRGB, img2_camRGB, img3_camRGB, img4_camRGB)
        )
        img1_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img1_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_1.numpy()},
                "lin_rec2020",
            )
        ).float()
        img2_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img2_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_2.numpy()},
                "lin_rec2020",
            )
        ).float()
        img3_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img3_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_3.numpy()},
                "lin_rec2020",
            )
        ).float()
        img4_rec2020 = torch.from_numpy(
            raw.camRGB_to_profiledRGB_img(
                img4_camRGB.numpy(),
                {"rgb_xyz_matrix": rgb_xyz_matrix_4.numpy()},
                "lin_rec2020",
            )
        ).float()
        images_rec2020 = rawproc.camRGB_to_lin_rec2020_images(
            images_camRGB, rgb_xyz_matrices
        )
        individual_outputs = torch.stack(
            (img1_rec2020, img2_rec2020, img3_rec2020, img4_rec2020)
        )

        assert torch.allclose(individual_outputs, images_rec2020, atol=1e-06)
=======
import numpy as np
import pytest
import cv2
from unittest.mock import patch, MagicMock
from rawnind.dependencies import raw_processing as raw

def test_raw_loader_load_raw_data(mock_rawpy_data):
    """
    Test RawLoader.load_raw_data with mocked rawpy.

    Objective: Verify RawLoader processes RAW files correctly.
    Test criteria: Returns expected image array and metadata structure.
    How testing fulfills purpose: Ensures RAW loading pipeline works with external libraries.
    Components mocked: rawpy using a fixture to control its behavior.
    Reason for hermeticity: RAW file processing requires external files/libraries.
    """
    mock_raw = mock_rawpy_data

    # Set specific raw_image_visible for this test
    mock_raw.raw_image = np.random.randint(0, 65535, (1024, 1024), dtype=np.uint16)
    # Ensure raw_colors and rgb_xyz_matrix are set explicitly or reset if fixture changes them
    mock_raw.raw_colors = np.array([[0, 1], [2, 3]])  # RGGB pattern
    mock_raw.rgb_xyz_matrix = np.eye(3).astype(np.float32)
    mock_raw.raw_pattern = np.array([[0, 1], [2, 3]])
        
        

    loader = raw.RawLoader(raw.ProcessingConfig())
    image, metadata = loader.load_raw_data('dummy.cr2')

    assert isinstance(image, np.ndarray)
    assert image.shape == (1, 512, 512)
    assert isinstance(metadata, raw.Metadata)
    assert metadata.bayer_pattern == raw.BayerPattern.RGGB
    assert metadata.rgb_xyz_matrix is not None

class TestRawLoaderInternalMethods:
    """Tests for internal helper methods of RawLoader."""
    
    @pytest.fixture
    def mock_rawpy_for_internal(self):
        """Fixture to mock rawpy.imread and rawpy.RawPy for internal RawLoader tests."""
        with patch('rawnind.dependencies.raw_processing.rawpy.imread') as mock_imread, \
             patch('rawnind.dependencies.raw_processing.rawpy.RawPy') as mock_rawpy_class:
            mock_raw = MagicMock()
            mock_raw.raw_image = np.zeros((10, 10), dtype=np.uint16) # Smaller image for simplicity
            mock_raw.raw_colors = np.array([[0, 1], [2, 3]]) # RGGB pattern for consistent mocking
            mock_sizes_dict = {'raw_width': 10, 'raw_height': 10, 'width': 10, 'height': 10, 'iwidth': 10, 'iheight': 10, 'top_margin': 0, 'left_margin': 0}
            mock_raw.sizes = MagicMock()
            mock_raw.sizes._asdict.return_value = mock_sizes_dict
            mock_raw.color_desc = b'RGBG'
            mock_raw.rgb_xyz_matrix = np.eye(3)
            # Add other necessary rawpy attributes that _RawLoader might access
            mock_raw.camera_whitebalance = np.array([1.0, 1.0, 1.0, 1.0])
            mock_raw.black_level_per_channel = np.array([0, 0, 0, 0])
            mock_raw.white_level = 16000
            mock_raw.camera_white_level_per_channel = np.array([16000, 16000, 16000, 16000])
            mock_raw.daylight_whitebalance = np.array([1.0, 1.0, 1.0, 1.0])

            mock_imread.return_value = mock_raw
            mock_rawpy_class.return_value = mock_raw # In case RawPy is instantiated
            yield mock_imread, mock_rawpy_class, mock_raw

    def test_remove_empty_borders(self, mock_rawpy_for_internal):
        """
        Test RawLoader._remove_empty_borders removes specified borders.

        Objective: Verify _remove_empty_borders correctly crops image and updates metadata based on margins.
        Test criteria: Image and relevant metadata dimensions are reduced by exact margin values.
        How testing fulfills purpose: Ensures initial raw cropping logic matches domain expertise.
        Components mocked: rawpy (via fixture to avoid file I/O).
        Reason for hermeticity: Isolates internal method for precise testing.
        """
        # Unpack the mock objects from the fixture
        mock_imread, mock_rawpy_class, mock_raw = mock_rawpy_for_internal

        loader = raw.RawLoader(raw.ProcessingConfig(crop_all=False)) # Set crop_all=False to only test border removal

        # Setup initial image and metadata with borders
        initial_mono_img = np.random.rand(1, 10, 10).astype(np.float32)
        metadata_dict = {
            "sizes": {'raw_width': 10, 'raw_height': 10, 'width': 8, 'height': 8, 'iwidth': 8, 'iheight': 8, 'top_margin': 1, 'left_margin': 1},
        }
        
        # Call the internal method
        processed_mono_img = loader._remove_empty_borders(initial_mono_img, metadata_dict)

        # Assertions
        assert processed_mono_img.shape == (1, 9, 9) # Image should be cropped by 1 from top/left
        assert metadata_dict["sizes"]["top_margin"] == 0
        assert metadata_dict["sizes"]["left_margin"] == 0
        assert metadata_dict["sizes"]["raw_height"] == 9
        assert metadata_dict["sizes"]["raw_width"] == 9
        # Ensure that the content is correctly shifted
        assert np.all(processed_mono_img[:, :, :] == initial_mono_img[:, 1:, 1:])

    def test_remove_empty_borders_with_crop_all(self, mock_rawpy_for_internal):
        """
        Test RawLoader._remove_empty_borders with crop_all=True.

        Objective: Verify _remove_empty_borders crops to minimum of all stated sizes.
        Test criteria: Final image and metadata dimensions match the minimum of given sizes.
        How testing fulfills purpose: Ensures that the final image dimensions are consistent with metadata after cropping.
        Components mocked: rawpy (via fixture to avoid file I/O).
        Reason for hermeticity: Isolates internal method for precise testing.
        """
        # Unpack the mock objects from the fixture
        mock_imread, mock_rawpy_class, mock_raw = mock_rawpy_for_internal

        loader = raw.RawLoader(raw.ProcessingConfig(crop_all=True))

        initial_mono_img = np.random.rand(1, 10, 10).astype(np.float32)
        metadata_dict = {
            "sizes": {'raw_width': 10, 'raw_height': 10, 'width': 8, 'height': 8, 'iwidth': 8, 'iheight': 8, 'top_margin': 0, 'left_margin': 0},
        }

        processed_mono_img = loader._remove_empty_borders(initial_mono_img, metadata_dict)

        assert processed_mono_img.shape == (1, 8, 8)
        assert metadata_dict["sizes"]["raw_height"] == 8
        assert metadata_dict["sizes"]["raw_width"] == 8


def test_bayer_processor_initialization():
    """
    Test BayerProcessor initialization with default config.

    Objective: Ensure BayerProcessor can be instantiated with ProcessingConfig.
    Test criteria: Instance is created without errors, has expected attributes.
    How testing fulfills purpose: Verifies basic setup of Bayer processing pipeline.
    Components mocked: None - pure instantiation test.
    Reason for hermeticity: No external dependencies in initialization.
    """
    config = raw.ProcessingConfig()
    processor = raw.BayerProcessor(config)
    assert processor is not None
    assert hasattr(processor, 'config')

def test_bayer_processor_apply_white_balance_numerical():
    """
    Test BayerProcessor.apply_white_balance applies and reverses white balance numerically.

    Objective: Verify accurate numerical application and reversal of white balance.
    Test criteria: Specific pixel values match expected values after WB and reverse WB.
    How testing fulfills purpose: Ensures WB logic is numerically correct and reversible.
    Components mocked: None - direct numpy operations.
    Reason for hermeticity: Uses controlled numerical inputs for precise validation.
    """
    config = raw.ProcessingConfig(wb_type="daylight")
    processor = raw.BayerProcessor(config)

    # Simple 2x2 mono Bayer image (representing RGGB)
    # R val, G1 val, G2 val, B val at (0,0), (0,1), (1,0), (1,1) if flattened
    initial_bayer = np.array([
        [[0.1, 0.2],
         [0.3, 0.4]]
    ], dtype=np.float32)

    # Metadata with specific daylight_whitebalance_norm, normalized by G1 (index 1)
    # The RGGB mapping in apply_white_balance: R=0, G1=1, G2=3, B=2
    # So wb_norm[0] for R, wb_norm[1] for G1, wb_norm[3] for G2, wb_norm[2] for B
    wb_norm_values = np.array([1.5, 1.0, 0.8, 1.2], dtype=np.float32) # R, G, B, G (legacy order)

    metadata = raw.Metadata(
        fpath='dummy.cr2',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [3, 2]]), # Example pattern matching order in wb_norm
        sizes={'raw_width': 2, 'raw_height': 2},
        camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        black_level_per_channel=np.array([0, 0, 0, 0]),
        white_level=1,
        camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance=np.array([1.5, 1.0, 1.2, 1.0]), # Raw value (R, G, B, G)
        rgb_xyz_matrix=np.eye(3),
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance_norm=wb_norm_values # Normalized values for apply_white_balance
    )

    # Expected values after applying white balance (initial_bayer * wb_norm_values_mapped)
    expected_r = initial_bayer[0, 0, 0] * wb_norm_values[0] # R
    expected_g1 = initial_bayer[0, 0, 1] * wb_norm_values[1] # G1
    expected_g2 = initial_bayer[0, 1, 0] * wb_norm_values[3] # G2
    expected_b = initial_bayer[0, 1, 1] * wb_norm_values[2] # B

    # Apply forward white balance
    wb_applied = processor.apply_white_balance(initial_bayer.copy(), metadata, reverse=False)

    assert np.isclose(wb_applied[0, 0, 0], expected_r)
    assert np.isclose(wb_applied[0, 0, 1], expected_g1)
    assert np.isclose(wb_applied[0, 1, 0], expected_g2)
    assert np.isclose(wb_applied[0, 1, 1], expected_b)

    # Apply reverse white balance
    wb_reversed = processor.apply_white_balance(wb_applied.copy(), metadata, reverse=True)

    # Should be back to original values approximately
    assert np.allclose(wb_reversed, initial_bayer) # Should be back to original values approximately

@patch('rawnind.dependencies.raw_processing.cv2.demosaicing')
def test_demosaic_function(mock_cv2_demosaic):
    """
    Test top-level demosaic function.

    Objective: Verify demosaic wrapper works with different methods.
    Test criteria: Calls appropriate backend, returns RGB image.
    How testing fulfills purpose: Ensures demosaic interface is functional.
    Components mocked: cv2.demosaicing to avoid OpenCV dependency in tests.
    Reason for hermeticity: OpenCV operations require compiled libraries.
    """
    mock_cv2_demosaic.return_value = np.random.rand(256, 256, 3).astype(np.float32)

    bayer_mosaic = np.random.rand(1, 256, 256).astype(np.float32)
    metadata = raw.Metadata(
        fpath='dummy.cr2',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [1, 2]]),
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
    
    config = raw.ProcessingConfig()
    processor = raw.BayerProcessor(config)
    result = processor.demosaic(bayer_mosaic, metadata)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3  # RGB channels (CHW format)
    mock_cv2_demosaic.assert_called_once()

def test_demosaic_integration_real_cv2():
    """
    Test demosaic function with real OpenCV (integration test).

    Objective: Verify demosaic produces valid RGB output from Bayer mosaic.
    Test criteria: Returns RGB image with correct shape, finite values, proper dynamic range.
    How testing for this criteria fulfills purpose: Ensures demosaic algorithm works correctly with real cv2.
    Components mocked: None - uses real cv2.demosaicing.
    Reason for hermeticity: Uses synthetic Bayer data, no external files.
    """
    # Create synthetic RGGB Bayer mosaic (4x4 -> 2x2 RGB after demosaic)
    # RGGB pattern: R G / G B
    bayer_mosaic = np.array([
        [0.8, 0.6, 0.7, 0.5],  # R G R G
        [0.4, 0.9, 0.3, 0.8],  # G B G B
        [0.6, 0.5, 0.8, 0.4],  # R G R G
        [0.7, 0.8, 0.6, 0.9],  # G B G B
    ], dtype=np.float32).reshape(1, 4, 4)

    metadata = raw.Metadata(
        fpath='synthetic.cr2',
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
    
    config = raw.ProcessingConfig()
    processor = raw.BayerProcessor(config)
    # Test with default EA method
    rgb_result = processor.demosaic(bayer_mosaic, metadata)

    assert isinstance(rgb_result, np.ndarray)
    assert rgb_result.shape == (3, 4, 4)  # RGB, same spatial dims
    assert rgb_result.dtype == np.float32
    assert np.all(np.isfinite(rgb_result))
    assert 0 <= rgb_result.min() <= rgb_result.max() <= 1

    # Test with basic method
    config.demosaic_method=cv2.COLOR_BayerRGGB2RGB
    processor = raw.BayerProcessor(config)
    rgb_result_basic = processor.demosaic(bayer_mosaic, metadata)

    assert rgb_result_basic.shape == (3, 4, 4)
    assert rgb_result_basic.dtype == np.float32
    assert np.all(np.isfinite(rgb_result_basic))
    assert 0 <= rgb_result_basic.min() <= rgb_result_basic.max() <= 1
    # Different methods produce different results, just verify both work


def test_bayer_pattern_transformations():
    """
    Test mono_to_rggb and rggb_to_mono transformations.

    Objective: Verify Bayer pattern transformations preserve data correctly.
    Test criteria: Round-trip conversion maintains original data, correct channel ordering.
    How testing for this criteria fulfills purpose: Ensures Bayer pattern logic matches domain knowledge.
    Components mocked: None - pure numpy operations.
    Reason for hermeticity: No external dependencies, deterministic operations.
    """
    # Create test Bayer mosaic (RGGB pattern)
    mono_bayer = np.array([
        [0.8, 0.6, 0.7, 0.5],  # R G R G
        [0.4, 0.9, 0.3, 0.8],  # G B G B
        [0.6, 0.5, 0.8, 0.4],  # R G R G
        [0.7, 0.8, 0.6, 0.9],  # G B G B
    ], dtype=np.float32).reshape(1, 4, 4)

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

    processor = raw.BayerProcessor(raw.ProcessingConfig())

    # Convert mono to RGGB
    rggb = processor.mono_to_rggb(mono_bayer, metadata)

    assert rggb.shape == (4, 2, 2)  # 4 channels, halved spatial dims
    assert rggb.dtype == np.float32

    # Verify channel values match expected RGGB pattern
    assert rggb[0, 0, 0] == 0.8  # R at (0,0)
    assert rggb[1, 0, 0] == 0.6  # G at (0,1) -> G1
    assert rggb[2, 0, 0] == 0.4  # G at (1,0) -> G2
    assert rggb[3, 0, 0] == 0.9  # B at (1,1)

    # Convert back to mono
    mono_reconstructed = processor.rggb_to_mono(rggb)

    assert mono_reconstructed.shape == mono_bayer.shape
    assert np.allclose(mono_reconstructed, mono_bayer)
class TestExposureCheck:
    """Tests for the is_exposure_ok function."""

    @pytest.fixture
    def mock_metadata_exposure_check(self):
        """Fixture for Metadata used in exposure checks."""
        return raw.Metadata(
            fpath='dummy.cr2',
            bayer_pattern=raw.BayerPattern.RGGB,
            rgbg_pattern=raw.BAYER_PATTERNS["RGGB"],
            sizes={'raw_width': 10, 'raw_height': 10},
            camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            black_level_per_channel=np.array([0, 0, 0, 0]),
            white_level=1, # This is crucial for scaling. Normalized images in [0,1].
            camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
            rgb_xyz_matrix=np.eye(3),
            overexposure_lb=1.0, # Assumes image is already scaled to 0-1 range based on white_level.
            camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
            daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
        )

    def test_is_exposure_ok_good_exposure(self, mock_metadata_exposure_check):
        """
        Test is_exposure_ok with an image that has good exposure.

        Objective: Verify is_exposure_ok returns True for well-exposed images.
        Test criteria: No pixels exceed over/underexposure thresholds significantly.
        How testing fulfills purpose: Ensures correct identification of usable images.
        Components mocked: None, uses controlled numpy data.
        Reason for hermeticity: Isolates logic for precise validation.
        """
        # Create a mono Bayer image with values in acceptable range (e.g., 0.2 to 0.8)
        mono_img = np.full((1, 10, 10), 0.5, dtype=np.float32) # Uniformly 'grey'
        assert raw.is_exposure_ok(mono_img, mock_metadata_exposure_check) == True

    def test_is_exposure_ok_overexposed(self, mock_metadata_exposure_check):
        """
        Test is_exposure_ok with an overexposed image.

        Objective: Verify is_exposure_ok returns False for images with too much overexposure.
        Test criteria: A significant portion of pixels exceed oe_threshold.
        How testing fulfills purpose: Ensures overexposed images are correctly flagged.
        Components mocked: None, uses controlled numpy data.
        Reason for hermeticity: Isolates logic for precise validation.
        """
        mono_img = np.zeros((1, 10, 10), dtype=np.float32)
        # Set 30% of pixels to be overexposed (oe_threshold=0.99, overexposure_lb=1.0 -> >0.99)
        mono_img[:, :3, :] = 0.995 # 30 pixels overexposed out of 100
        # Default qty_threshold is 0.75, so 30% overexposure should be flagged as bad
        assert raw.is_exposure_ok(mono_img, mock_metadata_exposure_check, oe_threshold=0.99, qty_threshold=0.25) == False

    def test_is_exposure_ok_underexposed(self, mock_metadata_exposure_check):
        """
        Test is_exposure_ok with an underexposed image.

        Objective: Verify is_exposure_ok returns False for images with too much underexposure.
        Test criteria: A significant portion of pixels are below ue_threshold.
        How testing fulfills purpose: Ensures underexposed images are correctly flagged.
        Components mocked: None, uses controlled numpy data.
        Reason for hermeticity: Isolates logic for precise validation.
        """
        mono_img = np.ones((1, 10, 10), dtype=np.float32)
        # Set 30% of pixels to be underexposed (ue_threshold=0.001 -> <0.001)
        mono_img[:, :3, :] = 0.0005 # 30 pixels underexposed out of 100
        assert raw.is_exposure_ok(mono_img, mock_metadata_exposure_check, ue_threshold=0.001, qty_threshold=0.25) == False

    def test_is_exposure_ok_mix_within_threshold(self, mock_metadata_exposure_check):
        """
        Test is_exposure_ok with mixed over/underexposure within quantity threshold.

        Objective: Verify is_exposure_ok returns True when total problematic pixels are acceptable.
        Test criteria: Combined over/underexposed pixels are below qty_threshold.
        How testing fulfills purpose: Ensures threshold logic is correctly applied for mixed conditions.
        Components mocked: None, uses controlled numpy data.
        Reason for hermeticity: Isolates logic for precise validation.
        """
        mono_img = np.full((1, 10, 10), 0.5, dtype=np.float32)
        # 10% overexposed, 10% underexposed. Total 20% problematic.
        # With default qty_threshold=0.75, this should pass.
        mono_img[:, :1, :] = 0.995  # Overexposed band
        mono_img[:, 1:2, :] = 0.0005 # Underexposed band
        assert raw.is_exposure_ok(mono_img, mock_metadata_exposure_check) == True

    def test_is_exposure_ok_mix_exceeding_threshold(self, mock_metadata_exposure_check):
        """
        Test is_exposure_ok with mixed over/underexposure exceeding quantity threshold.

        Objective: Verify is_exposure_ok returns False when total problematic pixels are too high.
        Test criteria: Combined over/underexposed pixels exceed qty_threshold.
        How testing fulfills purpose: Ensures threshold logic correctly flags problematic images.
        Components mocked: None, uses controlled numpy data.
        Reason for hermeticity: Isolates logic for precise validation.
        """
        mono_img = np.full((1, 10, 10), 0.5, dtype=np.float32)
        # 30% overexposed, 30% underexposed. Total 60% problematic.
        # With qty_threshold=0.5, this should fail.
        mono_img[:, :3, :] = 0.995  # Overexposed band
        mono_img[:, 3:6, :] = 0.0005 # Underexposed band
        assert raw.is_exposure_ok(mono_img, mock_metadata_exposure_check, qty_threshold=0.5) == False


def test_color_transformer_initialization():
    """
    Test ColorTransformer initialization.

    Objective: Ensure ColorTransformer can be instantiated.
    Test criteria: Instance is created without errors.
    How testing fulfills purpose: Verifies basic setup of color transformation utilities.
    Components mocked: None - pure instantiation test.
    Reason for hermeticity: No external dependencies in initialization.
    """
    transformer = raw.ColorTransformer()
    assert transformer is not None

def test_color_transformer_matrix_methods():
    """
    Test ColorTransformer matrix generation methods.

    Objective: Verify color space transformation matrices are computed.
    Test criteria: Returns valid 3x3 matrices for different profiles.
    How testing fulfills purpose: Ensures color management utilities work.
    Components mocked: None - pure mathematical operations.
    Reason for hermeticity: Matrix computations are deterministic.
    """
    transformer = raw.ColorTransformer()

    # Test XYZ to RGB matrix
    matrix = transformer.get_xyz_to_rgb_matrix('sRGB')
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)

    # Test camRGB to profiledRGB
    metadata = raw.Metadata(
        fpath='dummy.cr2',
        bayer_pattern=raw.BayerPattern.RGGB,
        rgbg_pattern=np.array([[0, 1], [1, 2]]),
        sizes={'raw_width': 256, 'raw_height': 256},
        camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        black_level_per_channel=np.array([0, 0, 0, 0]),
        white_level=1,
        camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        rgb_xyz_matrix=np.eye(3).astype(np.float32),
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
    )

    camrgb_matrix = transformer.get_camRGB_to_profiledRGB_img_matrix(metadata, 'sRGB')
    assert isinstance(camrgb_matrix, np.ndarray)
    assert camrgb_matrix.shape == (3, 3)

def test_color_transformer_camrgb_to_profiledrgb_img_numerical():
    """
    Test ColorTransformer.camRGB_to_profiledRGB_img for numerical accuracy with known inputs.

    Objective: Verify accurate numerical color transformation from camRGB to a profiled RGB.
    Test criteria: Specific pixel values match expected values after transformation.
    How testing fulfills purpose: Ensures color transformation logic is numerically correct.
    Components mocked: None - uses controlled numerical inputs for precise validation.
    Reason for hermeticity: Isolates numerical method for precise testing.
    """
    transformer = raw.ColorTransformer()

    # Create a simple 3-channel camRGB image
    camrgb_img = np.array([
        [[0.1, 0.2], [0.3, 0.4]],
        [[0.5, 0.6], [0.7, 0.8]],
        [[0.9, 0.0], [0.1, 0.2]]
    ], dtype=np.float32)

    # Simplified metadata with an identity rgb_xyz_matrix for easier calculation
    # and direct XYZ to lin_sRGB for validation
    metadata = raw.Metadata(
        fpath='dummy.cr2',
        bayer_pattern=raw.BayerPattern.RGGB, # Not directly used in this function, but required
        rgbg_pattern=np.array([[0, 1], [2, 3]]),
        sizes={'raw_width': 2, 'raw_height': 2},
        camera_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        black_level_per_channel=np.array([0, 0, 0, 0]),
        white_level=1,
        camera_white_level_per_channel=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance=np.array([1.0, 1.0, 1.0, 1.0]),
        rgb_xyz_matrix=np.eye(3).astype(np.float32), # Identity for simplicity
        overexposure_lb=1.0,
        camera_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0]),
        daylight_whitebalance_norm=np.array([1.0, 1.0, 1.0, 1.0])
    )

    # Manually compute expected output for 'lin_sRGB' with identity matrices
    xyz_to_srgb_matrix = np.array([
        [3.24100326, -1.53739899, -0.49861587],
        [-0.96922426, 1.87592999, 0.04155422],
        [0.05563942, -0.2040112, 1.05714897]
    ], dtype=np.float32)
    
    # Since rgb_xyz_matrix is identity, cam_to_xyzd65 will also be identity
    # So the full color_matrix will be xyz_to_srgb_matrix
    expected_profiled_img = (xyz_to_srgb_matrix @ camrgb_img.reshape(3, -1)).reshape(camrgb_img.shape)

    profiled_img = transformer.camRGB_to_profiledRGB_img(camrgb_img, metadata, 'lin_sRGB')

    assert np.allclose(profiled_img, expected_profiled_img, atol=1e-6)
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
