import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
from src.rawnind.dependencies import raw_processing as raw # Re-import raw_processing to ensure it's up-to-date
from src.rawnind.dependencies.raw_processing import ProcessingConfig # Import ProcessingConfig

@pytest.fixture
def mock_rawpy_data(monkeypatch):
    """
    Fixture to mock rawpy.imread, allowing tests to control the returned RawPy object.
    
    Yields a MagicMock representing the RawPy object, allowing individual tests to
    configure its attributes (e.g., raw_image, raw_colors, sizes) before calling
    functions that depend on rawpy.
    """
    mock_raw_instance = MagicMock()
    mock_raw_instance.raw_image = np.zeros((512, 512), dtype=np.uint16)
    mock_raw_instance.raw_colors = np.array([[0, 1], [2, 3]]) # Default to RGGB
    mock_sizes_dict = {'raw_width': 512, 'raw_height': 512, 'width': 512, 'height': 512, 'iwidth': 512, 'iheight': 512, 'top_margin': 0, 'left_margin': 0}
    mock_raw_instance.sizes = MagicMock()
    mock_raw_instance.sizes._asdict.return_value = mock_sizes_dict
    mock_raw_instance.color_desc = b'RGBG'
    mock_raw_instance.camera_whitebalance = np.array([1.0, 1.0, 1.0, 1.0])
    mock_raw_instance.black_level_per_channel = np.array([0, 0, 0, 0])
    mock_raw_instance.white_level = 65535
    mock_raw_instance.camera_white_level_per_channel = np.array([65535, 65535, 65535, 65535])
    mock_raw_instance.daylight_whitebalance = np.array([1.0, 1.0, 1.0, 1.0])
    mock_raw_instance.rgb_xyz_matrix = np.eye(3)

    with patch('rawnind.dependencies.raw_processing.rawpy.imread', return_value=mock_raw_instance) as mock_imread:
        yield mock_raw_instance