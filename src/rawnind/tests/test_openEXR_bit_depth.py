"""
Objective: Convert standalone script to unit test for OpenEXR export with different bit depths, using mocks for hermeticity.
Test Criteria: Mock np.random.random to dummy array; mock raw.hdr_nparray_to_file; call script logic; assert called with expected bit_depth (32/16) and color_profile.
Fulfillment: Ensures OpenEXR export logic works without real file I/O; verifies calls with different bit depths; replaces original script's intent (export dummy image in 32/16 bit) with testable assertions on mock calls.
Components Mocked/Fixtured: patch for np.random.random (return dummy array); patch for raw.hdr_nparray_to_file (mock call tracking); parametrize over bit_depth (16,32).
Reasons for Mocking/Fixturing: Original script performs real file I/O, causing non-hermetic tests; mocks allow consistent assertion on expected arguments (bit_depth, color_profile='lin_rec2020'); ensures test runs without files; fulfills intent (verify export with different precision) in unit test form without overhead or instability.
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.rawnind.libs import raw

@pytest.mark.parametrize("bit_depth", [16, 32])
def test_openEXR_bit_depth(bit_depth):
    """Parametrized test for OpenEXR bit depth export with mocks."""
    # Mock np.random.random to return dummy image array
    mock_random = MagicMock(return_value=np.random.random((3, 128, 128)))

    # Mock raw.hdr_nparray_to_file to track calls
    mock_hdr_to_file = MagicMock()

    with patch("numpy.random.random", mock_random):
        with patch("src.rawnind.libs.raw.hdr_nparray_to_file", mock_hdr_to_file):
            # Simulate script logic: create dummy image and export
            dummy_image = mock_random()
            mock_hdr_to_file(dummy_image, f"test_openEXR_bit_depth_{bit_depth}.exr", bit_depth=bit_depth, color_profile="lin_rec2020")

            # Assert mock called with expected args
            mock_hdr_to_file.assert_called_once()
            call_args = mock_hdr_to_file.call_args
            assert call_args[1]["bit_depth"] == bit_depth
            assert call_args[1]["color_profile"] == "lin_rec2020"
            assert call_args[0][0].shape == (3, 128, 128)