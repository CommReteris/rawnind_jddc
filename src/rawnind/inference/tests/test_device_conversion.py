import pytest
import torch
from rawnind.inference.clean_api import convert_device_format

class TestConvertDeviceFormat:
    """Unit tests for the convert_device_format utility function."""

    @pytest.mark.parametrize("input_device, expected_output", [
        ("cpu", -1),
        ("cuda", 0),
        ("cuda:0", 0),
        ("cuda:1", 1),
        (torch.device("cpu"), -1),
        (torch.device("cuda:0"), 0),
        (torch.device("cuda:1"), 1),
        (-1, -1),
        (0, 0),
        (1, 1),
        ("custom_device", "custom_device"), # Pass-through for unknown strings
    ])
    def test_valid_device_conversions(self, input_device, expected_output):
        """Test valid device conversions."""
        assert convert_device_format(input_device) == expected_output

    def test_unsupported_device_type_raises_error(self):
        """Test that unsupported device types raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported device specification"):
            convert_device_format(None) # None is not a supported type
        
        with pytest.raises(ValueError, match="Unsupported device specification"):
            convert_device_format([]) # List is not a supported type
            
        with pytest.raises(ValueError, match="Unsupported device specification"):
            convert_device_format({}) # Dict is not a supported type