"""Example test demonstrating proper PyTorch integration for RawNIND tests.

This file shows how to properly initialize models, handle devices,
and perform assertions using PyTorch's testing utilities within
the existing RawNIND test framework.
"""

import torch
from .models.compression_autoencoders import (
    AbstractRawImageCompressor,
    BalleEncoder,
    BalleDecoder
)

class TestPyTorchIntegration:
    """Test class demonstrating proper PyTorch integration patterns."""

    def __init__(self):
        """Initialize with a simple model for testing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create encoder and decoder classes
        self.model = AbstractRawImageCompressor(
            device=device,
            in_channels=4,
            hidden_out_channels=192,
            bitstream_out_channels=320,
            encoder_cls=lambda **kwargs: BalleEncoder(**kwargs),
            decoder_cls=lambda **kwargs: BalleDecoder(**kwargs)
        ).to(device)

    def test_model_device_placement(self):
        """Test that models are properly placed on the correct device."""
        # Check that the model is a proper PyTorch module
        assert isinstance(self.model, torch.nn.Module)

        # Test device placement
        expected_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        actual_device = next(self.model.parameters()).device

        print(f"Model parameters are on device: {actual_device}")
        print(f"Expected device: {expected_device}")

        # Compare device types instead of exact objects
        assert str(actual_device) == str(expected_device)

    def test_model_forward_pass(self):
        """Test that models can process input tensors correctly."""
        # Create a dummy input tensor (batch_size=1, channels=4, height=64, width=64)
        dummy_input = torch.randn(1, 4, 64, 64)

        # Forward pass
        with torch.no_grad():
            output = self.model(dummy_input)

        # Check output shape and type
        assert isinstance(output, dict)  # AbstractRawImageCompressor returns a dictionary
        assert "reconstructed_image" in output
        assert output["reconstructed_image"].shape == (1, 4, 64, 64)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output['reconstructed_image'].shape}")

    def test_batch_handling(self):
        """Test proper batch handling for single images."""
        # Create a single image tensor (no batch dimension)
        single_image = torch.randn(4, 64, 64)

        # Test that the model can handle single images
        with torch.no_grad():
            output = self.model(single_image.unsqueeze(0))  # Add batch dimension manually

        # Output should be a batched tensor
        assert len(output["reconstructed_image"].shape) == 4  # (batch_size=1, channels, height, width)
        assert output["reconstructed_image"].shape[0] == 1    # Batch size should be 1
        print(f"Single image test - Output batch shape: {output['reconstructed_image'].shape}")

if __name__ == "__main__":
    test_instance = TestPyTorchIntegration()
    test_instance.test_model_device_placement()
    test_instance.test_model_forward_pass()
    test_instance.test_batch_handling()
    print("All tests passed!")