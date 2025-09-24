import pytest
import torch
from unittest.mock import MagicMock, patch

from rawnind.inference.clean_api import InferenceConfig, CleanCompressor

@pytest.fixture
def mock_compression_model():
    """Mock a compression model that returns a dict with reconstructed_image and bpp."""
    model = MagicMock(spec=torch.nn.Module)
    model.eval.return_value = model
    model.to.return_value = model
    # Simulate forward pass for compress_and_denoise
    model.return_value = {
        "reconstructed_image": torch.randn(1, 3, 128, 128),
        "bpp": torch.tensor(0.5, dtype=torch.float32)
    }
    return model

@pytest.fixture
def compressor_inference_config():
    """Base InferenceConfig for compressor testing."""
    return InferenceConfig(
        architecture="ManyPriors",
        input_channels=3,
        device="cpu",
    )

class TestCleanCompressor:
    """Unit tests for CleanCompressor class."""

    def test_init(self, mock_compression_model, compressor_inference_config):
        """Test CleanCompressor initialization."""
        compressor = CleanCompressor(model=mock_compression_model, config=compressor_inference_config)
        assert compressor.model == mock_compression_model
        assert compressor.config == compressor_inference_config
        assert compressor.device == torch.device("cpu")
        mock_compression_model.eval.assert_called_once()
        mock_compression_model.to.assert_called_once_with(torch.device("cpu"))

    @pytest.mark.parametrize("input_shape, expected_output_shape", [
        ((3, 128, 128), (3, 128, 128)), # Single image
        ((2, 3, 128, 128), (2, 3, 128, 128)), # Batch image
    ])
    def test_compress_and_denoise(self, mock_compression_model, compressor_inference_config, input_shape, expected_output_shape):
        """Test compress_and_denoise method."""
        compressor = CleanCompressor(model=mock_compression_model, config=compressor_inference_config)
        
        dummy_input = torch.randn(input_shape)
        
        # Ensure mock model returns correct shape for reconstructed_image and bpp
        if len(input_shape) == 3: # Handle single image input -> single bpp
            mock_compression_model.return_value = {
                "reconstructed_image": torch.randn(1, *expected_output_shape),
                "bpp": torch.tensor(0.5, dtype=torch.float32)
            }
        else: # Handle batch image input -> batch bpp
            mock_compression_model.return_value = {
                "reconstructed_image": torch.randn(*expected_output_shape),
                "bpp": torch.randn(input_shape[0], dtype=torch.float32) * 0.1 + 0.3 # Simulate batch BPP
            }


        results = compressor.compress_and_denoise(dummy_input)
        
        assert isinstance(results, dict)
        assert "denoised_image" in results
        assert "bpp" in results
        assert "compression_ratio" in results
        
        assert isinstance(results["denoised_image"], torch.Tensor)
        assert results["denoised_image"].shape == expected_output_shape
        assert isinstance(results["bpp"], float)
        assert results["bpp"] >= 0
        assert results["compression_ratio"] >= 0 or results["compression_ratio"] == float('inf')

        mock_compression_model.assert_called_once_with(dummy_input.unsqueeze(0) if len(input_shape) == 3 else dummy_input)

    def test_decompress(self, mock_compression_model, compressor_inference_config):
        """Test decompress method (simplified for this model type)."""
        compressor = CleanCompressor(model=mock_compression_model, config=compressor_inference_config)
        
        dummy_compressed_data = torch.randn(3, 128, 128) # Simulate compressed data
        mock_compression_model.return_value = {"reconstructed_image": torch.randn(1, 3, 128, 128)} # Only image for decompress
        
        decompressed_image = compressor.decompress(dummy_compressed_data)
        
        assert isinstance(decompressed_image, torch.Tensor)
        assert decompressed_image.shape == (3, 128, 128)
        mock_compression_model.assert_called_once_with(dummy_compressed_data.unsqueeze(0))

    def test_compress_alias(self, mock_compression_model, compressor_inference_config):
        """Test 'compress' alias calls 'compress_and_denoise'."""
        compressor = CleanCompressor(model=mock_compression_model, config=compressor_inference_config)
        
        with patch.object(compressor, 'compress_and_denoise', wraps=compressor.compress_and_denoise) as mock_compress_and_denoise:
            dummy_input = torch.randn(3, 128, 128)
            compressor.compress(dummy_input)
            mock_compress_and_denoise.assert_called_once_with(dummy_input, None) # Ensure it's called with the same args