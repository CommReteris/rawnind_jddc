import pytest
import torch
from unittest.mock import MagicMock, patch

from rawnind.inference.clean_api import InferenceConfig, CleanDenoiser, CleanBayerDenoiser
from rawnind.dependencies import raw_processing as rawproc

@pytest.fixture
def mock_model():
    """Mock a generic PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.eval.return_value = model
    model.to.return_value = model
    # By default, mock its forward pass to return a dummy tensor
    model.return_value = torch.randn(1, 3, 128, 128)
    return model

@pytest.fixture
def base_inference_config():
    """Base InferenceConfig for denoiser testing."""
    return InferenceConfig(
        architecture="unet",
        input_channels=3,
        device="cpu",
    )

class TestCleanDenoiser:
    """Unit tests for CleanDenoiser class."""

    def test_init(self, mock_model, base_inference_config):
        """Test CleanDenoiser initialization."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        assert denoiser.model == mock_model
        assert denoiser.config == base_inference_config
        assert denoiser.device == torch.device("cpu")
        mock_model.eval.assert_called_once()
        mock_model.to.assert_called_once_with(torch.device("cpu"))

    @pytest.mark.parametrize("input_shape, expected_output_shape", [
        ((3, 128, 128), (3, 128, 128)),  # Single image
        ((2, 3, 128, 128), (2, 3, 128, 128)),  # Batch image
    ])
    def test_denoise(self, mock_model, base_inference_config, input_shape, expected_output_shape):
        """Test denoise method for single and batch inputs."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)

        dummy_input = torch.randn(input_shape)
        batched_shape = (1,) + expected_output_shape if len(input_shape) == 3 else expected_output_shape
        mock_model.return_value = torch.randn(batched_shape)  # Ensure mock model returns correct shape

        mock_model.reset_mock()

        output = denoiser.denoise(dummy_input)

        assert isinstance(output, torch.Tensor)
        assert output.shape == expected_output_shape
        assert mock_model.call_count == 1
        called_tensor = mock_model.call_args[0][0]
        if len(input_shape) == 3:
            assert torch.equal(called_tensor, dummy_input.unsqueeze(0))
        else:
            assert torch.equal(called_tensor, dummy_input)

    def test_denoise_batch(self, mock_model, base_inference_config):
        """Test denoise_batch method."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        dummy_batch = torch.randn(2, 3, 128, 128)
        mock_model.return_value = torch.randn(2, 3, 128, 128)

        output = denoiser.denoise_batch(dummy_batch)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (2, 3, 128, 128)
        mock_model.assert_called_once_with(dummy_batch)

    def test_denoise_input_channel_validation(self, mock_model, base_inference_config):
        """Test denoise with invalid input channels."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        invalid_input = torch.randn(4, 64, 64)  # Expected 3 channels, got 4
        with pytest.raises(ValueError, match=f"Expected {base_inference_config.input_channels} channels, got 4"):
            denoiser.denoise(invalid_input)

    def test_denoise_return_dict(self, mock_model, base_inference_config):
        """Test denoise method with return_dict=True."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        mock_model.return_value = {"reconstructed_image": torch.randn(1, 3, 128, 128), "some_other_info": "test"}

        output_dict = denoiser.denoise(torch.randn(3, 128, 128), return_dict=True)

        assert isinstance(output_dict, dict)
        assert "denoised_image" in output_dict
        assert "some_other_info" in output_dict
        assert output_dict["denoised_image"].shape == (3, 128, 128)

    @pytest.mark.skip(reason="TODO: CleanDenoiser should gracefully handle rank-2 inputs by lifting them to batched tensors")
    def test_denoise_low_rank_input(self, mock_model, base_inference_config):
        """Placeholder to ensure future support for 2D tensors."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        image = torch.randn(128, 128)
        denoiser.denoise(image)

    @pytest.mark.skip(reason="TODO: Detect models that only accept singleton batches and surface a clearer error")
    def test_denoise_batch_for_singleton_only_model(self, mock_model, base_inference_config):
        """Placeholder highlighting need for explicit messaging when model rejects multi-image batches."""
        denoiser = CleanDenoiser(model=mock_model, config=base_inference_config)
        batch = torch.randn(2, 3, 32, 32)
        denoiser.denoise(batch)


class TestCleanBayerDenoiser:
    """Unit tests for CleanBayerDenoiser class."""

    @pytest.fixture
    def bayer_inference_config(self):
        """Bayer-specific InferenceConfig for denoiser testing."""
        return InferenceConfig(
            architecture="utnet3",
            input_channels=4,
            device="cpu",
        )

    def test_init(self, mock_model, bayer_inference_config):
        """Test CleanBayerDenoiser initialization."""
        denoiser = CleanBayerDenoiser(model=mock_model, config=bayer_inference_config)
        assert denoiser.model == mock_model
        assert denoiser.config == bayer_inference_config
        assert denoiser.device == torch.device("cpu")
        assert denoiser.supports_bayer == True
        assert denoiser.demosaic_fn == rawproc.demosaic # Should be set from dependencies

    @pytest.mark.parametrize("input_shape, gt_shape, expected_output_shape", [
        ((4, 128, 128), (3, 256, 256), (3, 256, 256)),  # Single Bayer image
        ((2, 4, 128, 128), (2, 3, 256, 256), (2, 3, 256, 256)),  # Batch Bayer image
    ])
    @patch('rawnind.dependencies.raw_processing.camRGB_to_lin_rec2020_images')
    @patch('rawnind.dependencies.raw_processing.demosaic')  # patch demosaic if model does not output RGB
    def test_denoise_bayer(self, mock_demosaic, mock_camrgb, mock_model, bayer_inference_config, input_shape, gt_shape, expected_output_shape):
        """Test denoise_bayer method."""
        denoiser = CleanBayerDenoiser(model=mock_model, config=bayer_inference_config)

        dummy_bayer_input = torch.randn(input_shape)
        dummy_rgb_xyz_matrix = torch.eye(3) if len(input_shape) == 3 else torch.eye(3).unsqueeze(0)

        mock_model.return_value = torch.randn(expected_output_shape)
        mock_camrgb.return_value = torch.randn(expected_output_shape)

        mock_model.reset_mock()

        output = denoiser.denoise_bayer(dummy_bayer_input, dummy_rgb_xyz_matrix)

        assert isinstance(output, torch.Tensor)
        assert output.shape == expected_output_shape
        assert mock_model.call_count == 1
        called_tensor = mock_model.call_args[0][0]
        if len(input_shape) == 3:
            assert torch.equal(called_tensor, dummy_bayer_input.unsqueeze(0))
        else:
            assert torch.equal(called_tensor, dummy_bayer_input)
        mock_camrgb.assert_called_once() # camRGB_to_lin_rec2020_images should always be called

    def test_denoise_bayer_input_channel_validation(self, mock_model, bayer_inference_config):
        """Test denoise_bayer with invalid input channels."""
        denoiser = CleanBayerDenoiser(model=mock_model, config=bayer_inference_config)
        invalid_bayer_input = torch.randn(3, 128, 128) # Expected 4 channels, got 3
        dummy_rgb_xyz_matrix = torch.eye(3)
        with pytest.raises(ValueError, match="Bayer image must have 4 channels, got 3"):
            denoiser.denoise_bayer(invalid_bayer_input, dummy_rgb_xyz_matrix)

    def test_denoise_bayer_rgb_xyz_matrix_validation(self, mock_model, bayer_inference_config):
        """Test denoise_bayer with invalid rgb_xyz_matrix shape."""
        denoiser = CleanBayerDenoiser(model=mock_model, config=bayer_inference_config)
        dummy_bayer_input = torch.randn(4, 128, 128)
        invalid_matrix = torch.randn(3, 2) # Not 3x3
        with pytest.raises(ValueError, match="RGB XYZ matrix must be 3x3"):
            denoiser.denoise_bayer(dummy_bayer_input, invalid_matrix)

    def test_denoise_bayer_return_dict(self, mock_model, bayer_inference_config):
        """Test denoise_bayer method with return_dict=True."""
        denoiser = CleanBayerDenoiser(model=mock_model, config=bayer_inference_config)
        mock_model.return_value = {"reconstructed_image": torch.randn(1, 3, 256, 256), "bpp": torch.tensor(0.5)}

        with patch('rawnind.dependencies.raw_processing.camRGB_to_lin_rec2020_images', return_value=torch.randn(1, 3, 256, 256)) as mock_camrgb:
            output_dict = denoiser.denoise_bayer(torch.randn(4, 128, 128), torch.eye(3), return_dict=True)

            assert isinstance(output_dict, dict)
            assert "denoised_image" in output_dict
            assert "bpp" in output_dict
            assert output_dict["denoised_image"].shape == (3, 256, 256)
