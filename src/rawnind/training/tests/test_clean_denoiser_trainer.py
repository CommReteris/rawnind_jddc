import pytest
import torch
from unittest.mock import MagicMock, patch

from rawnind.training.clean_api import TrainingConfig, CleanDenoiserTrainer
from rawnind.dependencies import raw_processing as raw

@pytest.fixture
def base_denoiser_config():
    return TrainingConfig(
        model_architecture="unet",
        input_channels=3,
        output_channels=3,
        learning_rate=1e-4,
        batch_size=1,
        crop_size=128,
        total_steps=10,
        validation_interval=5,
        loss_function="mse",
        device="cpu",
    )

@pytest.fixture
def mock_inference_denoiser_factory():
    """Mock `create_rgb_denoiser` and `create_bayer_denoiser` from inference.clean_api."""
    with patch('rawnind.inference.clean_api.create_rgb_denoiser') as mock_rgb_denoiser_factory, \
         patch('rawnind.inference.clean_api.create_bayer_denoiser') as mock_bayer_denoiser_factory:
        
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_model.to.return_value = mock_model # Ensure .to() returns self
        mock_model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

        # For RGB denoiser
        mock_rgb_denoiser_factory.return_value = MagicMock(
            model=mock_model, 
            demosaic_fn=None # RGB has no demosaic_fn
        )
        # For Bayer denoiser
        mock_bayer_denoiser_factory.return_value = MagicMock(
            model=mock_model, 
            demosaic_fn=MagicMock() # Bayer has one
        )
        yield mock_rgb_denoiser_factory, mock_bayer_denoiser_factory

class TestCleanDenoiserTrainer:
    """Unit tests for the CleanDenoiserTrainer class."""

    def test_init_rgb_valid(self, base_denoiser_config, mock_inference_denoiser_factory):
        """Test RGB denoiser trainer initialization with valid config."""
        trainer = CleanDenoiserTrainer(config=base_denoiser_config, training_type="rgb_to_rgb")
        assert trainer.training_type == "rgb_to_rgb"
        assert trainer.config.input_channels == 3
        mock_inference_denoiser_factory[0].assert_called_once()
        assert trainer.demosaic_fn is None

    def test_init_bayer_valid(self, base_denoiser_config, mock_inference_denoiser_factory):
        """Test Bayer denoiser trainer initialization with valid config."""
        bayer_config = TrainingConfig(**vars(base_denoiser_config), input_channels=4, output_channels=3, crop_size=256)
        trainer = CleanDenoiserTrainer(config=bayer_config, training_type="bayer_to_rgb")
        assert trainer.training_type == "bayer_to_rgb"
        assert trainer.config.input_channels == 4
        mock_inference_denoiser_factory[1].assert_called_once()
        assert trainer.demosaic_fn is not None # Should be set for Bayer

    def test_init_rgb_invalid_channels(self, base_denoiser_config, mock_inference_denoiser_factory):
        """Test RGB denoiser trainer init with invalid input channels."""
        invalid_config = TrainingConfig(**vars(base_denoiser_config), input_channels=4)
        with pytest.raises(ValueError, match="RGB training requires 3 input channels"):
            CleanDenoiserTrainer(config=invalid_config, training_type="rgb_to_rgb")

    def test_init_bayer_invalid_channels(self, base_denoiser_config, mock_inference_denoiser_factory):
        """Test Bayer denoiser trainer init with invalid input channels."""
        invalid_config = TrainingConfig(**vars(base_denoiser_config), input_channels=3)
        with pytest.raises(ValueError, match="Bayer training requires 4 input channels"):
            CleanDenoiserTrainer(config=invalid_config, training_type="bayer_to_rgb")

    @pytest.mark.parametrize("model_output_channels, expected_demosaic_call, expected_camrgb_call_shape", [
        (4, True, (1, 3, 256, 256)), # Model outputs Bayer, needs demosaicing and color transform
        (3, False, (1, 3, 128, 128)), # Model outputs RGB, only needs color transform
    ])
    @patch('rawnind.dependencies.raw_processing.demosaic')
    @patch('rawnind.dependencies.raw_processing.camRGB_to_lin_rec2020_images')
    def test_process_bayer_output(self, mock_camrgb, mock_demosaic, model_output_channels, expected_demosaic_call, expected_camrgb_call_shape, base_denoiser_config, mock_inference_denoiser_factory):
        """Test process_bayer_output with mock data for demosaicing and color transformation."""
        bayer_config = TrainingConfig(**vars(base_denoiser_config), input_channels=4, output_channels=3, crop_size=128)
        trainer = CleanDenoiserTrainer(config=bayer_config, training_type="bayer_to_rgb")

        # Mock outputs for raw processing functions
        mock_demosaic.return_value = torch.randn(1, 3, 256, 256) # Demosaiced output (2x resolution)
        mock_camrgb.return_value = torch.randn(1, 3, 256, 256)   # Final linear Rec2020 output

        model_output_shape = (1, model_output_channels, 128, 128)
        model_output_tensor = torch.randn(model_output_shape)
        xyz_matrices = torch.eye(3).unsqueeze(0)
        bayer_input_tensor = torch.randn(1, 4, 128, 128)

        processed_output = trainer.process_bayer_output(
            model_output=model_output_tensor,
            xyz_matrices=xyz_matrices,
            bayer_input=bayer_input_tensor
        )

        if expected_demosaic_call:
            mock_demosaic.assert_called_once_with(model_output_tensor)
            mock_camrgb.assert_called_once_with(mock_demosaic.return_value, xyz_matrices)
        else:
            mock_demosaic.assert_not_called()
            mock_camrgb.assert_called_once_with(model_output_tensor, xyz_matrices)
        
        assert processed_output.shape == expected_camrgb_call_shape
        assert isinstance(processed_output, torch.Tensor)

    def test_process_bayer_output_unsupported_channels(self, base_denoiser_config, mock_inference_denoiser_factory):
        """Test process_bayer_output with unexpected model output channels."""
        bayer_config = TrainingConfig(**vars(base_denoiser_config), input_channels=4, output_channels=3)
        trainer = CleanDenoiserTrainer(config=bayer_config, training_type="bayer_to_rgb")
        
        with pytest.raises(ValueError, match="Unexpected model output channels"):
            trainer.process_bayer_output(
                model_output=torch.randn(1, 2, 128, 128), # 2 channels - unsupported
                xyz_matrices=torch.eye(3).unsqueeze(0),
                bayer_input=torch.randn(1, 4, 128, 128)
            )