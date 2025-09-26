import pytest
import torch
from unittest.mock import MagicMock, patch, call

from rawnind.training.clean_api import TrainingConfig, CleanDenoiseCompressTrainer

@pytest.fixture
def base_denoise_compress_config():
    return TrainingConfig(
        model_architecture="autoencoder",
        input_channels=3,
        output_channels=3,
        learning_rate=1e-4,
        batch_size=1,
        crop_size=128,
        total_steps=10,
        validation_interval=5,
        loss_function="mse",
        device="cpu",
        compression_lambda=0.01,
        bit_estimator_lr_multiplier=10.0,
        filter_units=48, # Needed for compression model input
    )

@pytest.fixture
def mock_compression_model():
    """Mock `AbstractRawImageCompressor` which also has get_parameters."""
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.to.return_value = mock_model
    # Simulate get_parameters from the actual model
    mock_model.get_parameters.return_value = [
        {'params': [torch.nn.Parameter(torch.randn(1))], 'lr': 1e-4, 'name': 'encoder'},
        {'params': [torch.nn.Parameter(torch.randn(1))], 'lr': 1e-4, 'name': 'decoder'}
    ]
    mock_model.Encoder = MagicMock() # For lowercase alias check
    mock_model.Decoder = MagicMock() # For lowercase alias check
    return mock_model

@pytest.fixture
def mock_bit_estimator():
    """Mock `MultiHeadBitEstimator`."""
    mock_estimator = MagicMock(spec=torch.nn.Module)
    mock_estimator.to.return_value = mock_estimator
    mock_estimator.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
    return mock_estimator

class TestCleanDenoiseCompressTrainer:
    """Unit tests for the CleanDenoiseCompressTrainer class."""

    @patch('rawnind.models.compression_autoencoders.AbstractRawImageCompressor')
    @patch('rawnind.models.bitEstimator.MultiHeadBitEstimator')
    def test_init_valid(self, MockMultiHeadBitEstimator, MockAbstractRawImageCompressor, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test denoise+compress trainer initialization with valid config."""
        MockAbstractRawImageCompressor.return_value = mock_compression_model
        MockMultiHeadBitEstimator.return_value = mock_bit_estimator

        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")

        assert trainer.config == base_denoise_compress_config
        assert trainer.training_type == "rgb_to_rgb"
        assert trainer.compression_lambda == base_denoise_compress_config.compression_lambda
        assert trainer.bit_estimator_lr_multiplier == base_denoise_compress_config.bit_estimator_lr_multiplier
        assert isinstance(trainer.bit_estimator, MagicMock)
        assert isinstance(trainer.model, MagicMock)
        
        MockAbstractRawImageCompressor.assert_called_once()
        MockMultiHeadBitEstimator.assert_called_once_with(
            channel=base_denoise_compress_config.filter_units * 2, # filter_units * 2 for bitstream_out_channels
            nb_head=16
        )
        assert trainer.model == trainer.compression_model # Alias check
        assert trainer.model.encoder == trainer.model.Encoder # Lowercase alias for encoder
        assert trainer.model.decoder == trainer.model.Decoder # Lowercase alias for decoder

    def test_init_missing_compression_lambda(self, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test init raises ValueError if compression_lambda is missing."""
        invalid_config = TrainingConfig(**vars(base_denoise_compress_config), compression_lambda=None)
        with pytest.raises(ValueError, match="compression_lambda must be specified"):
            CleanDenoiseCompressTrainer(config=invalid_config, training_type="rgb_to_rgb")

    @patch('rawnind.models.compression_autoencoders.AbstractRawImageCompressor')
    @patch('rawnind.models.bitEstimator.MultiHeadBitEstimator')
    def test_create_optimizer_multiple_param_groups(self, MockMultiHeadBitEstimator, MockAbstractRawImageCompressor, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test _create_optimizer sets up multiple parameter groups with correct LRs."""
        MockAbstractRawImageCompressor.return_value = mock_compression_model
        MockMultiHeadBitEstimator.return_value = mock_bit_estimator

        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")

        optimizer = trainer.optimizer
        assert isinstance(optimizer, torch.optim.Adam)
        assert len(optimizer.param_groups) > 1 

        # Verify learning rates for different groups
        found_model_lr = False
        found_bit_estimator_lr = False
        for param_group in optimizer.param_groups:
            if 'name' in param_group and (param_group['name'] == 'encoder' or param_group['name'] == 'decoder'):
                assert param_group['lr'] == base_denoise_compress_config.learning_rate
                found_model_lr = True
            if param_group['lr'] == base_denoise_compress_config.learning_rate * base_denoise_compress_config.bit_estimator_lr_multiplier:
                found_bit_estimator_lr = True
        
        assert found_model_lr, "Model parameter groups with correct LR not found"
        assert found_bit_estimator_lr, "Bit estimator parameter group with correct LR not found"

        # Test fallback when model doesn't have get_parameters
        mock_compression_model_no_get_params = MagicMock(spec=torch.nn.Module)
        mock_compression_model_no_get_params.to.return_value = mock_compression_model_no_get_params
        mock_compression_model_no_get_params.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]
        MockAbstractRawImageCompressor.return_value = mock_compression_model_no_get_params # Use this mock

        trainer_fallback = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")
        optimizer_fallback = trainer_fallback.optimizer
        assert len(optimizer_fallback.param_groups) == 2 # Model params + bit estimator params only
        assert optimizer_fallback.param_groups[0]['lr'] == base_denoise_compress_config.learning_rate
        assert optimizer_fallback.param_groups[1]['lr'] == base_denoise_compress_config.learning_rate * base_denoise_compress_config.bit_estimator_lr_multiplier


    def test_compute_loss_with_bpp(self, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test compute_loss with bpp for combined rate-distortion."""
        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")
        trainer.loss_fn = MagicMock(return_value=torch.tensor(0.1)) # Mock visual loss
        
        predictions = torch.randn(1, 3, 128, 128)
        ground_truth = torch.randn(1, 3, 128, 128)
        masks = torch.ones(1, 1, 128, 128)
        bpp_tensor = torch.tensor(0.5)

        combined_loss = trainer.compute_loss(predictions, ground_truth, masks, bpp_tensor)
        
        # Expected: (visual_loss * compression_lambda) + bpp
        expected_loss = (0.1 * base_denoise_compress_config.compression_lambda) + bpp_tensor
        assert torch.isclose(combined_loss, expected_loss)

    def test_compute_loss_without_bpp(self, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test compute_loss without bpp (should still apply lambda to visual loss)."""
        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")
        trainer.loss_fn = MagicMock(return_value=torch.tensor(0.1)) # Mock visual loss
        
        predictions = torch.randn(1, 3, 128, 128)
        ground_truth = torch.randn(1, 3, 128, 128)
        masks = torch.ones(1, 1, 128, 128)

        combined_loss = trainer.compute_loss(predictions, ground_truth, masks, bpp=None)
        
        # Expected: visual_loss * compression_lambda
        expected_loss = 0.1 * base_denoise_compress_config.compression_lambda
        assert torch.isclose(combined_loss, torch.tensor(expected_loss))

    def test_get_optimizer_param_groups(self, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test get_optimizer_param_groups returns current optimizer groups."""
        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")
        param_groups = trainer.get_optimizer_param_groups()
        assert param_groups == trainer.optimizer.param_groups
        assert len(param_groups) > 1 # Should have multiple groups

    def test_compute_joint_loss(self, base_denoise_compress_config, mock_compression_model, mock_bit_estimator):
        """Test compute_joint_loss returns total loss and components."""
        trainer = CleanDenoiseCompressTrainer(config=base_denoise_compress_config, training_type="rgb_to_rgb")
        trainer.loss_fn = MagicMock(return_value=torch.tensor(0.1)) # Mock visual loss
        
        predictions = torch.randn(1, 3, 128, 128)
        ground_truth = torch.randn(1, 3, 128, 128)
        masks = torch.ones(1, 1, 128, 128)
        bpp_tensor = torch.tensor(0.5)

        total_loss, loss_components = trainer.compute_joint_loss(predictions, ground_truth, masks, bpp_tensor)
        
        # Expected: visual_loss + bpp * compression_lambda
        expected_visual_loss = 0.1
        expected_rate_loss = bpp_tensor.item() * base_denoise_compress_config.compression_lambda
        expected_total_loss = expected_visual_loss + expected_rate_loss

        assert torch.isclose(total_loss, torch.tensor(expected_total_loss))
        assert loss_components['distortion_loss'] == expected_visual_loss
        assert loss_components['rate_loss'] == expected_rate_loss
        assert loss_components['combined_loss'] == expected_total_loss
