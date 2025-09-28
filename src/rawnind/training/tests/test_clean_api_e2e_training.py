"""
E2E Tests for Training Package Clean API

This test suite demonstrates end-to-end training workflows using the clean,
programmatic interfaces of the training package, without CLI dependencies.
These tests validate full integration across dataset, training loops,
model instantiation, and experiment management.
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from rawnind.training.clean_api import (
    TrainingConfig,
    ExperimentConfig,
    create_denoiser_trainer,
    create_denoise_compress_trainer,
    create_experiment_manager,
    create_training_config_from_yaml,
)
from rawnind.dataset.clean_api import (
    create_training_dataset,
    create_validation_dataset,
    DatasetConfig
)

# Mock models for testing to avoid actual model loading and complex inference
@pytest.fixture
def mock_denoiser_model():
    """Mock a basic denoising model."""
    mock_model = MagicMock(spec=torch.nn.Module)
    # Ensure forward returns a tensor of appropriate shape
    mock_model.return_value = torch.randn(1, 3, 128, 128)
    # Add get_parameters method
    mock_model.get_parameters = MagicMock(return_value=[])
    return mock_model

@pytest.fixture
def mock_denoise_compress_model():
    """Mock a denoise+compress model that returns a dict."""
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.return_value = {
        "reconstructed_image": torch.randn(1, 3, 128, 128),
        "bpp": torch.tensor(0.5)
    }
    # Add mock get_parameters for parameter groups
    mock_model.get_parameters.return_value = [
        {'params': [torch.nn.Parameter(torch.randn(1))],'lr': 1e-4},
        {'params': [torch.nn.Parameter(torch.randn(1))], 'lr': 1e-3}
    ]
    mock_model.encoder = MagicMock()
    mock_model.decoder = MagicMock()
    return mock_model

@pytest.fixture
def mock_bayer_denoiser_model():
    """Mock a Bayer denoising model."""
    mock_model = MagicMock(spec=torch.nn.Module)
    # Bayer input results in RGB output (often 2x resolution)
    mock_model.return_value = torch.randn(1, 3, 256, 256)
    # Add get_parameters method
    mock_model.get_parameters = MagicMock(return_value=[])
    return mock_model

@pytest.fixture
def mock_dataloader(config_type="rgb_pairs", crop_size=128, input_channels=3, output_channels=3):
    """Create a mock dataloader yielding dummy batches."""
    def _mock_dataloader_factory():
        for _ in range(5): # Yield a few batches
            if config_type == "bayer_pairs":
                noisy_images = torch.randn(1, input_channels, crop_size, crop_size) # 4ch Bayer input
                clean_images = torch.randn(1, output_channels, crop_size * 2, crop_size * 2) # 3ch RGB GT, 2x res
                masks = torch.ones(1, 1, crop_size * 2, crop_size * 2) # Mask also 2x res
                rgb_xyz_matrices = torch.eye(3).unsqueeze(0)
                yield {
                    'clean_images': clean_images,
                    'noisy_images': noisy_images,
                    'masks': masks,
                    'rgb_xyz_matrices': rgb_xyz_matrices,
                    'image_paths': [f"mock_bayer_{_}.raw"]
                }
            else:
                yield {
                    'clean_images': torch.randn(1, output_channels, crop_size, crop_size),
                    'noisy_images': torch.randn(1, input_channels, crop_size, crop_size),
                    'masks': torch.ones(1, 1, crop_size, crop_size),
                    'image_paths': [f"mock_rgb_{_}.jpg"]
                }
    return _mock_dataloader_factory()

@pytest.fixture
def create_mock_dataset_factories(mock_dataloader):
    """Fixture to mock dataset creation functions."""
    with patch('rawnind.training.clean_api.create_training_dataset') as mock_create_training_dataset, \
         patch('rawnind.training.clean_api.create_validation_dataset') as mock_create_validation_dataset:
        
        mock_create_training_dataset.return_value = Mock()
        mock_create_training_dataset.return_value.return_value = mock_dataloader
        mock_create_validation_dataset.return_value = Mock()
        mock_create_validation_dataset.return_value.return_value = mock_dataloader
        yield

class TestE2ETrainingCleanAPI:
    """End-to-end tests for the training package's clean API."""

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mock_denoiser_model, mock_denoise_compress_model, mock_bayer_denoiser_model, create_mock_dataset_factories):
        """Patch model creation calls within clean_api.py."""
        # Patch the model creation specific to CleanTrainer base class
        # Correctly mock the return value from inference clean_api factory functions
        patcher_rgb_denoiser = patch('rawnind.inference.clean_api.create_rgb_denoiser')
        mock_create_rgb_denoiser = patcher_rgb_denoiser.start()
        mock_create_rgb_denoiser.return_value = MagicMock(model=mock_denoiser_model, demosaic_fn=None)

        patcher_bayer_denoiser = patch('rawnind.inference.clean_api.create_bayer_denoiser')
        mock_create_bayer_denoiser = patcher_bayer_denoiser.start()
        mock_create_bayer_denoiser.return_value = MagicMock(model=mock_bayer_denoiser_model, demosaic_fn=MagicMock())


        patcher_compression_autoencoder = patch('rawnind.models.compression_autoencoders.AbstractRawImageCompressor')
        mock_compression_autoencoder = patcher_compression_autoencoder.start()
        mock_compression_autoencoder.return_value = mock_denoise_compress_model
        
        patcher_bit_estimator = patch('rawnind.models.bitEstimator.MultiHeadBitEstimator')
        mock_bit_estimator = patcher_bit_estimator.start()
        mock_bit_estimator.return_value = MagicMock(spec=torch.nn.Module) # Mock for the bit estimator

        self.mock_create_rgb_denoiser = mock_create_rgb_denoiser
        self.mock_create_bayer_denoiser = mock_create_bayer_denoiser
        self.mock_compression_autoencoder = mock_compression_autoencoder
        self.mock_bit_estimator = mock_bit_estimator
        yield
        patcher_rgb_denoiser.stop()
        patcher_bayer_denoiser.stop()
        patcher_compression_autoencoder.stop()
        patcher_bit_estimator.stop()

    def test_rgb_denoiser_training_e2e(self, tmp_path):
        """
        Test a full RGB denoiser training workflow end-to-end.
        """
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=1,
            crop_size=128,
            total_steps=5,  # Short run for E2E
            validation_interval=2,
            loss_function="mse",
            device="cpu",
            additional_metrics=["psnr"]
        )

        experiment_config = ExperimentConfig(
            experiment_name="test_rgb_denoiser",
            save_directory=str(tmp_path / "exp_rgb_denoiser"),
            checkpoint_interval=2,
            metrics_to_track=["loss", "val_psnr"]
        )

        trainer = create_denoiser_trainer(training_type="rgb_to_rgb", config=config)
        experiment_manager = create_experiment_manager(config=experiment_config)
        
        # Mock actual dataloader iteration
        mock_train_dataloader = mock_dataloader(config_type="rgb_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)
        mock_val_dataloader = mock_dataloader(config_type="rgb_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)

        # Run training loop
        results = trainer.train(
            train_dataloader=mock_train_dataloader,
            validation_dataloader=mock_val_dataloader,
            experiment_manager=experiment_manager,
            max_steps=config.total_steps
        )

        assert results['steps_completed'] == config.total_steps
        assert 'final_loss' in results
        assert len(results['training_loss_history']) == config.total_steps
        assert len(results['validation_metrics_history']) > 0
        assert experiment_manager.metrics_history
        assert experiment_manager.config.checkpoint_dir.exists()
        assert list(experiment_manager.config.checkpoint_dir.iterdir()) # Check if checkpoints were saved

    def test_bayer_denoiser_training_e2e(self, tmp_path):
        """
        Test a full Bayer denoiser training workflow end-to-end (domain preservation).
        """
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=4,  # Bayer
            output_channels=3, # RGB output
            learning_rate=1e-4,
            batch_size=1,
            crop_size=128, # Must be >160 for MS-SSIM, but using 128 for general test. Validated separately.
            total_steps=5,
            validation_interval=2,
            loss_function="mse", # Using MSE for this E2E, MS-SSIM specific is tested below
            device="cpu",
            additional_metrics=["psnr"]
        )

        experiment_config = ExperimentConfig(
            experiment_name="test_bayer_denoiser",
            save_directory=str(tmp_path / "exp_bayer_denoiser"),
            checkpoint_interval=2,
            metrics_to_track=["loss", "val_psnr"]
        )

        trainer = create_denoiser_trainer(training_type="bayer_to_rgb", config=config)
        experiment_manager = create_experiment_manager(config=experiment_config)
        
        # Mock actual dataloader iteration, Bayer style
        mock_train_dataloader = mock_dataloader(config_type="bayer_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)
        mock_val_dataloader = mock_dataloader(config_type="bayer_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)

        results = trainer.train(
            train_dataloader=mock_train_dataloader,
            validation_dataloader=mock_val_dataloader,
            experiment_manager=experiment_manager,
            max_steps=config.total_steps
        )

        assert results['steps_completed'] == config.total_steps
        assert 'final_loss' in results
        assert trainer.demosaic_fn is not None
        # Assertions for Bayer processing effects would go here if not mocked

    def test_ms_ssim_crop_size_validation(self):
        """
        Test that TrainingConfig enforces MS-SSIM crop_size constraint.
        (Domain Validation).
        """
        with pytest.raises(ValueError, match="MS-SSIM requires crop_size > 160"):
            TrainingConfig(
                model_architecture="unet",
                input_channels=3, output_channels=3,
                learning_rate=1e-4, batch_size=1,
                crop_size=160,  # Invalid for MS-SSIM
                total_steps=1, validation_interval=1,
                loss_function="ms_ssim", device="cpu"
            )
        
        # Valid config should pass
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3, output_channels=3,
            learning_rate=1e-4, batch_size=1,
            crop_size=162,  # Valid
            total_steps=1, validation_interval=1,
            loss_function="ms_ssim", device="cpu"
        )
        assert config.is_valid()
    
    def test_denoise_compress_training_e2e(self, tmp_path):
        """
        Test a full denoise+compress training workflow end-to-end.
        (Rate-Distortion Optimization & Complex Init Management).
        """
        config = TrainingConfig(
            model_architecture="autoencoder", # Assuming 'autoencoder' is a valid compression model
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=1,
            crop_size=192, # Valid for MS-SSIM if used, general purpose here
            total_steps=5,
            validation_interval=2,
            loss_function="mse", # Visual loss
            compression_lambda=0.01, # Rate-distortion trade-off
            bit_estimator_lr_multiplier=10.0, # Separate LR for bit estimator
            device="cpu",
            additional_metrics=["bpp", "combined"]
        )

        experiment_config = ExperimentConfig(
            experiment_name="test_dc_trainer",
            save_directory=str(tmp_path / "exp_dc_trainer"),
            checkpoint_interval=2,
            metrics_to_track=["combined", "val_bpp"] # Track combined loss and BPP
        )

        trainer = create_denoise_compress_trainer(training_type="rgb_to_rgb", config=config)
        experiment_manager = create_experiment_manager(config=experiment_config)
        
        # Mock actual dataloader iteration
        mock_train_dataloader = mock_dataloader(config_type="rgb_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)
        mock_val_dataloader = mock_dataloader(config_type="rgb_pairs", crop_size=config.crop_size, input_channels=config.input_channels, output_channels=config.output_channels)

        # Ensure model has get_parameters for autoencoder
        assert hasattr(trainer.model, 'get_parameters')
        
        # Ensure optimizer has multiple parameter groups
        assert len(trainer.optimizer.param_groups) > 1
        # Check if bit estimator LR is correctly applied
        assert any(pg['lr'] == config.learning_rate * config.bit_estimator_lr_multiplier for pg in trainer.optimizer.param_groups)

        results = trainer.train(
            train_dataloader=mock_train_dataloader,
            validation_dataloader=mock_val_dataloader,
            experiment_manager=experiment_manager,
            max_steps=config.total_steps
        )

        assert results['steps_completed'] == config.total_steps
        assert 'final_loss' in results
        # Assert BPP and combined loss are recorded
        assert any("bpp" in m for m in experiment_manager.metrics_history)
        assert any("combined" in m for m in experiment_manager.metrics_history)

    def test_clean_trainer_checkpoint_security(self, tmp_path, mock_denoiser_model):
        """
        Test PyTorch 2.6+ checkpoint security with custom dataclasses.
        (PyTorch Security).
        """
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3, output_channels=3,
            learning_rate=1e-4, batch_size=1,
            crop_size=128, total_steps=1, validation_interval=1,
            loss_function="mse", device="cpu"
        )
        trainer = create_denoiser_trainer(training_type="rgb_to_rgb", config=config)
        
        # Ensure the mock model is used
        trainer.model = mock_denoiser_model
        
        checkpoint_path = tmp_path / "test_checkpoint.pt"
        trainer.save_checkpoint(step=1, checkpoint_path=str(checkpoint_path))
        
        # Manually load to check for weights_only=False and add_safe_globals
        # This part primarily verifies that trainer.load_checkpoint calls these.
        # We can't directly assert on the internal torch.load call behavior here.
        # However, if the load_checkpoint method *failed* due to security, it would raise.
        
        # Mock torch.load to verify weights_only=False and add_safe_globals are called.
        with patch('torch.load') as mock_torch_load, \
             patch('torch.serialization.add_safe_globals') as mock_add_safe_globals:
            
            # Simulate what torch.load would return for a valid checkpoint
            mock_torch_load.return_value = {
                'step': 1,
                'model_state': trainer.model.state_dict(),
                'config': config,
                'optimizer_state': trainer.optimizer.state_dict(),
                'best_validation_losses': {}
            }
            
            trainer.load_checkpoint(str(checkpoint_path))
            
            mock_add_safe_globals.assert_called_once_with([TrainingConfig])
            # Check the call arguments for torch.load:
            # We need to custom assert for a keyword argument as patch won't check kwargs directly
            assert any(kwargs.get('weights_only') == False for args, kwargs in mock_torch_load.call_args_list)
            assert str(checkpoint_path) == mock_torch_load.call_args[0][0] # Path should be the first argument

        # Verify that loading actually restored state (e.g., config)
        loaded_trainer = create_denoiser_trainer(training_type="rgb_to_rgb", config=TrainingConfig(
            model_architecture="unet", input_channels=3, output_channels=3,
            learning_rate=1e-5, batch_size=2, crop_size=64, total_steps=10, validation_interval=5,
            loss_function="l1", device="cuda" # Different params to ensure override
        ))
        loaded_trainer.load_checkpoint(str(checkpoint_path))
        assert loaded_trainer.config.learning_rate == config.learning_rate
        assert loaded_trainer.current_step == 1
