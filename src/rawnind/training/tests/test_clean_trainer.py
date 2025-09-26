import pytest
import torch
import os
from pathlib import Path
from unittest.mock import MagicMock, patch, Mock
import logging

from rawnind.training.clean_api import (
    TrainingConfig,
    CleanTrainer,
    create_denoiser_trainer,
    create_denoise_compress_trainer,
)
from rawnind.dependencies.pt_losses import losses as pt_losses_dict

@pytest.fixture
def base_training_config():
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

@pytest.fixture
def mock_pt_losses():
    """Mock the pt_losses.losses dictionary to control loss functions."""
    with patch('rawnind.training.clean_api.losses', new={
        "mse": MagicMock(return_value=MagicMock(return_value=torch.tensor(0.1))),
        "ms_ssim_loss": MagicMock(return_value=MagicMock(return_value=torch.tensor(0.9))),
        "l1_loss_pytorch": MagicMock(return_value=MagicMock(return_value=torch.tensor(0.05)))
    }) as mock_losses:
        yield mock_losses

class TestCleanTrainer:
    """Unit tests for the CleanTrainer base class."""

    def test_init(self, base_training_config, mock_inference_denoiser_factory):
        """Test initialization of CleanTrainer."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")

        assert trainer.config == base_training_config
        assert trainer.training_type == "rgb_to_rgb"
        assert trainer.device == torch.device("cpu")
        assert isinstance(trainer.model, MagicMock)
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        mock_inference_denoiser_factory[0].assert_called_once() # create_rgb_denoiser
        
    def test_create_model_rgb(self, base_training_config, mock_inference_denoiser_factory):
        """Test _create_model for RGB training type."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        trainer._create_model() # Call again to ensure it works
        mock_inference_denoiser_factory[0].assert_called_once_with(
            architecture="unet",
            device="cpu",
            filter_units=base_training_config.filter_units
        )
        assert trainer.demosaic_fn is None

    def test_create_model_bayer(self, base_training_config, mock_inference_denoiser_factory):
        """Test _create_model for Bayer training type."""
        bayer_config = TrainingConfig(**vars(base_training_config), input_channels=4, output_channels=3)
        trainer = CleanTrainer(config=bayer_config, training_type="bayer_to_rgb")
        trainer._create_model()
        mock_inference_denoiser_factory[1].assert_called_once_with(
            architecture="unet",
            device="cpu",
            filter_units=bayer_config.filter_units
        )
        assert isinstance(trainer.demosaic_fn, MagicMock) # Should be set for Bayer

    def test_create_optimizer(self, base_training_config, mock_inference_denoiser_factory):
        """Test _create_optimizer basic functionality."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.optimizer.param_groups[0]['lr'] == base_training_config.learning_rate

    @pytest.mark.parametrize("loss_name, expected_loss_class", [
        ("mse", pt_losses_dict["mse"]),
        ("ms_ssim", pt_losses_dict["ms_ssim"]),
        ("l1", torch.nn.L1Loss), # L1 is direct PyTorch
    ])
    def test_create_loss_function(self, base_training_config, mock_inference_denoiser_factory, mock_pt_losses, loss_name, expected_loss_class):
        """Test _create_loss_function mapping."""
        config = TrainingConfig(**vars(base_training_config), loss_function=loss_name)
        trainer = CleanTrainer(config=config, training_type="rgb_to_rgb")
        
        if loss_name == "l1":
            assert isinstance(trainer.loss_fn, torch.nn.L1Loss)
        else:
            mock_losses[loss_name if loss_name=="mse" else "ms_ssim_loss"].assert_called_once()


    @pytest.mark.parametrize("has_masks, is_bayer, prediction_res_factor", [
        (True, False, 1),  # RGB with masks
        (False, False, 1), # RGB no masks
        (True, True, 2),   # Bayer with masks, 2x resolution
        (True, True, 1),   # Bayer, predictions already 2x res, masks not
        (True, True, 0.5), # Bayer, predictions 0.5x res (unlikely, but test interpolation)
    ])
    def test_compute_loss_masking_and_bayer_interpolation(self, base_training_config, mock_inference_denoiser_factory, has_masks, is_bayer, prediction_res_factor):
        """Test compute_loss with masking and Bayer mask/GT interpolation."""
        crop_size = 128
        bayer_config = TrainingConfig(**vars(base_training_config), input_channels=4, output_channels=3, crop_size=crop_size)
        rgb_config = base_training_config
        
        config = bayer_config if is_bayer else rgb_config
        training_type = "bayer_to_rgb" if is_bayer else "rgb_to_rgb"
        trainer = CleanTrainer(config=config, training_type=training_type)
        trainer.loss_fn = MagicMock(return_value=torch.tensor(0.5)) # Mock loss function

        predictions_shape = (1, config.output_channels, crop_size * prediction_res_factor, crop_size * prediction_res_factor)
        gt_shape = (1, config.output_channels, crop_size, crop_size) # Original GT size
        mask_shape = (1, 1, crop_size, crop_size) # Original mask size

        predictions = torch.randn(predictions_shape)
        ground_truth = torch.randn(gt_shape)
        masks = torch.ones(mask_shape) if has_masks else None

        # Call compute_loss
        loss = trainer.compute_loss(predictions, ground_truth, masks)

        assert isinstance(loss, torch.Tensor)
        if has_masks and is_bayer and prediction_res_factor != 1:
            # Mask and ground truth should have been interpolated
            assert trainer.loss_fn.call_args[0][0].shape == predictions_shape
            assert trainer.loss_fn.call_args[0][1].shape == predictions_shape
        elif has_masks:
            # Masks applied, no interpolation needed
            assert trainer.loss_fn.call_args[0][0].shape == (1, config.output_channels, crop_size, crop_size)
        else:
            # No masks, direct loss computation
            assert trainer.loss_fn.call_args[0][0].shape == predictions_shape

    def test_update_learning_rate_improvement(self, base_training_config, mock_inference_denoiser_factory):
        """Test update_learning_rate when model improves."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        initial_lr = trainer.get_current_learning_rate()
        
        # Simulate improvement
        trainer.update_learning_rate(validation_metrics={'loss': 0.05}, step=1)
        assert trainer.best_validation_losses['loss'] == 0.05
        assert trainer.get_current_learning_rate() == initial_lr # LR should not change yet

    def test_update_learning_rate_decay(self, base_training_config, mock_inference_denoiser_factory):
        """Test update_learning_rate when no improvement and patience reached."""
        config = TrainingConfig(**vars(base_training_config), patience=5, lr_decay_factor=0.5)
        trainer = CleanTrainer(config=config, training_type="rgb_to_rgb")
        
        initial_lr = trainer.get_current_learning_rate()
        trainer.best_validation_losses['loss'] = 0.1 # Some initial best loss
        trainer.lr_adjustment_allowed_step = 5 # Set patience step

        # Simulate no improvement past patience
        trainer.update_learning_rate(validation_metrics={'loss': 0.15}, step=6) # step > lr_adjustment_allowed_step
        assert trainer.get_current_learning_rate() == initial_lr * config.lr_decay_factor
        assert trainer.lr_adjustment_allowed_step == 6 + config.patience # Patience reset

    @patch('torch.save')
    @patch('torch.load')
    @patch('torch.serialization.add_safe_globals')
    def test_save_and_load_checkpoint(self, mock_add_safe_globals, mock_torch_load, mock_torch_save, base_training_config, mock_inference_denoiser_factory, tmp_path):
        """Test save_checkpoint and load_checkpoint functionality."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        checkpoint_path = tmp_path / "test_checkpoint.pt"

        # Simulate a step
        trainer.current_step = 5
        trainer.best_validation_losses = {'loss': 0.01}

        # Save checkpoint
        trainer.save_checkpoint(step=trainer.current_step, checkpoint_path=str(checkpoint_path))
        mock_torch_save.assert_called_once()
        saved_data = mock_torch_save.call_args[0][0]
        assert saved_data['step'] == 5
        assert 'model_state' in saved_data
        assert 'optimizer_state' in saved_data
        assert saved_data['config'] == base_training_config

        # Mock torch.load to return the saved data
        mock_torch_load.return_value = saved_data

        # Create a new trainer to load into
        new_trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        new_trainer.model.load_state_dict(trainer.model.state_dict()) # Pre-load state to avoid issues with mock
        new_trainer.optimizer.load_state_dict(trainer.optimizer.state_dict())
        
        # Load checkpoint
        new_trainer.load_checkpoint(str(checkpoint_path)) # Call it directly

        mock_add_safe_globals.assert_called_with([TrainingConfig]) # Check argument
        assert mock_torch_load.call_args[1]['weights_only'] == False # Check keyword argument

        assert new_trainer.current_step == 5
        assert new_trainer.best_validation_losses == {'loss': 0.01}
        # Further checks can be added for model and optimizer state_dict equality

    @pytest.mark.parametrize("fallback_present", [True, False])
    def test_prepare_datasets_mock_fallback(self, base_training_config, mock_inference_denoiser_factory, fallback_present):
        """Test prepare_datasets uses mock fallback if dataset package import fails."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        
        if fallback_present:
            with patch('rawnind.training.clean_api.create_training_datasets', side_effect=ImportError), \
                 patch.object(trainer, '_create_mock_datasets') as mock_create_mock_datasets:
                
                mock_create_mock_datasets.return_value = {
                    'train_loader': MagicMock(),
                    'val_loader': MagicMock(),
                    'test_loader': MagicMock()
                }
                
                datasets = trainer.prepare_datasets(dataset_config={})
                mock_create_mock_datasets.assert_called_once()
                assert 'train_loader' in datasets
        else:
            # If import succeeds, it should call the real function.
            # We don't have enough mocks for the real dataset, so let's mock it again
            with patch('rawnind.training.clean_api.create_training_datasets') as mock_create_training_datasets, \
                 patch.object(trainer, '_create_mock_datasets') as mock_create_mock_datasets: # Ensure it's NOT called
                
                mock_create_training_datasets.return_value = { # Simulate successful return
                    'train_dataloader': MagicMock(),
                    'validation_dataloader': MagicMock(),
                    'test_dataloader': MagicMock()
                }

                datasets = trainer.prepare_datasets(dataset_config={})
                mock_create_training_datasets.assert_called_once()
                mock_create_mock_datasets.assert_not_called()
                assert 'train_loader' in datasets # Now correctly mapped to train_loader
                assert 'val_loader' in datasets
                assert 'test_loader' in datasets


    def test_validate_and_test_flow(self, base_training_config, mock_inference_denoiser_factory, tmp_path):
        """Test the basic flow of validate and test methods."""
        trainer = CleanTrainer(config=base_training_config, training_type="rgb_to_rgb")
        # Ensure model is in training mode initially
        trainer.model.train() 

        # Mock dataloader to return a few batches
        mock_dataloader_iter = iter([
            {'clean_images': torch.randn(1, 3, 128, 128), 'noisy_images': torch.randn(1, 3, 128, 128), 'masks': torch.ones(1, 1, 128, 128)},
            {'clean_images': torch.randn(1, 3, 128, 128), 'noisy_images': torch.randn(1, 3, 128, 128), 'masks': torch.ones(1, 1, 128, 128)}
        ])
        
        # Patch compute_image_metrics
        with patch('rawnind.inference.clean_api.compute_image_metrics', return_value={'psnr': 25.0}) as mock_compute_metrics, \
             patch.object(trainer, '_save_validation_outputs') as mock_save_outputs:

            # Test validate
            metrics_result = trainer.validate(
                validation_dataloader=mock_dataloader_iter,
                compute_metrics=['loss', 'psnr'],
                save_outputs=True,
                output_directory=str(tmp_path / "val_outputs")
            )

            assert 'loss' in metrics_result
            assert 'psnr' in metrics_result
            assert trainer.model.eval.called # Should switch to eval mode
            assert trainer.model.train.called # Should switch back to train mode

            mock_compute_metrics.assert_called_once()
            mock_save_outputs.assert_called_once() # Called for each batch if batch_size=1
            assert metrics_result.get('outputs_saved') == True

            # Reset mocks for test method
            mock_dataloader_iter = iter([
                {'clean_images': torch.randn(1, 3, 128, 128), 'noisy_images': torch.randn(1, 3, 128, 128), 'masks': torch.ones(1, 1, 128, 128)},
            ])
            mock_compute_metrics.reset_mock()
            mock_save_outputs.reset_mock()

            # Test test
            test_results = trainer.test(
                test_dataloader=mock_dataloader_iter,
                test_name="my_test_run",
                save_outputs=True,
                compute_metrics=['loss']
            )

            assert 'test_name' in test_results
            assert test_results['test_name'] == "my_test_run"
            assert 'loss' in test_results
            assert mock_save_outputs.called # Should also save outputs