"""
E2E Tests for Training Package Clean API

This test suite demonstrates how the training package should work with clean,
programmatic interfaces without CLI dependencies. These tests serve as 
specifications for the desired API design.

The training package should provide:
1. Clean factory functions for creating trainers
2. Programmatic configuration without CLI parsing
3. Model training with explicit parameters
4. Experiment management and checkpointing
5. Validation and testing utilities
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import current training package (these imports will likely fail initially)
try:
    from rawnind.training import (
        create_denoiser_trainer,
        create_denoise_compress_trainer, 
        create_experiment_manager,
        TrainingConfig,
        ExperimentConfig
    )
except ImportError:
    # These are the clean interfaces we want to implement
    pytest.skip("Clean training interfaces not yet implemented", allow_module_level=True)


class TestTrainingFactoryFunctions:
    """Test clean factory functions for creating training components."""
    
    def test_create_denoiser_trainer_rgb(self):
        """Test creating RGB denoiser trainer with clean API."""
        # Configuration should be explicit, not CLI-based
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=4,
            crop_size=128,
            total_steps=1000,
            validation_interval=100,
            patience=500,
            loss_function="ms_ssim",
            device="cpu"
        )
        
        # Should create trainer without CLI dependencies
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Verify trainer properties
        assert trainer is not None
        assert trainer.config.input_channels == 3
        assert trainer.config.output_channels == 3
        assert trainer.config.learning_rate == 1e-4
        assert trainer.device == torch.device("cpu")
        assert hasattr(trainer, 'model')
        assert hasattr(trainer, 'optimizer')
        
    def test_create_denoiser_trainer_bayer(self):
        """Test creating Bayer denoiser trainer with clean API."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=4,  # Bayer has 4 channels
            output_channels=3,  # RGB output
            learning_rate=1e-4,
            batch_size=2,
            crop_size=256,
            total_steps=2000,
            validation_interval=200,
            patience=1000,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="bayer_to_rgb",
            config=config
        )
        
        assert trainer is not None
        assert trainer.config.input_channels == 4
        assert trainer.config.output_channels == 3
        assert hasattr(trainer, 'demosaic_fn')  # Bayer-specific functionality
        
    def test_create_denoise_compress_trainer(self):
        """Test creating joint denoise+compress trainer."""
        config = TrainingConfig(
            model_architecture="autoencoder",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=4,
            crop_size=128,
            total_steps=1000,
            validation_interval=100,
            loss_function="ms_ssim",
            compression_lambda=0.01,  # Compression-specific parameter
            bit_estimator_lr_multiplier=1.0,
            device="cpu"
        )
        
        trainer = create_denoise_compress_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        assert trainer is not None
        assert trainer.config.compression_lambda == 0.01
        assert hasattr(trainer, 'compression_model')
        assert hasattr(trainer, 'bit_estimator')
        
    def test_create_experiment_manager(self):
        """Test creating experiment manager with clean API."""
        exp_config = ExperimentConfig(
            experiment_name="test_denoiser",
            save_directory=tempfile.mkdtemp(),
            checkpoint_interval=100,
            keep_best_n_models=3,
            metrics_to_track=["loss", "ms_ssim", "psnr"]
        )
        
        exp_manager = create_experiment_manager(exp_config)
        
        assert exp_manager is not None
        assert exp_manager.config.experiment_name == "test_denoiser"
        assert exp_manager.config.keep_best_n_models == 3
        assert "ms_ssim" in exp_manager.config.metrics_to_track
        
        # Cleanup
        shutil.rmtree(exp_config.save_directory)


class TestTrainingConfigClasses:
    """Test configuration classes for training."""
    
    def test_training_config_validation(self):
        """Test that training config validates parameters correctly."""
        # Valid config should work
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=4,
            crop_size=128,
            total_steps=1000,
            validation_interval=100,
            loss_function="mse",
            device="cpu"
        )
        assert config.is_valid()
        
        # Invalid config should raise errors
        with pytest.raises(ValueError, match="Learning rate must be positive"):
            TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=-1e-4,  # Invalid
                batch_size=4,
                crop_size=128,
                total_steps=1000,
                validation_interval=100,
                loss_function="mse",
                device="cpu"
            )
            
    def test_experiment_config_path_resolution(self):
        """Test that experiment config resolves paths correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="test_exp",
                save_directory=tmpdir,
                checkpoint_interval=100,
                keep_best_n_models=3
            )
            
            # Should create necessary directories
            assert config.checkpoint_dir.exists()
            assert config.results_dir.exists()
            assert config.logs_dir.exists()


class TestCleanTrainingWorkflow:
    """Test complete training workflow with clean API."""
    
    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset for testing."""
        def mock_dataloader():
            for i in range(5):  # Small dataset for testing
                batch = {
                    'clean_images': torch.randn(2, 3, 64, 64),  # batch_size=2, RGB, 64x64
                    'noisy_images': torch.randn(2, 3, 64, 64),
                    'masks': torch.ones(2, 3, 64, 64),
                    'image_paths': [f'image_{i}_clean.jpg', f'image_{i}_noisy.jpg']
                }
                yield batch
        return mock_dataloader
    
    def test_training_workflow_rgb_denoiser(self, mock_dataset):
        """Test complete training workflow for RGB denoiser."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create trainer with clean config
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=2,
                crop_size=64,
                total_steps=10,  # Very short for testing
                validation_interval=5,
                loss_function="mse",
                device="cpu"
            )
            
            exp_config = ExperimentConfig(
                experiment_name="test_rgb_denoiser",
                save_directory=tmpdir,
                checkpoint_interval=5,
                keep_best_n_models=2
            )
            
            trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            exp_manager = create_experiment_manager(exp_config)
            
            # Set up training data
            train_loader = mock_dataset()
            val_loader = mock_dataset()
            
            # Run training
            training_results = trainer.train(
                train_dataloader=train_loader,
                validation_dataloader=val_loader,
                experiment_manager=exp_manager
            )
            
            # Verify training completed successfully
            assert training_results['steps_completed'] == config.total_steps
            assert 'final_loss' in training_results
            assert training_results['final_loss'] > 0
            
            # Verify checkpoints were saved
            checkpoint_files = list(Path(tmpdir).glob("**/*.pt"))
            assert len(checkpoint_files) >= 1
            
            # Verify training metrics were recorded
            assert 'training_loss_history' in training_results
            assert len(training_results['training_loss_history']) > 0
    
    def test_training_workflow_bayer_denoiser(self, mock_dataset):
        """Test complete training workflow for Bayer denoiser."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=4,  # Bayer pattern
                output_channels=3,  # RGB output
                learning_rate=1e-4,
                batch_size=2,
                crop_size=64,
                total_steps=10,
                validation_interval=5,
                loss_function="ms_ssim",
                device="cpu"
            )
            
            exp_config = ExperimentConfig(
                experiment_name="test_bayer_denoiser",
                save_directory=tmpdir,
                checkpoint_interval=5,
                keep_best_n_models=2
            )
            
            trainer = create_denoiser_trainer(
                training_type="bayer_to_rgb",
                config=config
            )
            
            # Mock Bayer dataset
            def mock_bayer_dataloader():
                for i in range(5):
                    batch = {
                        'clean_images': torch.randn(2, 3, 64, 64),  # Ground truth RGB
                        'noisy_images': torch.randn(2, 4, 64, 64),  # Noisy Bayer
                        'masks': torch.ones(2, 3, 64, 64),
                        'rgb_xyz_matrices': torch.randn(2, 3, 3),  # Color transformation matrices
                        'image_paths': [f'bayer_{i}_clean.jpg', f'bayer_{i}_noisy.raw']
                    }
                    yield batch
            
            exp_manager = create_experiment_manager(exp_config)
            
            training_results = trainer.train(
                train_dataloader=mock_bayer_dataloader(),
                validation_dataloader=mock_bayer_dataloader(),
                experiment_manager=exp_manager
            )
            
            assert training_results['steps_completed'] == config.total_steps
            assert 'final_loss' in training_results


class TestModelCheckpointingAndResumption:
    """Test model checkpointing and training resumption."""
    
    def test_save_and_load_checkpoint(self):
        """Test saving and loading training checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=4,
                crop_size=64,
                total_steps=100,
                validation_interval=25,
                loss_function="mse",
                device="cpu"
            )
            
            trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            # Save checkpoint
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")
            checkpoint_info = trainer.save_checkpoint(
                step=50,
                checkpoint_path=checkpoint_path,
                include_optimizer=True
            )
            
            assert os.path.exists(checkpoint_path)
            assert checkpoint_info['step'] == 50
            assert 'model_state' in checkpoint_info
            assert 'optimizer_state' in checkpoint_info
            
            # Load checkpoint into new trainer
            new_trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            resume_info = new_trainer.load_checkpoint(checkpoint_path)
            
            assert resume_info['step'] == 50
            assert new_trainer.current_step == 50
            
            # Model parameters should match
            original_params = list(trainer.model.parameters())
            loaded_params = list(new_trainer.model.parameters())
            
            for orig, loaded in zip(original_params, loaded_params):
                assert torch.allclose(orig, loaded)
    
    def test_resume_training_from_checkpoint(self):
        """Test resuming training from a saved checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=2,
                crop_size=64,
                total_steps=20,
                validation_interval=10,
                loss_function="mse",
                device="cpu"
            )
            
            # Create and partially train a model
            trainer1 = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            # Mock data for partial training
            def mock_dataloader():
                for i in range(3):
                    yield {
                        'clean_images': torch.randn(2, 3, 64, 64),
                        'noisy_images': torch.randn(2, 3, 64, 64),
                        'masks': torch.ones(2, 3, 64, 64),
                        'image_paths': [f'image_{i}_clean.jpg', f'image_{i}_noisy.jpg']
                    }
            
            # Run partial training
            exp_config = ExperimentConfig(
                experiment_name="test_resume",
                save_directory=tmpdir,
                checkpoint_interval=10
            )
            exp_manager = create_experiment_manager(exp_config)
            
            # Train partially (should save checkpoint at step 10)
            partial_results = trainer1.train(
                train_dataloader=mock_dataloader(),
                validation_dataloader=mock_dataloader(),
                experiment_manager=exp_manager,
                max_steps=10
            )
            
            assert partial_results['steps_completed'] == 10
            
            # Create new trainer and resume
            trainer2 = create_denoiser_trainer(
                training_type="rgb_to_rgb", 
                config=config
            )
            
            # Load from experiment directory
            resume_info = trainer2.resume_from_experiment(exp_config.save_directory)
            assert resume_info['resumed_from_step'] == 10
            
            # Continue training
            continued_results = trainer2.train(
                train_dataloader=mock_dataloader(),
                validation_dataloader=mock_dataloader(),
                experiment_manager=exp_manager,
                max_steps=config.total_steps  # Complete the training
            )
            
            assert continued_results['steps_completed'] == config.total_steps


class TestExperimentManagement:
    """Test experiment management utilities."""
    
    def test_experiment_manager_basic_functionality(self):
        """Test basic experiment manager operations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="test_experiment",
                save_directory=tmpdir,
                checkpoint_interval=50,
                keep_best_n_models=3,
                metrics_to_track=["loss", "ms_ssim", "psnr"]
            )
            
            exp_manager = create_experiment_manager(config)
            
            # Test recording metrics
            exp_manager.record_metrics(
                step=100,
                metrics={
                    "train_loss": 0.5,
                    "val_loss": 0.6,
                    "val_ms_ssim": 0.85,
                    "val_psnr": 28.5
                }
            )
            
            # Test getting best steps
            best_steps = exp_manager.get_best_steps()
            assert isinstance(best_steps, dict)
            assert "loss" in best_steps or len(best_steps) == 0  # Might be empty initially
            
            # Test checkpoint management
            checkpoint_info = exp_manager.should_save_checkpoint(step=100)
            assert checkpoint_info['should_save'] == True  # Should save at interval
            
            checkpoint_info = exp_manager.should_save_checkpoint(step=25)
            assert checkpoint_info['should_save'] == False  # Not at interval
    
    def test_experiment_cleanup(self):
        """Test experiment cleanup functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="test_cleanup",
                save_directory=tmpdir,
                checkpoint_interval=25,
                keep_best_n_models=2
            )
            
            exp_manager = create_experiment_manager(config)
            
            # Simulate saving multiple checkpoints
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            for step in [25, 50, 75, 100, 125]:
                checkpoint_path = checkpoint_dir / f"model_step_{step}.pt"
                torch.save({'step': step, 'loss': 1.0 / step}, checkpoint_path)
                
                exp_manager.record_metrics(
                    step=step,
                    metrics={"val_loss": 1.0 / step}  # Later steps have lower loss
                )
            
            # Run cleanup - should keep only best models
            cleaned_info = exp_manager.cleanup_checkpoints()
            
            assert cleaned_info['checkpoints_removed'] > 0
            assert cleaned_info['checkpoints_kept'] == config.keep_best_n_models
            
            # Verify best models are kept
            remaining_files = list(checkpoint_dir.glob("*.pt"))
            assert len(remaining_files) <= config.keep_best_n_models


class TestTrainingDataIntegration:
    """Test integration with dataset package."""
    
    @patch('rawnind.dataset.create_training_dataset')  # Mock dataset creation
    def test_training_with_real_dataset_interface(self, mock_create_dataset):
        """Test training with dataset package integration."""
        # Mock dataset creation to return our test data
        def mock_rgb_dataset():
            for i in range(3):
                yield {
                    'clean_images': torch.randn(2, 3, 128, 128),
                    'noisy_images': torch.randn(2, 3, 128, 128),
                    'masks': torch.ones(2, 3, 128, 128),
                    'image_paths': [f'clean_{i}.jpg', f'noisy_{i}.jpg']
                }
        
        mock_create_dataset.return_value = {
            'train_loader': mock_rgb_dataset(),
            'val_loader': mock_rgb_dataset(),
            'test_loader': mock_rgb_dataset()
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=2,
                crop_size=128,
                total_steps=15,
                validation_interval=5,
                loss_function="mse",
                device="cpu"
            )
            
            trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            # Use dataset package interface (mocked)
            dataset_config = {
                'dataset_type': 'rgb_pairs',
                'train_data_paths': ['/path/to/train'],
                'val_data_paths': ['/path/to/val'],
                'crop_size': 128,
                'augmentations': ['flip', 'rotate']
            }
            
            datasets = trainer.prepare_datasets(dataset_config)
            
            # Verify dataset preparation
            assert 'train_loader' in datasets
            assert 'val_loader' in datasets
            
            # Run training with prepared datasets
            exp_config = ExperimentConfig(
                experiment_name="test_dataset_integration",
                save_directory=tmpdir,
                checkpoint_interval=5
            )
            exp_manager = create_experiment_manager(exp_config)
            
            results = trainer.train(
                train_dataloader=datasets['train_loader'],
                validation_dataloader=datasets['val_loader'],
                experiment_manager=exp_manager
            )
            
            assert results['steps_completed'] == config.total_steps


class TestTrainingValidationAndTesting:
    """Test validation and testing functionality during training."""
    
    def test_standalone_validation(self):
        """Test running validation independently of training loop."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=100,
            validation_interval=25,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Mock validation data
        def val_dataloader():
            for i in range(3):
                yield {
                    'clean_images': torch.randn(1, 3, 64, 64),
                    'noisy_images': torch.randn(1, 3, 64, 64),
                    'masks': torch.ones(1, 3, 64, 64),
                    'image_paths': [f'val_{i}.jpg']
                }
        
        # Run standalone validation
        val_results = trainer.validate(
            validation_dataloader=val_dataloader(),
            compute_metrics=['loss', 'ms_ssim', 'psnr']
        )
        
        assert 'loss' in val_results
        assert 'ms_ssim' in val_results
        assert 'psnr' in val_results
        assert all(isinstance(v, float) for v in val_results.values())
    
    def test_custom_test_evaluation(self):
        """Test running custom test evaluation."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=1,
            crop_size=64,
            total_steps=100,
            validation_interval=25,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Mock custom test data
        def custom_test_dataloader():
            for i in range(2):
                yield {
                    'clean_images': torch.randn(1, 3, 64, 64),
                    'noisy_images': torch.randn(1, 3, 64, 64),
                    'masks': torch.ones(1, 3, 64, 64),
                    'image_paths': [f'custom_test_{i}.jpg']
                }
        
        # Run custom test
        test_results = trainer.test(
            test_dataloader=custom_test_dataloader(),
            test_name="custom_evaluation",
            save_outputs=False,
            compute_metrics=['loss', 'ms_ssim']
        )
        
        assert 'loss' in test_results
        assert 'ms_ssim' in test_results
        assert test_results['test_name'] == "custom_evaluation"


class TestHyperparameterManagement:
    """Test hyperparameter management and optimization."""
    
    def test_learning_rate_scheduling(self):
        """Test learning rate scheduling during training."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-3,  # Higher initial LR
            batch_size=2,
            crop_size=64,
            total_steps=100,
            validation_interval=20,
            patience=40,  # LR decay patience
            lr_decay_factor=0.5,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Initial learning rate should match config
        assert trainer.get_current_learning_rate() == 1e-3
        
        # First establish good performance baseline
        trainer.update_learning_rate(
        validation_metrics={'loss': 0.1},  # Good performance
        step=20
        )
        
        # Then simulate poor validation performance to trigger LR decay
        trainer.update_learning_rate(
            validation_metrics={'loss': 0.9},  # Poor performance
            step=80  # Past patience window (20 + 40 = 60, so 80 > 60)
        )

        # Learning rate should have decayed
        new_lr = trainer.get_current_learning_rate()
        assert new_lr < 1e-3  # Should be reduced
        assert abs(new_lr - 5e-4) < 1e-6  # Should be ~0.5 * 1e-3
    
    def test_early_stopping(self):
        """Test early stopping functionality."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=1000,  # Long training
            validation_interval=10,
            early_stopping_patience=30,  # Stop if no improvement for 30 steps
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Mock data that doesn't improve (constant loss)
        def mock_dataloader():
            for i in range(100):  # Many batches
                yield {
                    'clean_images': torch.randn(2, 3, 64, 64),
                    'noisy_images': torch.randn(2, 3, 64, 64),
                    'masks': torch.ones(2, 3, 64, 64),
                    'image_paths': [f'image_{i}.jpg']
                }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_name="test_early_stopping",
                save_directory=tmpdir,
                checkpoint_interval=10
            )
            exp_manager = create_experiment_manager(exp_config)
            
            # Training should stop early due to no improvement
            results = trainer.train(
                train_dataloader=mock_dataloader(),
                validation_dataloader=mock_dataloader(),
                experiment_manager=exp_manager
            )
            
            # Should have stopped before total_steps due to early stopping
            assert results['steps_completed'] < config.total_steps
            assert results['early_stopped'] == True
            assert results['early_stop_reason'] == "No improvement in validation loss"


class TestTrainingMetrics:
    """Test training metrics computation and tracking."""
    
    def test_loss_computation(self):
        """Test that loss functions work correctly."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=10,
            validation_interval=5,
            loss_function="ms_ssim",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Test loss computation directly
        pred_images = torch.randn(2, 3, 64, 64)
        gt_images = torch.randn(2, 3, 64, 64)
        masks = torch.ones(2, 3, 64, 64)
        
        loss_value = trainer.compute_loss(
            predictions=pred_images,
            ground_truth=gt_images,
            masks=masks
        )
        
        assert isinstance(loss_value, torch.Tensor)
        assert loss_value.requires_grad == True
        assert loss_value.dim() == 0  # Scalar loss
    
    def test_metrics_computation_during_training(self):
        """Test that metrics are computed correctly during training."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=10,
            validation_interval=5,
            loss_function="mse",
            additional_metrics=['ms_ssim', 'psnr'],
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        def mock_dataloader():
            for i in range(3):
                yield {
                    'clean_images': torch.randn(2, 3, 64, 64),
                    'noisy_images': torch.randn(2, 3, 64, 64),
                    'masks': torch.ones(2, 3, 64, 64),
                    'image_paths': [f'image_{i}.jpg']
                }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_name="test_metrics",
                save_directory=tmpdir,
                checkpoint_interval=5
            )
            exp_manager = create_experiment_manager(exp_config)
            
            results = trainer.train(
                train_dataloader=mock_dataloader(),
                validation_dataloader=mock_dataloader(),
                experiment_manager=exp_manager
            )
            
            # Verify additional metrics were computed
            assert 'validation_metrics_history' in results
            for step_metrics in results['validation_metrics_history']:
                assert 'ms_ssim' in step_metrics
                assert 'psnr' in step_metrics


class TestJointDenoiseCompressTraining:
    """Test joint denoising+compression training functionality."""
    
    def test_denoise_compress_trainer_creation(self):
        """Test creating joint denoise+compress trainer."""
        config = TrainingConfig(
            model_architecture="autoencoder",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=128,
            total_steps=100,
            validation_interval=25,
            loss_function="ms_ssim",
            compression_lambda=0.01,
            bit_estimator_lr_multiplier=2.0,
            device="cpu"
        )
        
        trainer = create_denoise_compress_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Verify specific denoise+compress functionality
        assert trainer is not None
        assert hasattr(trainer, 'compression_model')
        assert hasattr(trainer, 'bit_estimator') 
        assert trainer.config.compression_lambda == 0.01
        
        # Should have multiple optimizer parameter groups
        param_groups = trainer.get_optimizer_param_groups()
        assert len(param_groups) >= 2  # At least autoencoder + bit estimator
    
    def test_joint_loss_computation(self):
        """Test joint loss computation (distortion + rate)."""
        config = TrainingConfig(
            model_architecture="autoencoder",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=10,
            validation_interval=5,
            loss_function="ms_ssim",
            compression_lambda=0.1,
            device="cpu"
        )
        
        trainer = create_denoise_compress_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Mock model output with compression
        pred_images = torch.randn(2, 3, 64, 64)
        gt_images = torch.randn(2, 3, 64, 64)
        masks = torch.ones(2, 3, 64, 64)
        bpp = torch.tensor(2.5)  # Bits per pixel
        
        total_loss, loss_components = trainer.compute_joint_loss(
            predictions=pred_images,
            ground_truth=gt_images,
            masks=masks,
            bits_per_pixel=bpp
        )
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.requires_grad == True
        assert 'distortion_loss' in loss_components
        assert 'rate_loss' in loss_components
        assert loss_components['rate_loss'] == bpp * config.compression_lambda


class TestBayerSpecificTraining:
    """Test Bayer-specific training functionality."""
    
    def test_bayer_training_with_demosaicing(self):
        """Test Bayer training with demosaicing operations."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=4,  # Bayer channels
            output_channels=3,  # RGB output  
            learning_rate=1e-4,
            batch_size=2,
            crop_size=128,
            total_steps=10,
            validation_interval=5,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="bayer_to_rgb",
            config=config
        )
        
        # Verify Bayer-specific functionality
        assert hasattr(trainer, 'demosaic_fn')
        assert hasattr(trainer, 'process_bayer_output')
        
        # Test Bayer-specific processing
        bayer_input = torch.randn(2, 4, 128, 128)  # Bayer pattern
        rgb_output = torch.randn(2, 3, 128, 128)  # Model output
        xyz_matrices = torch.randn(2, 3, 3)  # Color transformation
        
        processed_output = trainer.process_bayer_output(
            model_output=rgb_output,
            xyz_matrices=xyz_matrices,
            bayer_input=bayer_input
        )
        
        assert processed_output.shape == (2, 3, 128, 128)  # Should be RGB
    
    def test_bayer_color_matrix_handling(self):
        """Test handling of color transformation matrices in Bayer training."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=4,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=1,
            crop_size=64,
            total_steps=5,
            validation_interval=5,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="bayer_to_rgb",
            config=config
        )
        
        # Mock Bayer batch with color matrices
        def bayer_dataloader():
            yield {
                'clean_images': torch.randn(1, 3, 64, 64),
                'noisy_images': torch.randn(1, 4, 64, 64),  # Bayer
                'masks': torch.ones(1, 3, 64, 64),
                'rgb_xyz_matrices': torch.eye(3).unsqueeze(0),  # Identity matrix
                'image_paths': ['test_bayer.raw']
            }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            exp_config = ExperimentConfig(
                experiment_name="test_bayer_matrices",
                save_directory=tmpdir,
                checkpoint_interval=5
            )
            exp_manager = create_experiment_manager(exp_config)
            
            results = trainer.train(
                train_dataloader=bayer_dataloader(),
                validation_dataloader=bayer_dataloader(),
                experiment_manager=exp_manager
            )
            
            assert results['steps_completed'] == config.total_steps


class TestTrainingConfigurationValidation:
    """Test configuration validation and error handling."""
    
    def test_invalid_model_architecture(self):
        """Test handling of invalid model architectures."""
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            TrainingConfig(
                model_architecture="invalid_arch",  # Not supported
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=4,
                crop_size=128,
                total_steps=1000,
                validation_interval=100,
                loss_function="mse",
                device="cpu"
            )
    
    def test_invalid_loss_function(self):
        """Test handling of invalid loss functions."""
        with pytest.raises(ValueError, match="Unsupported loss function"):
            TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=4,
                crop_size=128,
                total_steps=1000,
                validation_interval=100,
                loss_function="invalid_loss",  # Not supported
                device="cpu"
            )
    
    def test_channel_mismatch_validation(self):
        """Test validation of input/output channel configurations."""
        # Bayer to RGB should be 4->3
        with pytest.raises(ValueError, match="Bayer training requires 4 input channels"):
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,  # Should be 4 for Bayer
                output_channels=3,
                learning_rate=1e-4,
                batch_size=4,
                crop_size=128,
                total_steps=1000,
                validation_interval=100,
                loss_function="mse",
                device="cpu"
            )
            
            create_denoiser_trainer(
                training_type="bayer_to_rgb",  # Bayer type but wrong channels
                config=config
            )


class TestTrainingModelArchitectures:
    """Test different model architectures for training."""
    
    def test_unet_architecture_training(self):
        """Test training with UNet architecture."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=5,
            validation_interval=5,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Verify UNet-specific properties
        assert trainer.model_architecture == "unet"
        assert hasattr(trainer.model, 'convs1')  # UNet encoder level 1
        assert hasattr(trainer.model, 'up1')     # UNet decoder level 1
        assert hasattr(trainer.model, 'output_module')  # UNet output layer
    
    def test_autoencoder_architecture_training(self):
        """Test training with autoencoder architecture for compression."""
        config = TrainingConfig(
            model_architecture="autoencoder",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=64,
            total_steps=5,
            validation_interval=5,
            loss_function="ms_ssim",
            compression_lambda=0.01,
            device="cpu"
        )
        
        trainer = create_denoise_compress_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        # Verify autoencoder-specific properties
        assert trainer.model_architecture == "autoencoder"
        assert hasattr(trainer.model, 'encoder')
        assert hasattr(trainer.model, 'decoder')
        assert hasattr(trainer, 'bit_estimator')


class TestMultiDeviceTraining:
    """Test training on different devices (CPU/GPU)."""
    
    def test_cpu_training(self):
        """Test training on CPU device."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=2,
            crop_size=32,  # Small for CPU
            total_steps=5,
            validation_interval=5,
            loss_function="mse",
            device="cpu"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        assert trainer.device == torch.device("cpu")
        assert next(trainer.model.parameters()).device == torch.device("cpu")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self):
        """Test training on GPU device (if available)."""
        config = TrainingConfig(
            model_architecture="unet",
            input_channels=3,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=4,
            crop_size=64,
            total_steps=5,
            validation_interval=5,
            loss_function="mse",
            device="cuda"
        )
        
        trainer = create_denoiser_trainer(
            training_type="rgb_to_rgb",
            config=config
        )
        
        assert trainer.device.type == "cuda"
        assert next(trainer.model.parameters()).device.type == "cuda"


class TestTrainingOutputSaving:
    """Test saving training outputs and debugging information."""
    
    def test_save_training_visualizations(self):
        """Test saving training visualizations during training."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=2,
                crop_size=64,
                total_steps=5,
                validation_interval=5,
                loss_function="mse",
                save_training_images=True,  # Enable saving
                device="cpu"
            )
            
            trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            def mock_dataloader():
                for i in range(2):
                    yield {
                        'clean_images': torch.randn(2, 3, 64, 64),
                        'noisy_images': torch.randn(2, 3, 64, 64),
                        'masks': torch.ones(2, 3, 64, 64),
                        'image_paths': [f'train_{i}_clean.jpg', f'train_{i}_noisy.jpg']
                    }
            
            exp_config = ExperimentConfig(
                experiment_name="test_visualizations",
                save_directory=tmpdir,
                checkpoint_interval=5
            )
            exp_manager = create_experiment_manager(exp_config)
            
            results = trainer.train(
                train_dataloader=mock_dataloader(),
                validation_dataloader=mock_dataloader(),
                experiment_manager=exp_manager
            )
            
            # Verify visualization files were saved
            visu_dir = Path(tmpdir) / "visualizations"
            if config.save_training_images:
                assert visu_dir.exists()
                image_files = list(visu_dir.glob("**/*.exr"))
                assert len(image_files) > 0  # Should have saved some images
    
    def test_save_validation_outputs(self):
        """Test saving validation outputs for analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                model_architecture="unet",
                input_channels=3,
                output_channels=3,
                learning_rate=1e-4,
                batch_size=1,
                crop_size=64,
                total_steps=5,
                validation_interval=5,
                loss_function="mse",
                device="cpu"
            )
            
            trainer = create_denoiser_trainer(
                training_type="rgb_to_rgb",
                config=config
            )
            
            def val_dataloader():
                yield {
                    'clean_images': torch.randn(1, 3, 64, 64),
                    'noisy_images': torch.randn(1, 3, 64, 64),
                    'masks': torch.ones(1, 3, 64, 64),
                    'image_paths': ['validation_sample.jpg']
                }
            
            # Run validation with output saving
            val_results = trainer.validate(
                validation_dataloader=val_dataloader(),
                compute_metrics=['loss', 'ms_ssim'],
                save_outputs=True,
                output_directory=tmpdir
            )
            
            assert 'loss' in val_results
            assert 'ms_ssim' in val_results
            
            # Verify output files were saved
            output_files = list(Path(tmpdir).glob("**/*_output.exr"))
            if val_results.get('outputs_saved', False):
                assert len(output_files) > 0


# These tests demonstrate the clean API we want to implement for the training package.
# The training package should support:
# 1. Factory functions for creating trainers without CLI dependencies
# 2. Configuration classes for explicit parameter specification
# 3. Clean training workflows with proper experiment management
# 4. Model checkpointing and resumption
# 5. Validation and testing utilities
# 6. Hyperparameter management (LR scheduling, early stopping)
# 7. Metrics computation and tracking
# 8. Support for different model architectures and training types
# 9. Multi-device training (CPU/GPU)
# 10. Training visualization and output saving