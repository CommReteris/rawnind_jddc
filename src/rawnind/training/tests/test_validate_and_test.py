import pytest
import torch
from unittest.mock import MagicMock

from rawnind.training.clean_api import (
    TrainingConfig,
    CleanDenoiserTrainer,
    CleanDenoiseCompressTrainer,
    create_denoiser_trainer,
    create_denoise_compress_trainer,
)

"""
Objective: Validate validation and test pipeline integrity across all 4 training class variants using the Clean API architecture.
Test Criteria: For each trainer type, instantiate with TrainingConfig; create mock dataloaders; call validate() and test(); assert methods return valid results.
Fulfillment: Ensures validation methods execute successfully across variants using the Clean API that was designed for TrainingConfig, completing the architectural refactoring vision.
Components: Uses Clean API trainers (CleanDenoiserTrainer, CleanDenoiseCompressTrainer) with TrainingConfig objects via factory functions and their native validate()/test() interface.
Architectural Vision: This test completes the refactoring from legacy CLI-based classes to modern Clean API classes, using the Clean API's native interface instead of forcing legacy patterns.
"""

@pytest.fixture
def base_training_config():
    """Base TrainingConfig for Clean API testing."""
    return TrainingConfig(
        model_architecture="unet",
        input_channels=3,  # Will be adjusted per training type
        output_channels=3,
        learning_rate=1e-4,
        batch_size=1,
        crop_size=64,
        total_steps=10,
        validation_interval=1,
        loss_function="mse",
        device="cpu",
        patience=1,
        lr_decay_factor=0.5,
        additional_metrics=["mse"],
        filter_units=48,
        compression_lambda=1.0,
        bit_estimator_lr_multiplier=1.0,
        test_interval=1,
        test_crop_size=64,
        val_crop_size=64,
        num_crops_per_image=1,
        save_training_images=False,
    )

@pytest.fixture(params=[
    ("denoise_compress", "bayer_to_rgb"),
    ("denoise_compress", "rgb_to_rgb"),
    ("denoise", "bayer_to_rgb"),
    ("denoise", "rgb_to_rgb"),
])
def trainer_spec(request):
    """Parametrized fixture for trainer specifications."""
    return request.param

def test_validate_and_test_clean_api(trainer_spec, base_training_config):
    """Parametrized test for validation and testing using Clean API architecture."""
    trainer_type, training_type = trainer_spec
    
    # Adjust config for training type
    training_config = base_training_config
    if training_type == "bayer_to_rgb":
        training_config.input_channels = 4
        training_config.output_channels = 3
    else:  # rgb_to_rgb
        training_config.input_channels = 3
        training_config.output_channels = 3
    
    # Create trainer using Clean API factory functions
    if trainer_type == "denoise_compress":
        trainer = create_denoise_compress_trainer(training_type, training_config)
    else:  # denoise
        trainer = create_denoiser_trainer(training_type, training_config)
    
    # Create mock dataloader that provides proper tensor shapes
    def create_mock_dataloader():
        """Create a mock dataloader that yields one batch with proper tensor shapes."""
        return iter([{
            'clean_images': torch.randn(1, training_config.output_channels, 64, 64),
            'noisy_images': torch.randn(1, training_config.input_channels, 64, 64), 
            'masks': torch.ones(1, 1, 64, 64)
        }])
    
    # Test validation using Clean API's native interface
    val_result = trainer.validate(validation_dataloader=create_mock_dataloader())
    assert val_result is not None
    assert isinstance(val_result, dict)
    assert 'loss' in val_result
    assert isinstance(val_result['loss'], (int, float))
    
    # Test testing using Clean API's native interface  
    test_result = trainer.test(test_dataloader=create_mock_dataloader())
    assert test_result is not None
    assert isinstance(test_result, dict)
    assert 'loss' in test_result
    assert 'test_name' in test_result
    assert test_result['test_name'] == 'test'
    assert isinstance(test_result['loss'], (int, float))
    
    # Verify trainer was created successfully with Clean API
    assert trainer.config == training_config
    assert trainer.training_type == training_type
    assert hasattr(trainer, 'model')
    assert hasattr(trainer, 'optimizer')
    assert hasattr(trainer, '_create_model')
    assert hasattr(trainer, '_create_optimizer')
    assert hasattr(trainer, 'compute_loss')