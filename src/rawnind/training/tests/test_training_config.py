import pytest
from pathlib import Path

from rawnind.training.clean_api import TrainingConfig, create_training_config_from_yaml

class TestTrainingConfig:
    """Unit tests for TrainingConfig dataclass and its validation."""

    def test_valid_config_init(self):
        """Test happy path initialization of TrainingConfig."""
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
            device="cpu",
            additional_metrics=["psnr", "ssim"]
        )
        assert config.model_architecture == "unet"
        assert config.learning_rate == 1e-4
        assert config.crop_size == 128
        assert config.is_valid()
        assert config.test_crop_size == 128
        assert config.val_crop_size == 128
        assert "psnr" in config.additional_metrics

    @pytest.mark.parametrize(
        "param, value, expected_match",
        [
            ("learning_rate", 0, "Learning rate must be positive"),
            ("batch_size", 0, "Batch size must be positive"),
            ("crop_size", 0, "Crop size must be positive"),
            ("total_steps", 0, "Total steps must be positive"),
            ("validation_interval", 0, "Validation interval must be positive"),
            ("model_architecture", "invalid_arch", "Unsupported model architecture"),
            ("loss_function", "invalid_loss", "Unsupported loss function"),
        ],
    )
    def test_invalid_training_config_raises_error(self, param, value, expected_match):
        """Test that invalid configurations raise ValueError."""
        base_config_params = {
            "model_architecture": "unet",
            "input_channels": 3,
            "output_channels": 3,
            "learning_rate": 1e-4,
            "batch_size": 4,
            "crop_size": 128,
            "total_steps": 1000,
            "validation_interval": 100,
            "loss_function": "mse",
            "device": "cpu",
        }
        base_config_params[param] = value
        with pytest.raises(ValueError, match=expected_match):
            TrainingConfig(**base_config_params)

    def test_ms_ssim_crop_size_constraint(self):
        """Test MS-SSIM specific crop size validation (domain expertise)."""
        # Invalid crop size for MS-SSIM
        with pytest.raises(ValueError, match="MS-SSIM requires crop_size > 160"):
            TrainingConfig(
                model_architecture="unet", input_channels=3, output_channels=3,
                learning_rate=1e-4, batch_size=1, crop_size=160,
                total_steps=1, validation_interval=1,
                loss_function="ms_ssim", device="cpu"
            )
        
        # Valid crop size for MS-SSIM
        config = TrainingConfig(
            model_architecture="unet", input_channels=3, output_channels=3,
            learning_rate=1e-4, batch_size=1, crop_size=192,
            total_steps=1, validation_interval=1,
            loss_function="ms_ssim", device="cpu"
        )
        assert config.is_valid()
    
    def test_create_training_config_from_yaml(self, tmp_path):
        """Test creating TrainingConfig from a YAML file."""
        yaml_content = """
        arch: utnet3
        in_channels: 4
        out_channels: 3
        init_lr: 5e-5
        batch_size: 2
        crop_size: 256
        tot_steps: 5000
        val_interval: 250
        loss: ms_ssim
        device: cuda
        patience: 2000
        lr_multiplier: 0.2
        filter_units: 32
        """
        yaml_path = tmp_path / "test_config.yaml"
        yaml_path.write_text(yaml_content)

        config = create_training_config_from_yaml(str(yaml_path))

        assert config.model_architecture == "utnet3"
        assert config.input_channels == 4
        assert config.output_channels == 3
        assert config.learning_rate == 5e-5
        assert config.batch_size == 2
        assert config.crop_size == 256
        assert config.total_steps == 5000
        assert config.validation_interval == 250
        assert config.loss_function == "ms_ssim"
        assert config.device == "cuda"
        assert config.patience == 2000
        assert config.lr_decay_factor == 0.2
        assert config.filter_units == 32
        assert config.is_valid()

    def test_create_training_config_from_yaml_with_overrides(self, tmp_path):
        """Test creating TrainingConfig from YAML with individual overrides."""
        yaml_content = """
        arch: unet
        in_channels: 3
        out_channels: 3
        init_lr: 1e-4
        batch_size: 4
        crop_size: 128
        tot_steps: 1000
        val_interval: 100
        loss: mse
        device: cpu
        """
        yaml_path = tmp_path / "override_config.yaml"
        yaml_path.write_text(yaml_content)

        # Override learning_rate and crop_size
        config = create_training_config_from_yaml(
            str(yaml_path), learning_rate=1e-3, crop_size=256, model_architecture="autoencoder"
        )

        assert config.model_architecture == "autoencoder" # Overridden
        assert config.learning_rate == 1e-3 # Overridden
        assert config.crop_size == 256     # Overridden
        assert config.batch_size == 4      # Original from YAML
        assert config.is_valid()