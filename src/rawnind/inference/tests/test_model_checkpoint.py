import pytest
import os
import yaml
from pathlib import Path

from rawnind.inference.clean_api import ModelCheckpoint

@pytest.fixture
def mock_model_dir(tmp_path):
    """Create a mock model directory structure for testing."""
    model_dir = tmp_path / "mock_experiment"
    (model_dir / "saved_models").mkdir(parents=True)
    
    # Create args.yaml
    args_content = {
        "arch": "unet",
        "in_channels": 3,
        "funit": 48,
        "loss": "mse",
        "match_gain": "output"
    }
    with open(model_dir / "args.yaml", "w") as f:
        yaml.safe_dump(args_content, f)

    # Create trainres.yaml
    trainres_content = {
        "best_step": {
            "val_mse": 100,
            "val_psnr": 150,
            "val_ms_ssim": 200,
        },
        "100": {"val_mse": 0.001, "val_psnr": 35.0},
        "150": {"val_psnr": 38.0, "val_mse": 0.0008},
        "200": {"val_ms_ssim": 0.98, "val_mse": 0.0005},
        "metrics_history": [
            {"step": 100, "val_mse": 0.001, "val_psnr": 35.0, "val_ms_ssim": 0.9}
        ]
    }
    with open(model_dir / "trainres.yaml", "w") as f:
        yaml.safe_dump(trainres_content, f)
    
    # Create dummy checkpoint files
    (model_dir / "saved_models" / "iter_50.pt").touch()
    (model_dir / "saved_models" / "iter_100.pt").touch()
    (model_dir / "saved_models" / "iter_150.pt").touch()
    (model_dir / "saved_models" / "iter_200.pt").touch()

    return model_dir

class TestModelCheckpoint:
    """Unit tests for the ModelCheckpoint dataclass."""

    def test_from_directory_valid_metric(self, mock_model_dir):
        """Test loading checkpoint info from directory with a valid metric."""
        checkpoint = ModelCheckpoint.from_directory(str(mock_model_dir), metric_name="val_ms_ssim")

        assert checkpoint.step_number == 200
        assert checkpoint.checkpoint_path == str(mock_model_dir / "saved_models" / "iter_200.pt")
        assert checkpoint.model_config["arch"] == "unet"
        assert checkpoint.metrics == {"val_ms_ssim": 0.98, "val_mse": 0.0005}

    def test_from_directory_default_metric(self, mock_model_dir):
        """Test loading checkpoint info with default metric (val_msssim)."""
        # trainres.yaml has best_step for val_msssim at 200
        checkpoint = ModelCheckpoint.from_directory(str(mock_model_dir)) 
        assert checkpoint.step_number == 200
        assert checkpoint.checkpoint_path == str(mock_model_dir / "saved_models" / "iter_200.pt")


    def test_from_directory_metric_not_found(self, mock_model_dir):
        """Test loading with a metric not found in trainres.yaml."""
        with pytest.raises(KeyError, match="Metric val_unknown not found in training results"):
            ModelCheckpoint.from_directory(str(mock_model_dir), metric_name="val_unknown")

    def test_from_directory_missing_trainres(self, tmp_path):
        """Test loading fails if trainres.yaml is missing."""
        model_dir = tmp_path / "missing_trainres_exp"
        model_dir.mkdir()
        (model_dir / "saved_models").mkdir() # Need saved_models dir
        with pytest.raises(FileNotFoundError, match="Training results not found"):
            ModelCheckpoint.from_directory(str(model_dir))

    def test_from_directory_missing_args(self, mock_model_dir):
        """Test loading still works if args.yaml is missing, but model_config is empty."""
        os.remove(mock_model_dir / "args.yaml")
        checkpoint = ModelCheckpoint.from_directory(str(mock_model_dir), metric_name="val_mse")
        assert checkpoint.model_config == {} # Should be empty dict

    def test_from_directory_missing_checkpoint_file(self, mock_model_dir):
        """Test loading fails if the best checkpoint file is missing."""
        os.remove(mock_model_dir / "saved_models" / "iter_200.pt") # Remove the best checkpoint
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            ModelCheckpoint.from_directory(str(mock_model_dir), metric_name="val_ms_ssim")