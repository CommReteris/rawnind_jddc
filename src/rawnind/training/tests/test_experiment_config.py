import pytest
from pathlib import Path
import os
from rawnind.training.clean_api import ExperimentConfig

class TestExperimentConfig:
    """Unit tests for ExperimentConfig dataclass."""

    def test_valid_experiment_config_init(self, tmp_path):
        """Test happy path initialization of ExperimentConfig."""
        exp_name = "test_exp_001"
        save_dir = str(tmp_path / "experiments")
        
        config = ExperimentConfig(
            experiment_name=exp_name,
            save_directory=save_dir,
            checkpoint_interval=50,
            keep_best_n_models=5,
            metrics_to_track=["loss", "val_psnr"]
        )

        assert config.experiment_name == exp_name
        assert config.save_directory == save_dir
        assert config.checkpoint_interval == 50
        assert config.keep_best_n_models == 5
        assert config.metrics_to_track == ["loss", "val_psnr"]
        
        # Verify derived paths and directory creation
        assert config.save_path == Path(save_dir)
        assert config.checkpoint_dir == Path(save_dir) / "checkpoints"
        assert config.results_dir == Path(save_dir) / "results"
        assert config.logs_dir == Path(save_dir) / "logs"
        assert config.visualizations_dir == Path(save_dir) / "visualizations"

        # Check if directories were created
        assert config.checkpoint_dir.is_dir()
        assert config.results_dir.is_dir()
        assert config.logs_dir.is_dir()
        assert config.visualizations_dir.is_dir()

    def test_default_values(self, tmp_path):
        """Test default values for optional parameters."""
        save_dir = str(tmp_path / "default_exp")
        config = ExperimentConfig(
            experiment_name="default_exp",
            save_directory=save_dir
        )
        assert config.checkpoint_interval == 100
        assert config.keep_best_n_models == 3
        assert config.metrics_to_track == ["loss"] # Default metric is "loss"

        # Check if directories were created
        assert (Path(save_dir) / "checkpoints").is_dir()
        assert (Path(save_dir) / "results").is_dir()

    def test_directory_creation_on_init(self, tmp_path):
        """Test that directories are created upon initialization."""
        base_path = tmp_path / "new_experiment"
        assert not base_path.exists()
        
        config = ExperimentConfig(
            experiment_name="init_test",
            save_directory=str(base_path)
        )
        
        assert base_path.is_dir()
        assert (base_path / "checkpoints").is_dir()
        assert (base_path / "results").is_dir()
        assert (base_path / "logs").is_dir()
        assert (base_path / "visualizations").is_dir()