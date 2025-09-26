import pytest
import torch
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

from rawnind.training.clean_api import ExperimentConfig, CleanExperimentManager
from rawnind.dependencies.json_saver import YAMLSaver

@pytest.fixture
def base_experiment_config(tmp_path):
    return ExperimentConfig(
        experiment_name="test_experiment",
        save_directory=str(tmp_path / "exp_dir"),
        checkpoint_interval=5,
        keep_best_n_models=2,
        metrics_to_track=["val_loss", "val_psnr"] # Track both loss (lower is better) and psnr (higher is better)
    )

class TestCleanExperimentManager:
    """Unit tests for the CleanExperimentManager class."""

    def test_init(self, base_experiment_config):
        """Test initialization of CleanExperimentManager."""
        manager = CleanExperimentManager(config=base_experiment_config)

        assert manager.config == base_experiment_config
        assert manager.metrics_history == []
        assert manager.best_steps == {}
        assert isinstance(manager.json_saver, YAMLSaver)
        assert manager.config.checkpoint_dir.is_dir()
        assert manager.config.results_dir.is_dir()

    def test_record_metrics_loss_improvement(self, base_experiment_config):
        """Test recording metrics and updating best_steps for loss (lower is better)."""
        manager = CleanExperimentManager(config=base_experiment_config)

        # Step 1: Initial record
        manager.record_metrics(step=1, metrics={'val_loss': 0.5, 'val_psnr': 20.0, 'train_loss': 0.6})
        assert manager.metrics_history[0]['step'] == 1
        assert manager.best_steps['val_loss'] == {'step': 1, 'value': 0.5}
        assert manager.best_steps['val_psnr'] == {'step': 1, 'value': 20.0}

        # Step 2: Loss improves, PSNR improves
        manager.record_metrics(step=2, metrics={'val_loss': 0.4, 'val_psnr': 22.0, 'train_loss': 0.5})
        assert manager.best_steps['val_loss'] == {'step': 2, 'value': 0.4}
        assert manager.best_steps['val_psnr'] == {'step': 2, 'value': 22.0}

        # Step 3: Loss does not improve, PSNR does not improve
        manager.record_metrics(step=3, metrics={'val_loss': 0.45, 'val_psnr': 21.0, 'train_loss': 0.4})
        assert manager.best_steps['val_loss'] == {'step': 2, 'value': 0.4} # Should remain at step 2
        assert manager.best_steps['val_psnr'] == {'step': 2, 'value': 22.0} # Should remain at step 2

    def test_record_metrics_psnr_improvement(self, tmp_path):
        """Test recording metrics and updating best_steps for PSNR (higher is better)."""
        config = ExperimentConfig(
            experiment_name="psnr_test",
            save_directory=str(tmp_path / "exp_psnr"),
            metrics_to_track=["val_psnr"]
        )
        manager = CleanExperimentManager(config=config)

        # Step 1: Initial record
        manager.record_metrics(step=1, metrics={'val_psnr': 20.0})
        assert manager.best_steps['val_psnr'] == {'step': 1, 'value': 20.0}

        # Step 2: PSNR improves
        manager.record_metrics(step=2, metrics={'val_psnr': 25.0})
        assert manager.best_steps['val_psnr'] == {'step': 2, 'value': 25.0}

        # Step 3: PSNR does not improve
        manager.record_metrics(step=3, metrics={'val_psnr': 22.0})
        assert manager.best_steps['val_psnr'] == {'step': 2, 'value': 25.0} # Should remain at step 2

    def test_get_best_steps(self, base_experiment_config):
        """Test get_best_steps returns mapping of metric to best step."""
        manager = CleanExperimentManager(config=base_experiment_config)
        manager.record_metrics(step=1, metrics={'val_loss': 0.5, 'val_psnr': 20.0})
        manager.record_metrics(step=2, metrics={'val_loss': 0.4, 'val_psnr': 22.0})
        best_steps = manager.get_best_steps()
        assert best_steps == {'val_loss': 2, 'val_psnr': 2}

    def test_should_save_checkpoint(self, base_experiment_config):
        """Test should_save_checkpoint logic."""
        config = base_experiment_config
        manager = CleanExperimentManager(config=config)

        # Should save at step 5
        result_5 = manager.should_save_checkpoint(step=5)
        assert result_5['should_save'] == True
        assert result_5['checkpoint_path'] == config.checkpoint_dir / "model_step_5.pt"

        # Should not save at step 6
        result_6 = manager.should_save_checkpoint(step=6)
        assert result_6['should_save'] == False
        assert result_6['checkpoint_path'] is None

        # Should save at step 10
        result_10 = manager.should_save_checkpoint(step=10)
        assert result_10['should_save'] == True

    @patch('pathlib.Path.unlink')
    def test_cleanup_checkpoints(self, mock_unlink, base_experiment_config):
        """Test cleanup_checkpoints logic for keeping best N models."""
        config = base_experiment_config
        manager = CleanExperimentManager(config=config)

        # Create dummy checkpoint files
        cp_dir = config.checkpoint_dir
        (cp_dir / "model_step_1.pt").touch()
        (cp_dir / "model_step_1.pt.opt").touch()
        (cp_dir / "model_step_5.pt").touch()
        (cp_dir / "model_step_5.pt.opt").touch()
        (cp_dir / "model_step_10.pt").touch()
        (cp_dir / "model_step_10.pt.opt").touch()
        (cp_dir / "model_step_15.pt").touch()
        (cp_dir / "model_step_15.pt.opt").touch()

        # Simulate best steps: step 5 for loss, step 10 for psnr
        manager.best_steps = {
            'val_loss': {'step': 5, 'value': 0.1},
            'val_psnr': {'step': 10, 'value': 30.0}
        }
        
        # Keep best N models (config.keep_best_n_models = 2)
        # Should keep step 5 and step 10.
        cleanup_stats = manager.cleanup_checkpoints()
        
        # Expecting 4 unlinks (step 1 and step 15 .pt and .opt files)
        assert mock_unlink.call_count == 4
        assert cleanup_stats['checkpoints_removed'] == 4
        assert cleanup_stats['checkpoints_kept'] == 4 # 2 .pt and 2 .opt files

        # Verify which files remain (only step 5 and 10 related)
        remaining_files = {f.name for f in cp_dir.iterdir()}
        assert "model_step_5.pt" in remaining_files
        assert "model_step_5.pt.opt" in remaining_files
        assert "model_step_10.pt" in remaining_files
        assert "model_step_10.pt.opt" in remaining_files
        assert "model_step_1.pt" not in remaining_files
        assert "model_step_15.pt" not in remaining_files

    @patch('pathlib.Path.unlink')
    def test_cleanup_checkpoints_not_enough_best(self, mock_unlink, base_experiment_config):
        """Test cleanup keeps most recent if not enough 'best' models."""
        config = base_experiment_config
        # Set keep_best_n_models to 3 explicitly for this test
        config.keep_best_n_models = 3 
        manager = CleanExperimentManager(config=config)

        # Create dummy checkpoint files for steps 1, 5, 10, 15, 20
        cp_dir = config.checkpoint_dir
        for step in [1, 5, 10, 15, 20]:
            (cp_dir / f"model_step_{step}.pt").touch()
            (cp_dir / f"model_step_{step}.pt.opt").touch()

        # Only one best step for actual tracking
        manager.best_steps = {
            'val_loss': {'step': 10, 'value': 0.1}, 
        }

        # Should keep: step 10 (best) + 2 most recent (15, 20)
        # So, steps 1, 5 should be removed (4 files)
        cleanup_stats = manager.cleanup_checkpoints()
        
        assert mock_unlink.call_count == 4 # iter_1, iter_5 (.pt and .opt)
        assert cleanup_stats['checkpoints_removed'] == 4
        assert cleanup_stats['checkpoints_kept'] == 6 # 3 .pt and 3 .opt

        remaining_files = {f.name for f in cp_dir.iterdir()}
        assert "model_step_10.pt" in remaining_files
        assert "model_step_15.pt" in remaining_files
        assert "model_step_20.pt" in remaining_files
        assert "model_step_1.pt" not in remaining_files
        assert "model_step_5.pt" not in remaining_files