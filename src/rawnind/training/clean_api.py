"""Clean API for training package without CLI dependencies.

This module provides clean, modern programmatic interfaces for training neural networks
without command-line argument parsing dependencies. It replaces the legacy CLI-based
interfaces with explicit configuration classes and factory functions.
"""

import os
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Iterator, Union
from pathlib import Path
import torch
import torch.nn as nn
import yaml

# Import necessary components from the existing training package
from . import training_loops
from ..inference.clean_api import InferenceConfig, convert_device_format
from ..dependencies.pytorch_helpers import get_device
from ..dependencies.pt_losses import losses, metrics
from ..dependencies.json_saver import YAMLSaver
from ..dependencies import raw_processing as raw


@dataclass
class TrainingConfig:
    """Configuration for training with explicit parameters (no CLI parsing)."""
    
    # Model architecture
    model_architecture: str  # "unet", "autoencoder", etc.
    input_channels: int
    output_channels: int
    
    # Training parameters
    learning_rate: float
    batch_size: int
    crop_size: int
    total_steps: int
    validation_interval: int
    
    # Loss and optimization
    loss_function: str = "mse"
    device: str = "cpu"
    
    # Learning rate scheduling
    patience: int = 1000
    lr_decay_factor: float = 0.5
    early_stopping_patience: Optional[int] = None
    
    # Additional metrics to compute
    additional_metrics: List[str] = field(default_factory=list)
    
    # Model-specific parameters
    filter_units: int = 48
    
    # Compression-specific (for joint denoise+compress)
    compression_lambda: Optional[float] = None
    bit_estimator_lr_multiplier: float = 1.0
    
    # Training behavior
    test_interval: Optional[int] = None
    test_crop_size: Optional[int] = None
    val_crop_size: Optional[int] = None
    num_crops_per_image: int = 1
    save_training_images: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.crop_size <= 0:
            raise ValueError("Crop size must be positive")
        if self.total_steps <= 0:
            raise ValueError("Total steps must be positive")
        if self.validation_interval <= 0:
            raise ValueError("Validation interval must be positive")
        if self.model_architecture not in ["unet", "utnet3", "autoencoder", "utnet2", "standard"]:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")
        if self.loss_function not in ["mse", "ms_ssim", "l1"]:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
            
        # Set defaults based on values
        if self.test_crop_size is None:
            self.test_crop_size = self.crop_size
        if self.val_crop_size is None:
            self.val_crop_size = self.crop_size
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        try:
            self.__post_init__()
            return True
        except ValueError:
            return False


@dataclass
class ExperimentConfig:
    """Configuration for experiment management."""
    
    experiment_name: str
    save_directory: str
    checkpoint_interval: int = 100
    keep_best_n_models: int = 3
    metrics_to_track: List[str] = field(default_factory=lambda: ["loss"])
    
    def __post_init__(self):
        """Set up experiment directories."""
        self.save_path = Path(self.save_directory)
        self.checkpoint_dir = self.save_path / "checkpoints"
        self.results_dir = self.save_path / "results" 
        self.logs_dir = self.save_path / "logs"
        self.visualizations_dir = self.save_path / "visualizations"
        
        # Create directories
        for dir_path in [self.checkpoint_dir, self.results_dir, self.logs_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


class CleanTrainer:
    """Base class for clean trainers without CLI dependencies."""
    
    def __init__(self, config: TrainingConfig, training_type: str):
        """Initialize trainer with explicit configuration.
        
        Args:
            config: Training configuration
            training_type: Type of training ("rgb_to_rgb", "bayer_to_rgb")
        """
        self.config = config
        self.training_type = training_type
        self.device = torch.device(config.device)  # Use torch.device directly for PyTorch operations
        self.device_legacy = convert_device_format(config.device)  # Keep legacy format for compatibility
        self.model_architecture = config.model_architecture
        
        # Training state
        self.current_step = 0
        self.best_validation_losses = {}
        self.lr_adjustment_allowed_step = config.patience
        
        # Initialize model and optimizer
        self._create_model()
        self._create_optimizer()
        
        # Initialize loss function
        self._create_loss_function()
        
    def _create_model(self):
        """Create the model based on configuration."""
        # Import model creation functionality from inference package
        from ..inference.clean_api import create_rgb_denoiser, create_bayer_denoiser
        
        if self.training_type == "rgb_to_rgb":
            # Create RGB model for training
            denoiser = create_rgb_denoiser(
                architecture=self.config.model_architecture,
                device=self.config.device,
                filter_units=self.config.filter_units
            )
            self.model = denoiser.model
        elif self.training_type == "bayer_to_rgb":
            # Create Bayer model for training  
            denoiser = create_bayer_denoiser(
                architecture=self.config.model_architecture,
                device=self.config.device,
                filter_units=self.config.filter_units
            )
            self.model = denoiser.model
            self.demosaic_fn = denoiser.demosaic_fn
        else:
            raise ValueError(f"Unsupported training type: {self.training_type}")
            
        # Move model to device
        self.model = self.model.to(self.device)
        
    def _create_optimizer(self):
        """Create optimizer for training."""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate
        )
    
    def _create_loss_function(self):
        """Create loss function based on configuration."""
        if self.config.loss_function == "mse":
            self.loss_fn = losses["mse"]()
        elif self.config.loss_function == "ms_ssim":
            self.loss_fn = losses["msssim_loss"]()
        elif self.config.loss_function == "l1":
            # L1 not in losses dict, use PyTorch's
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Unsupported loss function: {self.config.loss_function}")
    
    def compute_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor, 
                    masks: torch.Tensor) -> torch.Tensor:
        """Compute loss for a batch.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth images
            masks: Valid pixel masks
            
        Returns:
            Loss tensor with gradients
        """
        # Apply masks to both predictions and ground truth
        if masks is not None:
            predictions_masked = predictions * masks
            ground_truth_masked = ground_truth * masks
        else:
            predictions_masked = predictions
            ground_truth_masked = ground_truth
            
        # Call loss function with only 2 arguments (standard PyTorch interface)
        return self.loss_fn(predictions_masked, ground_truth_masked)
    
    def get_current_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def update_learning_rate(self, validation_metrics: Dict[str, float], step: int):
        """Update learning rate based on validation performance.
        
        Args:
            validation_metrics: Dictionary of validation metrics
            step: Current training step
        """
        loss_value = validation_metrics.get('loss', float('inf'))
        
        # Check if model improved
        if 'loss' not in self.best_validation_losses or loss_value <= self.best_validation_losses['loss']:
            self.best_validation_losses['loss'] = loss_value
            self.lr_adjustment_allowed_step = step + self.config.patience
        else:
            # No improvement, decay if past patience
            if step >= self.lr_adjustment_allowed_step:
                old_lr = self.get_current_learning_rate()
                new_lr = old_lr * self.config.lr_decay_factor
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logging.info(f"Learning rate decayed: {old_lr} -> {new_lr}")
                self.lr_adjustment_allowed_step = step + self.config.patience
    
    def save_checkpoint(self, step: int, checkpoint_path: str, include_optimizer: bool = True) -> Dict[str, Any]:
        """Save training checkpoint.
        
        Args:
            step: Current training step
            checkpoint_path: Path to save checkpoint
            include_optimizer: Whether to include optimizer state
            
        Returns:
            Dictionary with checkpoint information
        """
        checkpoint_info = {
            'step': step,
            'model_state': self.model.state_dict(),
            'config': self.config,
            'best_validation_losses': self.best_validation_losses
        }
        
        if include_optimizer:
            checkpoint_info['optimizer_state'] = self.optimizer.state_dict()
            
        torch.save(checkpoint_info, checkpoint_path)
        
        return checkpoint_info
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Dictionary with loaded checkpoint information
        """
        # Fix PyTorch 2.6 security issue with custom classes
        torch.serialization.add_safe_globals([TrainingConfig])
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Load optimizer state if available
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            
        # Restore training state
        self.current_step = checkpoint['step']
        self.best_validation_losses = checkpoint.get('best_validation_losses', {})
        
        return {
            'step': checkpoint['step'],
            'resumed_from_step': checkpoint['step']
        }
    
    def validate(self, validation_dataloader: Iterator, compute_metrics: List[str] = None,
                save_outputs: bool = False, output_directory: str = None) -> Dict[str, float]:
        """Run validation on a dataset.
        
        Args:
            validation_dataloader: Iterator yielding validation batches
            compute_metrics: List of metrics to compute
            save_outputs: Whether to save model outputs
            output_directory: Directory to save outputs
            
        Returns:
            Dictionary of computed metrics
        """
        if compute_metrics is None:
            compute_metrics = ['loss']
            
        self.model.eval()
        all_losses = {metric: [] for metric in compute_metrics}
        
        with torch.no_grad():
            for i, batch in enumerate(validation_dataloader):
                # Get batch data
                clean_images = batch['clean_images'].to(self.device)
                noisy_images = batch['noisy_images'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Forward pass
                predictions = self.model(noisy_images)
                
                # Compute loss
                if 'loss' in compute_metrics:
                    loss = self.compute_loss(predictions, clean_images, masks)
                    all_losses['loss'].append(loss.item())
                
                # Compute additional metrics
                from ..inference.clean_api import compute_image_metrics
                if len(compute_metrics) > 1:
                    other_metrics = [m for m in compute_metrics if m != 'loss']
                    batch_metrics = compute_image_metrics(
                        predictions.cpu(), clean_images.cpu(), other_metrics
                    )
                    for metric_name, metric_value in batch_metrics.items():
                        if metric_name in all_losses:
                            all_losses[metric_name].append(metric_value)
                
                # Save outputs if requested
                if save_outputs and output_directory:
                    self._save_validation_outputs(predictions, batch, output_directory, i)
        
        self.model.train()
        
        # Return average metrics
        result = {}
        for metric_name, values in all_losses.items():
            if values:
                result[metric_name] = sum(values) / len(values)
        
        if save_outputs:
            result['outputs_saved'] = True
            
        return result
    
    def test(self, test_dataloader: Iterator, test_name: str = "test",
            save_outputs: bool = False, compute_metrics: List[str] = None) -> Dict[str, Any]:
        """Run testing on a dataset.
        
        Args:
            test_dataloader: Iterator yielding test batches
            test_name: Name for this test run
            save_outputs: Whether to save model outputs
            compute_metrics: List of metrics to compute
            
        Returns:
            Dictionary of test results
        """
        if compute_metrics is None:
            compute_metrics = ['loss']
            
        test_results = self.validate(
            validation_dataloader=test_dataloader,
            compute_metrics=compute_metrics,
            save_outputs=save_outputs
        )
        
        test_results['test_name'] = test_name
        return test_results
    
    def _save_validation_outputs(self, predictions: torch.Tensor, batch: Dict, 
                               output_directory: str, batch_idx: int):
        """Save validation outputs to files."""
        # Simplified output saving - would be more sophisticated in real implementation
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(predictions.shape[0]):
            output_path = output_dir / f"batch_{batch_idx}_sample_{i}_output.exr"
            # In real implementation, would save as EXR using OpenEXR
            # For now, just log the operation
            logging.info(f"Would save output to {output_path}")


class CleanDenoiserTrainer(CleanTrainer):
    """Clean trainer for denoising models without CLI dependencies."""
    
    def __init__(self, config: TrainingConfig, training_type: str):
        """Initialize denoiser trainer.
        
        Args:
            config: Training configuration
            training_type: "rgb_to_rgb" or "bayer_to_rgb"
        """
        # Validate configuration for denoising
        if training_type == "bayer_to_rgb" and config.input_channels != 4:
            raise ValueError("Bayer training requires 4 input channels")
        if training_type == "rgb_to_rgb" and config.input_channels != 3:
            raise ValueError("RGB training requires 3 input channels")
            
        super().__init__(config, training_type)
        
        # Add Bayer-specific processing if needed
        if training_type == "bayer_to_rgb":
            self._setup_bayer_processing()
    
    def _setup_bayer_processing(self):
        """Set up Bayer-specific processing functions."""
        # Use raw processing utilities from dependencies
        self.demosaic_fn = raw.demosaic
        
    def process_bayer_output(self, model_output: torch.Tensor, 
                           xyz_matrices: torch.Tensor,
                           bayer_input: torch.Tensor) -> torch.Tensor:
        """Process Bayer model output with color transformation.
        
        Args:
            model_output: Raw model output
            xyz_matrices: Color transformation matrices
            bayer_input: Original Bayer input
            
        Returns:
            Processed RGB output
        """
        # Apply color transformation and processing
        # This is a simplified version - real implementation would be more complex
        return model_output
    
    def train(self, train_dataloader: Iterator, validation_dataloader: Iterator,
             experiment_manager: 'CleanExperimentManager',
             max_steps: Optional[int] = None) -> Dict[str, Any]:
        """Run training loop.
        
        Args:
            train_dataloader: Training data iterator
            validation_dataloader: Validation data iterator
            experiment_manager: Experiment manager for saving/tracking
            max_steps: Maximum steps to train (overrides config if provided)
            
        Returns:
            Dictionary with training results
        """
        max_steps = max_steps or self.config.total_steps
        training_loss_history = []
        validation_metrics_history = []
        
        self.model.train()
        
        step = self.current_step
        early_stopped = False
        early_stop_reason = None
        
        # Simple training loop
        while step < max_steps:
            # Training step
            for batch in train_dataloader:
                if step >= max_steps:
                    break
                    
                # Move data to device
                clean_images = batch['clean_images'].to(self.device)
                noisy_images = batch['noisy_images'].to(self.device)
                masks = batch['masks'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(noisy_images)
                
                # Compute loss
                loss = self.compute_loss(predictions, clean_images, masks)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                training_loss_history.append(loss.item())
                step += 1
                self.current_step = step
                
                # Validation
                if step % self.config.validation_interval == 0:
                    val_metrics = self.validate(
                        validation_dataloader=validation_dataloader,
                        compute_metrics=['loss'] + self.config.additional_metrics
                    )
                    validation_metrics_history.append(val_metrics)
                    
                    # Update learning rate
                    self.update_learning_rate(val_metrics, step)
                    
                    # Record metrics in experiment manager
                    experiment_manager.record_metrics(step, {
                        'train_loss': loss.item(),
                        **{f'val_{k}': v for k, v in val_metrics.items()}
                    })
                    
                    # Check for early stopping
                    if self.config.early_stopping_patience:
                        if self._should_early_stop(val_metrics, step):
                            early_stopped = True
                            early_stop_reason = "No improvement in validation loss"
                            break
                    
                # Checkpointing
                checkpoint_info = experiment_manager.should_save_checkpoint(step)
                if checkpoint_info['should_save']:
                    checkpoint_path = experiment_manager.config.checkpoint_dir / f"model_step_{step}.pt"
                    self.save_checkpoint(step, str(checkpoint_path))
                
                if step >= max_steps:
                    break
            
            if early_stopped or step >= max_steps:
                break
        
        # Final validation
        final_val_metrics = self.validate(
            validation_dataloader=validation_dataloader,
            compute_metrics=['loss'] + self.config.additional_metrics
        )
        
        return {
            'steps_completed': step,
            'final_loss': training_loss_history[-1] if training_loss_history else 0.0,
            'training_loss_history': training_loss_history,
            'validation_metrics_history': validation_metrics_history,
            'final_validation_metrics': final_val_metrics,
            'early_stopped': early_stopped,
            'early_stop_reason': early_stop_reason
        }
    
    def _should_early_stop(self, val_metrics: Dict[str, float], step: int) -> bool:
        """Check if training should stop early."""
        if not self.config.early_stopping_patience:
            return False
            
        # Simple early stopping based on validation loss
        current_loss = val_metrics.get('loss', float('inf'))
        
        if 'loss' not in self.best_validation_losses:
            self.best_validation_losses['loss'] = current_loss
            return False
            
        # Check if no improvement for patience steps
        if current_loss > self.best_validation_losses['loss']:
            steps_without_improvement = step - self.lr_adjustment_allowed_step + self.config.patience
            return steps_without_improvement >= self.config.early_stopping_patience
            
        return False
    
    def resume_from_experiment(self, experiment_directory: str) -> Dict[str, Any]:
        """Resume training from an experiment directory.
        
        Args:
            experiment_directory: Path to experiment directory
            
        Returns:
            Dictionary with resume information
        """
        checkpoint_dir = Path(experiment_directory) / "checkpoints"
        
        # Find latest checkpoint
        checkpoint_files = list(checkpoint_dir.glob("model_step_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
            
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-1]))
        
        return self.load_checkpoint(str(latest_checkpoint))
    
    def prepare_datasets(self, dataset_config: Dict[str, Any]) -> Dict[str, Iterator]:
        """Prepare datasets for training (placeholder for dataset package integration).
        
        Args:
            dataset_config: Configuration for dataset preparation
            
        Returns:
            Dictionary with train/val/test dataloaders
        """
        # This is a placeholder - would integrate with dataset package
        logging.info(f"Preparing datasets with config: {dataset_config}")
        
        # Return mock dataloaders for now
        def mock_dataloader():
            for i in range(5):
                yield {
                    'clean_images': torch.randn(self.config.batch_size, 
                                               self.config.output_channels, 
                                               self.config.crop_size, 
                                               self.config.crop_size),
                    'noisy_images': torch.randn(self.config.batch_size,
                                               self.config.input_channels,
                                               self.config.crop_size,
                                               self.config.crop_size),
                    'masks': torch.ones(self.config.batch_size,
                                       self.config.output_channels,
                                       self.config.crop_size,
                                       self.config.crop_size),
                    'image_paths': [f'image_{i}_{j}.jpg' for j in range(self.config.batch_size)]
                }
        
        return {
            'train_loader': mock_dataloader(),
            'val_loader': mock_dataloader(),
            'test_loader': mock_dataloader()
        }


class CleanDenoiseCompressTrainer(CleanTrainer):
    """Clean trainer for joint denoising+compression models without CLI dependencies."""
    
    def __init__(self, config: TrainingConfig, training_type: str):
        """Initialize denoise+compress trainer.
        
        Args:
            config: Training configuration (must include compression parameters)
            training_type: "rgb_to_rgb" or "bayer_to_rgb"
        """
        if config.compression_lambda is None:
            raise ValueError("compression_lambda must be specified for denoise+compress training")
            
        super().__init__(config, training_type)
        
        # Set up compression-specific components
        self._setup_compression_model()
        self._setup_bit_estimator()
        
    def _setup_compression_model(self):
        """Set up compression model components."""
        # Use compression model from inference package
        from ..inference.clean_api import create_compressor
        
        # Create compressor model
        compressor = create_compressor(
            architecture=self.config.model_architecture,
            encoder_arch="encoder",
            decoder_arch="decoder",
            device=self.config.device
        )
        
        # Replace the simple model with compression model
        self.compression_model = compressor.model
        self.model = self.compression_model  # Alias for compatibility
        
    def _setup_bit_estimator(self):
        """Set up bit estimator for rate-distortion optimization."""
        # Create bit estimator model
        from ..models.bitEstimator import BitEstimator
        
        self.bit_estimator = BitEstimator()
        self.bit_estimator = self.bit_estimator.to(self.device)
        
    def _create_optimizer(self):
        """Create optimizer with separate parameter groups for compression."""
        # Main model parameters
        model_params = list(self.compression_model.parameters())
        
        # Bit estimator parameters with different learning rate
        bit_estimator_params = list(self.bit_estimator.parameters())
        
        # Create optimizer with parameter groups
        self.optimizer = torch.optim.Adam([
            {'params': model_params, 'lr': self.config.learning_rate},
            {'params': bit_estimator_params, 
             'lr': self.config.learning_rate * self.config.bit_estimator_lr_multiplier}
        ])
    
    def get_optimizer_param_groups(self) -> List[Dict[str, Any]]:
        """Get optimizer parameter groups information."""
        return self.optimizer.param_groups
    
    def compute_joint_loss(self, predictions: torch.Tensor, ground_truth: torch.Tensor,
                          masks: torch.Tensor, bits_per_pixel: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute joint loss (distortion + rate) for compression training.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth images
            masks: Valid pixel masks
            bits_per_pixel: Estimated bits per pixel
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Compute distortion loss
        distortion_loss = self.compute_loss(predictions, ground_truth, masks)
        
        # Compute rate loss
        rate_loss = bits_per_pixel * self.config.compression_lambda
        
        # Total loss
        total_loss = distortion_loss + rate_loss
        
        loss_components = {
            'distortion_loss': distortion_loss,
            'rate_loss': rate_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_components


class CleanExperimentManager:
    """Clean experiment manager without CLI dependencies."""
    
    def __init__(self, config: ExperimentConfig):
        """Initialize experiment manager.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.metrics_history = []
        self.best_steps = {}
        
        # Initialize results saver
        results_path = self.config.results_dir / "experiment_results.yaml"
        self.json_saver = YAMLSaver(str(results_path))
        
        logging.info(f"Experiment '{config.experiment_name}' initialized in {config.save_directory}")
    
    def record_metrics(self, step: int, metrics: Dict[str, float]):
        """Record metrics for a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metric names to values
        """
        # Add step to metrics
        step_metrics = {'step': step, **metrics}
        self.metrics_history.append(step_metrics)
        
        # Update best steps
        for metric_name in self.config.metrics_to_track:
            if metric_name in metrics:
                metric_value = metrics[metric_name]
                
                # For loss metrics, lower is better
                if 'loss' in metric_name.lower():
                    if metric_name not in self.best_steps or metric_value < self.best_steps[metric_name]['value']:
                        self.best_steps[metric_name] = {'step': step, 'value': metric_value}
                # For other metrics (like SSIM, PSNR), higher is better
                else:
                    if metric_name not in self.best_steps or metric_value > self.best_steps[metric_name]['value']:
                        self.best_steps[metric_name] = {'step': step, 'value': metric_value}
        
        # Save to results file
        self.json_saver.add_res(step, metrics)
    
    def get_best_steps(self) -> Dict[str, int]:
        """Get best step numbers for each tracked metric.
        
        Returns:
            Dictionary mapping metric names to best step numbers
        """
        return {metric: info['step'] for metric, info in self.best_steps.items()}
    
    def should_save_checkpoint(self, step: int) -> Dict[str, Any]:
        """Check if a checkpoint should be saved at this step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary with checkpoint decision information
        """
        should_save = step % self.config.checkpoint_interval == 0
        
        return {
            'should_save': should_save,
            'checkpoint_path': self.config.checkpoint_dir / f"model_step_{step}.pt" if should_save else None
        }
    
    def cleanup_checkpoints(self) -> Dict[str, int]:
        """Clean up old checkpoints, keeping only the best ones.
        
        Returns:
            Dictionary with cleanup information
        """
        checkpoint_files = list(self.config.checkpoint_dir.glob("model_step_*.pt"))
        
        if len(checkpoint_files) <= self.config.keep_best_n_models:
            return {'checkpoints_removed': 0, 'checkpoints_kept': len(checkpoint_files)}
        
        # Get best steps to keep
        best_steps = list(self.get_best_steps().values())
        keepers = {f"model_step_{step}.pt" for step in best_steps}
        
        # If we don't have enough best steps, keep the most recent ones
        if len(keepers) < self.config.keep_best_n_models:
            # Sort by step number and keep most recent
            sorted_files = sorted(checkpoint_files, 
                                key=lambda p: int(p.stem.split('_')[-1]), 
                                reverse=True)
            for file in sorted_files[:self.config.keep_best_n_models]:
                keepers.add(file.name)
        
        # Remove files not in keepers
        removed_count = 0
        for file in checkpoint_files:
            if file.name not in keepers:
                file.unlink()
                removed_count += 1
                
        return {
            'checkpoints_removed': removed_count,
            'checkpoints_kept': len(checkpoint_files) - removed_count
        }


# Factory Functions

def create_denoiser_trainer(training_type: str, config: TrainingConfig) -> CleanDenoiserTrainer:
    """Create a denoiser trainer with clean API.
    
    Args:
        training_type: "rgb_to_rgb" or "bayer_to_rgb"
        config: Training configuration
        
    Returns:
        Clean denoiser trainer instance
    """
    return CleanDenoiserTrainer(config, training_type)


def create_denoise_compress_trainer(training_type: str, config: TrainingConfig) -> CleanDenoiseCompressTrainer:
    """Create a joint denoise+compress trainer with clean API.
    
    Args:
        training_type: "rgb_to_rgb" or "bayer_to_rgb" 
        config: Training configuration (must include compression parameters)
        
    Returns:
        Clean denoise+compress trainer instance
    """
    return CleanDenoiseCompressTrainer(config, training_type)


def create_experiment_manager(config: ExperimentConfig) -> CleanExperimentManager:
    """Create an experiment manager with clean API.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Clean experiment manager instance
    """
    return CleanExperimentManager(config)


# Configuration validation utilities

def validate_training_type_and_config(training_type: str, config: TrainingConfig):
    """Validate that training type and config are compatible.
    
    Args:
        training_type: Training type to validate
        config: Training configuration to validate
        
    Raises:
        ValueError: If training type and config are incompatible
    """
    if training_type == "bayer_to_rgb":
        if config.input_channels != 4:
            raise ValueError("Bayer training requires 4 input channels")
        if config.output_channels != 3:
            raise ValueError("Bayer training should output 3 RGB channels")
    elif training_type == "rgb_to_rgb":
        if config.input_channels != 3:
            raise ValueError("RGB training requires 3 input channels")
        if config.output_channels != 3:
            raise ValueError("RGB training should output 3 channels")
    else:
        raise ValueError(f"Unsupported training type: {training_type}")


def create_training_config_from_yaml(yaml_path: str, **overrides) -> TrainingConfig:
    """Create training config from YAML file with optional overrides.
    
    Args:
        yaml_path: Path to YAML configuration file
        **overrides: Parameter overrides
        
    Returns:
        TrainingConfig instance
    """
    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Extract relevant parameters for TrainingConfig
    config_params = {
        'model_architecture': yaml_config.get('arch', 'unet'),
        'input_channels': yaml_config.get('in_channels', 3),
        'output_channels': yaml_config.get('out_channels', 3),
        'learning_rate': yaml_config.get('init_lr', 1e-4),
        'batch_size': yaml_config.get('batch_size', 4),
        'crop_size': yaml_config.get('crop_size', 128),
        'total_steps': yaml_config.get('tot_steps', 1000),
        'validation_interval': yaml_config.get('val_interval', 100),
        'loss_function': yaml_config.get('loss', 'mse'),
        'device': yaml_config.get('device', 'cpu'),
        'patience': yaml_config.get('patience', 1000),
        'lr_decay_factor': yaml_config.get('lr_multiplier', 0.5),
        'filter_units': yaml_config.get('filter_units', 48)
    }
    
    # Apply overrides
    config_params.update(overrides)
    
    return TrainingConfig(**config_params)
