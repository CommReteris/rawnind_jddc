"""
Prototype architectures for handling the complexity gap between Legacy and Clean APIs.

Three approaches to address the under-engineered Clean API:
1. Strategy Pattern - Inject different behavior/complexity levels
2. Pure Modern Rewrite - Feature-complete Clean API with all legacy functionality  
3. Facade Pattern - Flexible connector to different backends
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
import tempfile
from pathlib import Path

from .clean_api import TrainingConfig, CleanDenoiserTrainer


# ============================================================================
# 1. STRATEGY PATTERN PROTOTYPE
# ============================================================================

@runtime_checkable
class TrainingStrategy(Protocol):
    """Protocol defining training strategy interface."""
    
    def setup_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
        """Set up experiment environment and return experiment metadata."""
        ...
    
    def create_dataloaders(self, config: TrainingConfig) -> Dict[str, Any]:
        """Create appropriate dataloaders for this strategy."""
        ...
    
    def validate_model(self, trainer, validation_data) -> Dict[str, float]:
        """Perform validation appropriate for this strategy."""
        ...


class SimpleTrainingStrategy:
    """Basic training strategy for quick prototyping and simple experiments."""
    
    def setup_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
        """Minimal experiment setup - just basic directory structure."""
        experiment_dir = Path(tempfile.mkdtemp(prefix="simple_exp_"))
        return {
            "experiment_dir": experiment_dir,
            "checkpoints_dir": experiment_dir / "checkpoints", 
            "results_file": experiment_dir / "results.yaml",
            "strategy_type": "simple"
        }
    
    def create_dataloaders(self, config: TrainingConfig) -> Dict[str, Any]:
        """Create basic mock dataloaders."""
        return {"train_loader": iter([]), "val_loader": iter([])}
    
    def validate_model(self, trainer, validation_data) -> Dict[str, float]:
        """Basic validation - just loss computation."""
        return trainer.validate(validation_data, compute_metrics=["loss"])


class ResearchTrainingStrategy:
    """Advanced strategy for research with sophisticated experiment tracking."""
    
    def setup_experiment(self, config: TrainingConfig) -> Dict[str, Any]:
        """Full experiment setup with auto-naming, versioning, locking."""
        # Auto-generate experiment name like legacy system
        exp_name = f"{config.model_architecture}_{config.input_channels}ch"
        base_dir = Path("models/rawnind") / exp_name
        
        # Handle iteration numbering like legacy
        iteration = 0
        while (base_dir.with_suffix(f"-{iteration}")).exists():
            iteration += 1
        
        experiment_dir = base_dir.with_suffix(f"-{iteration}")
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        return {
            "experiment_dir": experiment_dir,
            "experiment_name": f"{exp_name}-{iteration}",
            "checkpoints_dir": experiment_dir / "saved_models",
            "results_file": experiment_dir / "trainres.yaml", 
            "individual_results_dir": experiment_dir / "individual_results",
            "visualization_dir": experiment_dir / "visu",
            "strategy_type": "research",
            "supports_progressive_testing": True,
            "supports_validation_locking": True
        }
    
    def create_dataloaders(self, config: TrainingConfig) -> Dict[str, Any]:
        """Create sophisticated dataloaders with multi-dataset support."""
        # Implement complex dataset loading like legacy system
        return {
            "train_loader": iter([]),  # Would use multi-yaml loading
            "val_loader": iter([]),    # Would support test_reserve
            "test_loader": iter([]),   # Would handle progressive testing
            "supports_arbitrary_proc": True,
            "supports_data_pairing": True
        }
    
    def validate_model(self, trainer, validation_data) -> Dict[str, float]:
        """Research-grade validation with locking, individual results, etc."""
        # Would implement validation locking, individual result saving
        return trainer.validate(validation_data, 
                              compute_metrics=["loss", "mse", "psnr", "msssim"],
                              save_outputs=True)


class StrategicTrainer:
    """Trainer that uses injected strategy for behavior."""
    
    def __init__(self, config: TrainingConfig, strategy: TrainingStrategy):
        self.config = config
        self.strategy = strategy
        self.experiment_metadata = strategy.setup_experiment(config)
        self.dataloaders = strategy.create_dataloaders(config)
        
        # Core trainer functionality remains the same
        self.core_trainer = CleanDenoiserTrainer(config, "rgb_to_rgb")
    
    def train(self):
        """Training that adapts behavior based on strategy."""
        return self.strategy.validate_model(
            self.core_trainer, 
            self.dataloaders["val_loader"]
        )


# ============================================================================
# 2. PURE MODERN REWRITE PROTOTYPE
# ============================================================================

@dataclass
class AdvancedTrainingConfig:
    """Feature-complete modern rewrite with all legacy functionality."""
    
    # Core training (from TrainingConfig)
    model_architecture: str
    input_channels: int
    output_channels: int
    learning_rate: float
    batch_size: int
    crop_size: int
    total_steps: int
    validation_interval: int
    loss_function: str = "mse"
    device: str = "cpu"
    
    # Advanced experiment management (from legacy)
    experiment_name_pattern: str = "auto"  # "auto" generates like legacy
    experiment_base_directory: str = "models/rawnind"
    auto_increment_iterations: bool = True
    continue_from_previous: bool = False
    checkpoint_every_n_steps: int = 100
    keep_best_n_checkpoints: int = 5
    
    # Multi-dataset configuration (from legacy)
    clean_dataset_yamls: List[str] = field(default_factory=list)
    noisy_dataset_yamls: List[str] = field(default_factory=list) 
    test_reserve_patterns: List[str] = field(default_factory=list)
    data_pairing_strategy: str = "pair"  # "pair", "unpaired", "self_supervised"
    arbitrary_processing_method: Optional[str] = None
    
    # Advanced validation (from legacy)  
    enable_validation_locking: bool = True
    save_individual_results: bool = True
    save_validation_images: bool = False
    progressive_testing_enabled: bool = False
    
    # Transfer functions and visualization (from legacy)
    training_transfer_function: str = "None"
    validation_transfer_function: str = "pq" 
    debug_visualization_options: List[str] = field(default_factory=list)
    
    # Learning rate and optimization (from legacy)
    patience_steps: int = 1000
    lr_decay_factor: float = 0.5
    warmup_steps: int = 0
    gradient_clipping_max_norm: Optional[float] = None


class AdvancedTrainer:
    """Feature-complete modern trainer with all legacy functionality."""
    
    def __init__(self, config: AdvancedTrainingConfig, training_type: str):
        self.config = config
        self.training_type = training_type
        
        # Set up sophisticated experiment management
        self.experiment_manager = self._create_experiment_manager()
        
        # Set up advanced dataset loading  
        self.dataset_manager = self._create_dataset_manager()
        
        # Set up validation system with locking
        self.validation_manager = self._create_validation_manager()
        
        # Core trainer
        basic_config = self._convert_to_basic_config(config)
        self.core_trainer = CleanDenoiserTrainer(basic_config, training_type)
    
    def _create_experiment_manager(self):
        """Create sophisticated experiment management like legacy."""
        # Auto-generate experiment names, handle iterations, etc.
        pass
    
    def _create_dataset_manager(self):
        """Create multi-dataset loading like legacy."""
        # Handle multiple YAML files, arbitrary processing, data pairing
        pass
        
    def _create_validation_manager(self):
        """Create advanced validation with locking, result caching."""
        # Implement validation locking, individual results, progressive testing
        pass
    
    def _convert_to_basic_config(self, advanced_config: AdvancedTrainingConfig) -> TrainingConfig:
        """Convert advanced config to basic config for core trainer."""
        return TrainingConfig(
            model_architecture=advanced_config.model_architecture,
            input_channels=advanced_config.input_channels,
            output_channels=advanced_config.output_channels,
            learning_rate=advanced_config.learning_rate,
            batch_size=advanced_config.batch_size,
            crop_size=advanced_config.crop_size,
            total_steps=advanced_config.total_steps,
            validation_interval=advanced_config.validation_interval,
            loss_function=advanced_config.loss_function,
            device=advanced_config.device,
        )


# ============================================================================
# 3. FACADE PATTERN PROTOTYPE  
# ============================================================================

@runtime_checkable
class TrainingBackend(Protocol):
    """Protocol for pluggable training backends."""
    
    def create_trainer(self, config: Dict[str, Any], training_type: str) -> Any:
        """Create a trainer instance using this backend."""
        ...
    
    def supports_feature(self, feature: str) -> bool:
        """Check if backend supports a specific feature."""
        ...


class LegacyTrainingBackend:
    """Backend that uses the legacy training system."""
    
    def create_trainer(self, config: Dict[str, Any], training_type: str) -> Any:
        """Create trainer using legacy classes."""
        from . import denoise_compress_trainer, denoiser_trainer
        
        if training_type == "denoise_compress_bayer":
            return denoise_compress_trainer.DCTrainingBayerToProfiledRGB(**config)
        elif training_type == "denoise_compress_rgb":
            return denoise_compress_trainer.DCTrainingProfiledRGBToProfiledRGB(**config)
        # ... etc
    
    def supports_feature(self, feature: str) -> bool:
        """Legacy backend supports all advanced features."""
        return feature in [
            "experiment_auto_naming", "validation_locking", "progressive_testing",
            "multi_dataset_loading", "arbitrary_processing", "transfer_functions",
            "individual_result_tracking", "advanced_checkpointing"
        ]


class CleanTrainingBackend:
    """Backend that uses the Clean API system."""
    
    def create_trainer(self, config: Dict[str, Any], training_type: str) -> Any:
        """Create trainer using Clean API."""
        from .clean_api import create_denoiser_trainer, create_denoise_compress_trainer, TrainingConfig
        
        # Convert dict config to TrainingConfig
        training_config = TrainingConfig(**{
            k: v for k, v in config.items() 
            if k in TrainingConfig.__dataclass_fields__
        })
        
        if "compress" in training_type:
            return create_denoise_compress_trainer(training_type.split("_")[-1], training_config)
        else:
            return create_denoiser_trainer(training_type.split("_")[-1], training_config)
    
    def supports_feature(self, feature: str) -> bool:
        """Clean backend supports only basic features."""
        return feature in ["basic_training", "simple_validation", "checkpoint_saving"]


class MLOpsTrainingBackend:
    """Backend for modern MLOPS frameworks (Weights & Biases, MLflow, etc.)."""
    
    def create_trainer(self, config: Dict[str, Any], training_type: str) -> Any:
        """Create trainer integrated with MLOPS platform."""
        # Would integrate with external frameworks
        pass
    
    def supports_feature(self, feature: str) -> bool:
        """MLOPS backend supports cloud-native features."""
        return feature in [
            "distributed_training", "hyperparameter_sweeps", "cloud_storage",
            "experiment_tracking", "model_registry", "automated_deployment"
        ]


class UniversalTrainingFacade:
    """Facade providing unified interface to different training backends."""
    
    def __init__(self):
        self.backends = {
            "legacy": LegacyTrainingBackend(),
            "clean": CleanTrainingBackend(), 
            "mlops": MLOpsTrainingBackend()
        }
        self.default_backend = "clean"
    
    def create_trainer(self, config: TrainingConfig, training_type: str, 
                      backend: Optional[str] = None, 
                      required_features: List[str] = None) -> Any:
        """Create trainer using appropriate backend based on requirements."""
        
        if backend is None:
            # Auto-select backend based on required features
            if required_features:
                for backend_name, backend_impl in self.backends.items():
                    if all(backend_impl.supports_feature(feat) for feat in required_features):
                        backend = backend_name
                        break
                else:
                    raise ValueError(f"No backend supports all features: {required_features}")
            else:
                backend = self.default_backend
        
        # Convert TrainingConfig to dict for backend compatibility
        config_dict = vars(config)
        
        return self.backends[backend].create_trainer(config_dict, training_type)
    
    def list_supported_features(self, backend: str) -> List[str]:
        """List features supported by a specific backend."""
        backend_impl = self.backends[backend]
        all_features = [
            "basic_training", "simple_validation", "checkpoint_saving",
            "experiment_auto_naming", "validation_locking", "progressive_testing", 
            "multi_dataset_loading", "arbitrary_processing", "transfer_functions",
            "distributed_training", "hyperparameter_sweeps", "cloud_storage"
        ]
        return [feat for feat in all_features if backend_impl.supports_feature(feat)]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_strategy_pattern():
    """Example of using Strategy pattern."""
    config = TrainingConfig(
        model_architecture="unet",
        input_channels=3,
        output_channels=3,
        learning_rate=1e-4,
        batch_size=4,
        crop_size=128,
        total_steps=1000,
        validation_interval=100
    )
    
    # Simple experiment for quick testing
    simple_trainer = StrategicTrainer(config, SimpleTrainingStrategy())
    
    # Research experiment with full features  
    research_trainer = StrategicTrainer(config, ResearchTrainingStrategy())
    
    return simple_trainer, research_trainer


def example_pure_modern_rewrite():
    """Example of pure modern rewrite approach."""
    advanced_config = AdvancedTrainingConfig(
        model_architecture="unet",
        input_channels=3,
        output_channels=3,
        learning_rate=1e-4,
        batch_size=4,
        crop_size=128,
        total_steps=1000,
        validation_interval=100,
        # Advanced features
        experiment_name_pattern="auto",
        enable_validation_locking=True,
        save_individual_results=True,
        progressive_testing_enabled=True,
        clean_dataset_yamls=["path/to/clean.yaml"],
        noisy_dataset_yamls=["path/to/noisy.yaml"]
    )
    
    trainer = AdvancedTrainer(advanced_config, "rgb_to_rgb")
    return trainer


def example_facade_pattern():
    """Example of using Facade pattern."""
    config = TrainingConfig(
        model_architecture="unet", 
        input_channels=3,
        output_channels=3,
        learning_rate=1e-4,
        batch_size=4,
        crop_size=128,
        total_steps=1000,
        validation_interval=100
    )
    
    facade = UniversalTrainingFacade()
    
    # Simple training - auto-selects clean backend
    simple_trainer = facade.create_trainer(config, "denoise_rgb")
    
    # Research training - auto-selects legacy backend for advanced features
    research_trainer = facade.create_trainer(
        config, "denoise_rgb",
        required_features=["validation_locking", "progressive_testing"]
    )
    
    # MLOPS training - uses cloud backend
    mlops_trainer = facade.create_trainer(
        config, "denoise_rgb", 
        backend="mlops",
        required_features=["distributed_training", "experiment_tracking"]
    )
    
    return simple_trainer, research_trainer, mlops_trainer