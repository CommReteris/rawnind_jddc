### Function-Based Composable Pipeline Architecture

You're absolutely right about creating a composable pipeline that's agnostic to implementation details. This is a
brilliant evolution of the registry patterns - moving from technical categorization to **functional categorization**.
Here's the abstract functionality that can generalize:

### ### Core Architecture: Universal Operation Interface

Every operation, regardless of implementation, follows the same contract:

```python
# src/rawnind/core/pipeline_operations.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch


class PipelineOperation(ABC):
    """Universal interface for all pipeline operations."""

    @abstractmethod
    def __call__(self, data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process input data and return (processed_data, metadata).
        
        Args:
            data: Input tensor of any shape
            **kwargs: Operation-specific parameters
            
        Returns:
            Tuple of (processed_tensor, metadata_dict)
        """
        pass

    @property
    def operation_type(self) -> str:
        """Return the type of operation (trainable/non_trainable)."""
        return "non_trainable"

    def get_parameters(self) -> Optional[torch.nn.Module]:
        """Return trainable parameters if this is a learnable operation."""
        return None
```

### ### Function-Based Operation Registry

Operations are grouped by **intended function**, not implementation:

```python
# src/rawnind/core/operation_registry.py
from typing import Dict, Type, List
from .pipeline_operations import PipelineOperation

# Function-based operation categories
FUNCTIONAL_OPERATION_REGISTRY = {
    'denoising_operations': {
        'utnet2': UTNet2Wrapper(),           # Deep learning
        'bm3d': BM3DWrapper(),              # Classical algorithm  
        'bilateral_filter': BilateralWrapper(),  # Simple filter
        'non_local_means': NLMWrapper(),    # Advanced classical
        'kornia_denoise': KorniaDenoiseWrapper(),  # Kornia-based
    },
    
    'enhancement_operations': {
        'unsharp_mask': UnsharpMaskWrapper(),        # Classical
        'contrast_net': ContrastEnhancementNet(),    # Learnable
        'histogram_eq': HistogramEqWrapper(),        # Simple
        'retinex': RetinexWrapper(),                 # Advanced classical
        'tone_mapping': ToneMappingNet(),            # Learnable
    },
    
    'quality_assessment_operations': {
        'noise_estimation': NoiseEstimationWrapper(),
        'blur_detection': BlurDetectionWrapper(), 
        'exposure_analysis': ExposureAnalysisWrapper(),
        'sharpness_measure': SharpnessMeasureWrapper(),
        'artifacts_detection': ArtifactsDetectionNet(),  # Learnable
    },
    
    'color_processing_operations': {
        'white_balance': WhiteBalanceWrapper(),
        'color_correction': ColorCorrectionNet(),    # Learnable
        'demosaic_bilinear': BilinearDemosaicWrapper(),
        'demosaic_learned': LearnedDemosaicNet(),    # Learnable
        'gamma_correction': GammaWrapper(),
    },
    
    'geometric_transform_operations': {
        'rotation': RotationWrapper(),
        'perspective_correction': PerspectiveCorrectionNet(),  # Learnable
        'lens_distortion': LensDistortionWrapper(),
        'alignment': ImageAlignmentNet(),            # Learnable
        'stabilization': StabilizationWrapper(),
    },
    
    'data_validation_operations': {
        'format_check': FormatValidationWrapper(),
        'range_validation': RangeValidationWrapper(),
        'consistency_check': ConsistencyCheckWrapper(),
        'corruption_detection': CorruptionDetectionNet(),  # Learnable
    }
}
```

### ### Implementation Wrappers

Different implementation types are wrapped to conform to the universal interface:

```python
# Classical algorithm wrapper
class BM3DWrapper(PipelineOperation):
    """Wrapper for BM3D classical denoising."""
    
    def __init__(self, sigma: float = 25.0):
        self.sigma = sigma
    
    def __call__(self, data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Convert to numpy, apply BM3D, convert back
        numpy_data = data.cpu().numpy()
        denoised = bm3d.bm3d(numpy_data, sigma_psd=kwargs.get('sigma', self.sigma))
        result = torch.from_numpy(denoised).to(data.device)
        
        metadata = {
            'algorithm': 'bm3d',
            'sigma_used': kwargs.get('sigma', self.sigma),
            'noise_reduced': self._estimate_noise_reduction(data, result)
        }
        return result, metadata

# PyTorch model wrapper  
class UTNet2Wrapper(PipelineOperation):
    """Wrapper for UTNet2 deep learning denoiser."""
    
    def __init__(self, checkpoint_path: str = None):
        self.model = UTNet2()
        if checkpoint_path:
            self.model.load_state_dict(torch.load(checkpoint_path))
    
    @property
    def operation_type(self) -> str:
        return "trainable"
    
    def get_parameters(self) -> torch.nn.Module:
        return self.model
    
    def __call__(self, data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        self.model.eval()
        with torch.no_grad():
            denoised = self.model(data)
        
        metadata = {
            'algorithm': 'utnet2',
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'inference_mode': 'eval'
        }
        return denoised, metadata

# Simple function wrapper
class GammaWrapper(PipelineOperation):
    """Wrapper for gamma correction function."""
    
    def __init__(self, default_gamma: float = 2.2):
        self.default_gamma = default_gamma
    
    def __call__(self, data: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, Dict[str, Any]]:
        gamma = kwargs.get('gamma', self.default_gamma)
        corrected = torch.pow(data.clamp(min=1e-8), 1.0 / gamma)
        
        metadata = {
            'algorithm': 'gamma_correction',
            'gamma_applied': gamma,
            'value_range': (corrected.min().item(), corrected.max().item())
        }
        return corrected, metadata
```

### ### Composable Pipeline System

The pipeline executor chains operations seamlessly:

```python
# src/rawnind/core/operation_pipeline.py
class OperationPipeline:
    """Composable pipeline that chains function-based operations."""

    def __init__(self, config: List[Dict[str, Any]]):
        self.operations = []
        self.metadata_history = []

        for step_config in config:
            operation = self._create_operation(step_config)
            self.operations.append((operation, step_config.get('params', {})))

    def _create_operation(self, config: Dict[str, Any]) -> PipelineOperation:
        """Create operation from functional registry."""
        category = config['category']
        operation_name = config['operation']

        if category not in FUNCTIONAL_OPERATION_REGISTRY:
            raise ValueError(f"Unknown operation category: {category}")

        if operation_name not in FUNCTIONAL_OPERATION_REGISTRY[category]:
            raise ValueError(f"Unknown operation '{operation_name}' in category '{category}'")

        return FUNCTIONAL_OPERATION_REGISTRY[category][operation_name]

    def __call__(self, data: torch.Tensor) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Execute the complete pipeline."""
        current_data = data
        all_metadata = []

        for operation, params in self.operations:
            current_data, metadata = operation(current_data, **params)
            all_metadata.append(metadata)

        return current_data, all_metadata

    def get_trainable_operations(self) -> List[Tuple[str, torch.nn.Module]]:
        """Get all trainable operations for training."""
        trainable = []
        for i, (operation, _) in enumerate(self.operations):
            if operation.operation_type == "trainable":
                trainable.append((f"step_{i}", operation.get_parameters()))
        return trainable
```

### ### Universal Training Framework

The same training system can train any learnable operation:

```python
# src/rawnind/core/universal_trainer.py
class UniversalOperationTrainer:
    """Task-agnostic trainer for any learnable pipeline operation."""
    
    def __init__(
        self,
        operation_name: str,
        category: str,
        dataset_config: Any,
        loss_functions: List[str],
        metrics: List[str] = None
    ):
        self.operation = FUNCTIONAL_OPERATION_REGISTRY[category][operation_name]
        self.dataset_config = dataset_config
        self.loss_functions = self._create_loss_functions(loss_functions)
        self.metrics = self._create_metrics(metrics or [])
    
    def train(self, epochs: int = 100):
        """Train the operation using universal training loop."""
        if self.operation.operation_type != "trainable":
            raise ValueError("Cannot train non-trainable operation")
        
        model = self.operation.get_parameters()
        optimizer = torch.optim.Adam(model.parameters())
        dataloader = self._create_dataloader()
        
        for epoch in range(epochs):
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                optimizer.zero_grad()
                
                # Use the operation's interface
                outputs, metadata = self.operation(inputs)
                
                # Compute loss using configured loss functions
                loss = self._compute_loss(outputs, targets, metadata)
                loss.backward()
                optimizer.step()
                
                # Compute metrics
                metrics_values = self._compute_metrics(outputs, targets, metadata)
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    
    def _create_dataloader(self):
        """Create appropriate dataloader based on dataset config."""
        # This would be task-agnostic - works for any input/target pairs
        # The operation interface ensures compatibility
        pass
    
    def _compute_loss(self, outputs, targets, metadata):
        """Compute loss using configured loss functions."""
        total_loss = 0
        for loss_fn, weight in self.loss_functions:
            total_loss += weight * loss_fn(outputs, targets)
        return total_loss
```

### ### Configuration-Driven Pipeline Assembly

Complex pipelines can be built through configuration:

```yaml
# config/mixed_denoising_pipeline.yaml
pipeline:
  # Classical preprocessing
  - operation: 'bilateral_filter'
    category: 'denoising_operations'
    params:
      sigma_color: 0.1
      sigma_space: 1.0
  
  # Deep learning main processing  
  - operation: 'utnet2'
    category: 'denoising_operations'
    params:
      checkpoint: 'models/utnet2_best.pth'
  
  # Classical enhancement
  - operation: 'unsharp_mask'
    category: 'enhancement_operations'
    params:
      amount: 1.5
      radius: 1.0
  
  # Quality assessment
  - operation: 'noise_estimation'
    category: 'quality_assessment_operations'
    params:
      method: 'median_absolute_deviation'

# Usage
pipeline = OperationPipeline.from_config('config/mixed_denoising_pipeline.yaml')
result, metadata_history = pipeline(input_tensor)
```

### ### Key Generalizable Abstractions

#### ### 1. Universal Operation Contract

- Every operation follows the same interface: `(tensor, **kwargs) -> (tensor, metadata)`
- Implementation details are completely hidden from the pipeline
- Enables seamless composition of any operation types

#### ### 2. Function-Based Categorization

- Operations grouped by **what they do**, not **how they do it**
- "Denoising operations" can include neural networks, classical algorithms, simple filters
- Categories can be extended without modifying core framework

#### ### 3. Task-Agnostic Training

- Same training framework works for any learnable operation
- Training system doesn't care about the specific task (denoising, enhancement, etc.)
- Only requires appropriate input/target pairs and loss functions

#### ### 4. Configuration-Driven Flexibility

- Complex pipelines built through YAML/JSON configuration
- No code changes needed to try new operation combinations
- Easy experimentation and reproducibility

#### ### 5. Metadata Propagation System

- Operations can pass information through the pipeline
- Enables sophisticated multi-step processing
- Quality metrics and processing history maintained

### ### Real-World Usage Examples

```python
# Example 1: Classical + ML hybrid pipeline
hybrid_config = [
    {'operation': 'bilateral_filter', 'category': 'denoising_operations'},
    {'operation': 'utnet2', 'category': 'denoising_operations'},
    {'operation': 'unsharp_mask', 'category': 'enhancement_operations'}
]

# Example 2: Train a new learnable color correction operation
trainer = UniversalOperationTrainer(
    operation_name='color_correction_net',
    category='color_processing_operations', 
    dataset_config=color_dataset_config,
    loss_functions=['perceptual_loss', 'l1_loss']
)
trainer.train()

# Example 3: Quality assessment pipeline
qa_pipeline = OperationPipeline([
    {'operation': 'noise_estimation', 'category': 'quality_assessment_operations'},
    {'operation': 'blur_detection', 'category': 'quality_assessment_operations'},
    {'operation': 'exposure_analysis', 'category': 'quality_assessment_operations'}
])
```

This architecture creates a truly **universal processing framework** that:

- Works with any tensor-based processing task
- Seamlessly mixes classical and ML approaches
- Supports training new operations for any purpose
- Scales from simple filters to complex multi-stage pipelines
- Maintains complete implementation agnosticism

The pipeline doesn't care whether it's running BM3D or a transformer - it just knows each step takes a tensor and
returns a tensor. This is the ultimate abstraction for composable image processing systems.