### Combined Function-Based Composable Pipeline Architecture

The integration of the Enhanced Operation Registry (80+ operations) with the original Function-Based Composable Pipeline
Architecture creates a comprehensive system that maintains elegant simplicity while handling production complexity.
Here's how they combine and the additional considerations:

### ### Unified Operation Interface

The combined system uses an enhanced universal interface that handles both single and multi-tensor operations:

```python
# src/rawnind/core/unified_operation_interface.py
from typing import Union, List, Tuple, Dict, Any
import torch

class UnifiedPipelineOperation(ABC):
    """Universal interface supporting both simple and complex operations."""
    
    def __init__(self, spec: Optional[OperationSpec] = None):
        self.spec = spec or self._create_default_spec()
        self.metadata_cache = {}
    
    @abstractmethod
    def process_tensors(
        self, 
        data: Union[torch.Tensor, List[torch.Tensor]], 
        metadata: Dict[str, Any], 
        **kwargs
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Process data with unified interface."""
        pass
    
    def __call__(self, data, metadata=None, **kwargs):
        """Main entry point with automatic validation."""
        metadata = metadata or {}
        
        # Auto-convert single tensor to list for multi-operations
        if self.spec.input_count[0] > 1 and isinstance(data, torch.Tensor):
            raise ValueError(f"{self.spec.name} requires {self.spec.input_count} inputs")
        
        # Convert single tensor operations to use list interface internally
        input_data = [data] if isinstance(data, torch.Tensor) else data
        
        self.validate_inputs(input_data, metadata)
        outputs, output_metadata = self.process_tensors(input_data, metadata, **kwargs)
        
        # Convert back to single tensor if operation expects single output
        if self.spec.output_count == 1 and isinstance(outputs, list):
            outputs = outputs[0]
        
        return outputs, output_metadata
```

### ### Hierarchical Function-Based Registry

The operations are organized by functional intent with sub-categories for implementation details:

```python
# src/rawnind/core/unified_operation_registry.py
UNIFIED_FUNCTIONAL_REGISTRY = {
    'denoising_operations': {
        # Simple operations (original style)
        'denoise': SimpleDenoiseWrapper(),
        'bilateral_filter': BilateralWrapper(),
        
        # Raw-specific operations (enhanced registry)  
        'raw_denoise': RawDenoiseWrapper(),
        'hotpixel_correction': HotPixelWrapper(),
        
        # Multi-frame operations (enhanced registry)
        'temporal_denoise': TemporalDenoiseWrapper(),
        
        # ML operations (both styles)
        'utnet2': UTNet2Wrapper(),
        'learned_denoise': LearnedDenoiseNet(),
    },
    
    'enhancement_operations': {
        # Tone mapping sub-category
        'exposure_adjust': ExposureWrapper(),
        'filmic_tone_map': FilmicWrapper(),
        'sigmoid_tone_map': SigmoidWrapper(),
        
        # Sharpening sub-category  
        'unsharp_mask': UnsharpWrapper(),
        'adaptive_sharpen': AdaptiveSharpenWrapper(),
        
        # Lens correction sub-category
        'lens_correction': LensWrapper(),
        'chromatic_aberration': ChromaticWrapper(),
    },
    
    'burst_processing_operations': {
        # Multi-image fusion
        'hdr_merge': HDRMergeWrapper(),
        'focus_stack': FocusStackWrapper(),
        'panorama_stitch': PanoramaWrapper(),
        'super_resolution': SuperResWrapper(),
    },
    
    'quality_assessment_operations': {
        # Analysis operations
        'noise_estimation': NoiseEstimationWrapper(),
        'blur_detection': BlurDetectionWrapper(),
        'exposure_analysis': ExposureAnalysisWrapper(),
        'artifact_detection': ArtifactDetectionNet(),
    }
}
```

### ### Enhanced Metadata System with Dependency Resolution

The system validates metadata dependencies and provides intelligent error handling:

```python
# src/rawnind/core/metadata_system.py
class MetadataDependencyResolver:
    """Resolves and validates metadata dependencies across pipeline."""
    
    def __init__(self, operations: List[OperationSpec]):
        self.operations = operations
        self.dependency_graph = self._build_dependency_graph()
    
    def validate_pipeline(self, initial_metadata: Dict[str, Any]) -> List[str]:
        """Validate that all metadata requirements can be satisfied."""
        available = set(initial_metadata.keys())
        warnings = []
        
        for op in self.operations:
            # Check requirements
            missing = set(op.requires_metadata) - available
            if missing:
                # Try to suggest operations that could provide missing metadata
                suggestions = self._suggest_providers(missing)
                warnings.append(f"{op.name} missing: {missing}. Consider: {suggestions}")
            
            # Add produced metadata
            available.update(op.produces_metadata)
        
        return warnings
    
    def _suggest_providers(self, missing_metadata: set) -> List[str]:
        """Suggest operations that could provide missing metadata."""
        suggestions = []
        for category, ops in UNIFIED_FUNCTIONAL_REGISTRY.items():
            for op_name, op in ops.items():
                if hasattr(op, 'spec') and set(op.spec.produces_metadata) & missing_metadata:
                    suggestions.append(f"{category}.{op_name}")
        return suggestions
```

### ### Adaptive Training Framework

The training system adapts to different operation types and input/output patterns:

```python
# src/rawnind/core/adaptive_trainer.py
class AdaptiveUniversalTrainer:
    """Training framework that adapts to operation characteristics."""
    
    def __init__(self, operation_config: Dict[str, Any]):
        self.operation = self._create_operation(operation_config)
        self.batch_strategy = self._determine_batch_strategy()
        self.loss_strategy = self._determine_loss_strategy()
    
    def _determine_batch_strategy(self) -> str:
        """Determine batching strategy based on operation requirements."""
        if self.operation.spec.input_count[0] > 1:
            return "multi_image_grouping"
        elif "burst" in self.operation.spec.name.lower():
            return "temporal_batching"  
        else:
            return "standard_batching"
    
    def _create_training_batch(self, dataset_batch):
        """Create appropriate training batch based on operation type."""
        if self.batch_strategy == "multi_image_grouping":
            # Group images for multi-input operations
            return self._group_for_multi_input(dataset_batch)
        elif self.batch_strategy == "temporal_batching":
            # Create temporal sequences for burst processing
            return self._create_temporal_batches(dataset_batch)
        else:
            return dataset_batch
    
    def _compute_adaptive_loss(self, outputs, targets, metadata):
        """Compute loss adapted to operation output characteristics."""
        if isinstance(outputs, list) and len(outputs) != len(targets):
            # Operation changed tensor count - use adaptive loss
            return self._compute_count_adaptive_loss(outputs, targets, metadata)
        
        # Use standard loss computation
        return self._compute_standard_loss(outputs, targets, metadata)
```

### ### Layered Configuration System

The system supports both simple and advanced configuration modes:

```python
# src/rawnind/core/configuration_system.py
class LayeredConfigurationSystem:
    """Supports both simple and advanced configuration modes."""
    
    def __init__(self, registry: Dict):
        self.registry = registry
        self.operation_resolver = OperationResolver(registry)
    
    def resolve_simple_config(self, simple_config: List[Dict]) -> List[Dict]:
        """Expand simple configuration to full specification."""
        resolved = []
        
        for step in simple_config:
            if 'category' not in step:
                # Auto-resolve operation to category
                op_name = step['operation']
                category, full_spec = self.operation_resolver.resolve_operation(op_name)
                step['category'] = category
                step['resolved_spec'] = full_spec
            
            resolved.append(step)
        
        return resolved
    
    def validate_and_suggest(self, config: List[Dict]) -> Dict[str, Any]:
        """Validate configuration and provide suggestions."""
        resolved_config = self.resolve_simple_config(config)
        validator = SmartPipelineAssembler(self.registry)
        
        warnings = validator.validate_pipeline_compatibility(resolved_config)
        suggestions = []
        
        if warnings:
            for warning in warnings:
                # Generate specific suggestions for each warning
                suggestion = self._generate_suggestion(warning, resolved_config)
                suggestions.append(suggestion)
        
        return {
            'resolved_config': resolved_config,
            'warnings': warnings,
            'suggestions': suggestions,
            'auto_fixes': self._generate_auto_fixes(warnings, resolved_config)
        }

# Usage examples:
# Simple mode (original elegance)
simple_pipeline = [
    {'operation': 'denoise', 'params': {'strength': 0.3}},
    {'operation': 'sharpen', 'params': {'amount': 1.2}},
    {'operation': 'tone_map', 'params': {'method': 'filmic'}}
]

# Advanced mode (full specification)
advanced_pipeline = [
    {
        'operation': 'hdr_merge',
        'category': 'burst_processing_operations',
        'params': {'alignment_method': 'feature_based', 'merge_algorithm': 'exposure_fusion'},
        'constraints': {'min_images': 3, 'max_images': 9}
    }
]
```

### ### Optimized Execution Engine

The execution engine optimizes based on operation characteristics:

```python
# src/rawnind/core/optimized_executor.py
class OptimizedPipelineExecutor:
    """Execution engine optimized for mixed operation types."""
    
    def __init__(self, operations: List[UnifiedPipelineOperation]):
        self.operations = operations
        self.execution_plan = self._create_execution_plan()
    
    def _create_execution_plan(self) -> Dict[str, Any]:
        """Analyze pipeline and create optimized execution plan."""
        plan = {
            'batching_groups': [],
            'memory_checkpoints': [],
            'device_assignments': {},
            'parallelization_opportunities': []
        }
        
        # Group consecutive single-image operations for batching
        current_batch_group = []
        for i, op in enumerate(self.operations):
            if op.spec.input_count == (1, 1) and op.spec.output_count == 1:
                current_batch_group.append(i)
            else:
                if current_batch_group:
                    plan['batching_groups'].append(current_batch_group)
                    current_batch_group = []
                
                # Mark multi-image operations for special handling
                if op.spec.input_count[0] > 1:
                    plan['memory_checkpoints'].append(i)
        
        return plan
    
    def execute_optimized(self, data, metadata=None):
        """Execute pipeline with optimizations."""
        current_data = data
        current_metadata = metadata or {}
        
        for group in self.execution_plan['batching_groups']:
            # Batch process consecutive single-image operations
            current_data, current_metadata = self._execute_batched_group(
                group, current_data, current_metadata
            )
        
        # Handle memory-intensive operations individually
        for checkpoint_idx in self.execution_plan['memory_checkpoints']:
            operation = self.operations[checkpoint_idx]
            current_data, current_metadata = self._execute_with_memory_management(
                operation, current_data, current_metadata
            )
        
        return current_data, current_metadata
```

### ### Additional Considerations from Integration

#### ### 1. Version Management and Operation Variants

```python
# Operations can have multiple implementations
ENHANCED_REGISTRY = {
    'demosaic_operations': {
        'demosaic': {
            'bilinear': BilinearDemosaicWrapper(),
            'amaze': AMaZEDemosaicWrapper(), 
            'learned': LearnedDemosaicNet(),
        }
    }
}

# Configuration supports version selection
config = {
    'operation': 'demosaic.amaze',  # Specific implementation
    'params': {'quality': 'high'}
}
```

#### ### 2. Fallback Strategies and Error Handling

```python
class RobustPipelineExecutor:
    """Pipeline executor with fallback strategies."""
    
    def execute_with_fallbacks(self, data, metadata):
        """Execute with automatic fallback on constraint failures."""
        for operation in self.operations:
            try:
                data, metadata = operation(data, metadata)
            except ConstraintViolationError as e:
                # Try fallback operation
                fallback = self._get_fallback_operation(operation, e)
                if fallback:
                    data, metadata = fallback(data, metadata)
                else:
                    raise
        return data, metadata
```

#### ### 3. Pipeline Debugging and Introspection

```python
class PipelineDebugger:
    """Debugging tools for complex pipelines."""
    
    def trace_execution(self, pipeline, data):
        """Trace pipeline execution with intermediate results."""
        trace = {'inputs': data, 'steps': []}
        
        for i, operation in enumerate(pipeline.operations):
            step_result = operation(data)
            trace['steps'].append({
                'operation': operation.spec.name,
                'input_shape': data.shape if hasattr(data, 'shape') else str(type(data)),
                'output_shape': step_result[0].shape if hasattr(step_result[0], 'shape') else str(type(step_result[0])),
                'metadata': step_result[1]
            })
            data = step_result[0]
        
        return trace
```

### ### Key Benefits of the Combined System

#### ### 1. **Maintained Elegance**

- Simple operations still use the elegant single-line configuration
- Complex operations can be configured in detail when needed
- The "pipeline doesn't care about implementation" philosophy is preserved

#### ### 2. **Production Ready**

- 80+ operations covering complete raw development workflows
- Comprehensive constraint validation and metadata management
- Optimized execution for mixed operation types

#### ### 3. **Universal Applicability**

- Handles single images, burst processing, and multi-image workflows
- Works with classical algorithms, simple functions, and deep learning models
- Task-agnostic training supports any tensor-to-tensor learning problem

#### ### 4. **Intelligent Assistance**

- Auto-resolves simple configurations to full specifications
- Suggests missing operations and fixes pipeline incompatibilities
- Provides debugging and introspection tools

The combined system represents the evolution from an elegant concept to a comprehensive, production-ready framework that
maintains the core philosophy while handling real-world complexity. It's simultaneously simple enough for basic use and
powerful enough for sophisticated multi-stage processing workflows.