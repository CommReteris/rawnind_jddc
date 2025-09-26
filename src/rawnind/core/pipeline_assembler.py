"""Pipeline assembly and validation components."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import torch
import numpy as np
from .pipeline_operation import PipelineOperation
from .operation_spec import OperationSpec, TensorType
from .operation_registry import OPERATION_REGISTRY

logger = logging.getLogger(__name__)


@dataclass
class OperationPipeline:
    """Container for a sequence of pipeline operations.
    
    This class represents a complete pipeline configuration with
    its operations and metadata handling.
    """
    
    operations: List[PipelineOperation]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata_flow: Dict[str, List[str]] = field(default_factory=dict)
    
    def __call__(self, 
                 data: Union[torch.Tensor, np.ndarray, Dict[str, Any]], 
                 metadata: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Execute the pipeline.
        
        Args:
            data: Input data
            metadata: Initial metadata
            
        Returns:
            Tuple of (output_data, final_metadata)
        """
        current_data = data
        current_metadata = metadata or {}
        
        for i, operation in enumerate(self.operations):
            try:
                current_data, current_metadata = operation(current_data, current_metadata)
                logger.debug(f"Executed operation {i}: {operation.spec.name}")
            except Exception as e:
                logger.error(f"Error in operation {i} ({operation.spec.name}): {e}")
                raise
        
        return current_data, current_metadata
    
    def get_metadata_requirements(self) -> List[str]:
        """Get list of required metadata fields for the entire pipeline.
        
        Returns:
            List of required metadata field names
        """
        required = set()
        generated = set()
        
        for operation in self.operations:
            # Add required fields that haven't been generated yet
            for field in operation.spec.metadata_fields_required:
                if field not in generated:
                    required.add(field)
            
            # Track generated fields
            generated.update(operation.spec.metadata_fields_generated)
        
        return list(required)
    
    def estimate_resources(self, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Estimate total resource requirements for the pipeline.
        
        Args:
            input_shape: Initial input shape
            
        Returns:
            Dictionary with resource estimates
        """
        total_memory = 0
        total_compute = 0
        current_shape = input_shape
        
        for operation in self.operations:
            estimates = operation.estimate_resources(current_shape)
            total_memory = max(total_memory, estimates["memory_mb"])
            total_compute += estimates["compute_ms"]
            
            # Try to estimate output shape (simplified)
            if operation.spec.output_count == 1:
                # Assume shape is preserved unless specified otherwise
                pass  # Keep current_shape
        
        return {
            "total_memory_mb": total_memory,
            "total_compute_ms": total_compute,
            "num_operations": len(self.operations)
        }
    
    def to(self, device: str) -> "OperationPipeline":
        """Move all operations to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        for operation in self.operations:
            operation.to(device)
        return self
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the pipeline.
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            "name": self.name,
            "description": self.description,
            "num_operations": len(self.operations),
            "operations": [op.get_info() for op in self.operations],
            "metadata_requirements": self.get_metadata_requirements(),
            "metadata_flow": self.metadata_flow
        }


class SmartPipelineAssembler:
    """Intelligent pipeline assembler with validation and optimization.
    
    This class provides advanced pipeline assembly features including
    automatic validation, metadata flow analysis, and optimization.
    """
    
    def __init__(self, registry=None):
        """Initialize the assembler.
        
        Args:
            registry: Operation registry to use (defaults to global registry)
        """
        self.registry = registry or OPERATION_REGISTRY
    
    def assemble_from_names(self, 
                           operation_names: List[str],
                           configs: Optional[Dict[str, Dict[str, Any]]] = None,
                           name: Optional[str] = None,
                           description: Optional[str] = None) -> OperationPipeline:
        """Assemble a pipeline from operation names.
        
        Args:
            operation_names: List of operation names
            configs: Optional configuration for each operation
            name: Optional pipeline name
            description: Optional pipeline description
            
        Returns:
            Assembled pipeline
            
        Raises:
            ValueError: If pipeline is invalid
        """
        # Validate pipeline
        errors = self.registry.validate_pipeline(operation_names)
        if errors:
            raise ValueError(f"Invalid pipeline: {', '.join(errors)}")
        
        # Create operations
        operations = []
        configs = configs or {}
        
        for op_name in operation_names:
            config = configs.get(op_name)
            operation = self.registry.create_operation(op_name, config)
            operations.append(operation)
        
        # Analyze metadata flow
        metadata_flow = self._analyze_metadata_flow(operations)
        
        return OperationPipeline(
            operations=operations,
            name=name,
            description=description,
            metadata_flow=metadata_flow
        )
    
    def assemble_from_specs(self,
                           specs: List[Tuple[str, Optional[Dict[str, Any]]]],
                           name: Optional[str] = None,
                           description: Optional[str] = None) -> OperationPipeline:
        """Assemble a pipeline from operation specifications.
        
        Args:
            specs: List of (operation_name, config) tuples
            name: Optional pipeline name
            description: Optional pipeline description
            
        Returns:
            Assembled pipeline
        """
        operation_names = [spec[0] for spec in specs]
        configs = {spec[0]: spec[1] for spec in specs if spec[1] is not None}
        
        return self.assemble_from_names(
            operation_names=operation_names,
            configs=configs,
            name=name,
            description=description
        )
    
    def auto_assemble(self,
                     start_type: TensorType,
                     end_type: TensorType,
                     constraints: Optional[List[str]] = None) -> OperationPipeline:
        """Automatically assemble a pipeline between two tensor types.
        
        Args:
            start_type: Starting tensor type
            end_type: Target tensor type
            constraints: Optional list of required operations
            
        Returns:
            Assembled pipeline
            
        Raises:
            ValueError: If no valid path found
        """
        # Find possible paths
        paths = self.registry.get_pipeline_path(start_type, end_type)
        
        if not paths:
            raise ValueError(f"No path found from {start_type} to {end_type}")
        
        # Filter by constraints if provided
        if constraints:
            valid_paths = []
            for path in paths:
                if all(op in path for op in constraints):
                    valid_paths.append(path)
            paths = valid_paths
        
        if not paths:
            raise ValueError(
                f"No path found from {start_type} to {end_type} "
                f"with constraints {constraints}"
            )
        
        # Use the shortest path
        shortest_path = min(paths, key=len)
        
        return self.assemble_from_names(
            operation_names=shortest_path,
            name=f"auto_{start_type.value}_to_{end_type.value}",
            description=f"Auto-assembled pipeline from {start_type.value} to {end_type.value}"
        )
    
    def _analyze_metadata_flow(self, operations: List[PipelineOperation]) -> Dict[str, List[str]]:
        """Analyze metadata flow through the pipeline.
        
        Args:
            operations: List of operations
            
        Returns:
            Dictionary mapping metadata fields to operations that use/generate them
        """
        flow = {}
        
        for i, operation in enumerate(operations):
            op_name = f"{i}_{operation.spec.name}"
            
            # Track required fields
            for field in operation.spec.metadata_fields_required:
                if field not in flow:
                    flow[field] = []
                flow[field].append(f"{op_name}_requires")
            
            # Track generated fields
            for field in operation.spec.metadata_fields_generated:
                if field not in flow:
                    flow[field] = []
                flow[field].append(f"{op_name}_generates")
            
            # Track modified fields
            for field in operation.spec.metadata_fields_modified:
                if field not in flow:
                    flow[field] = []
                flow[field].append(f"{op_name}_modifies")
        
        return flow
    
    def optimize_pipeline(self, pipeline: OperationPipeline) -> OperationPipeline:
        """Optimize a pipeline for better performance.
        
        Args:
            pipeline: Pipeline to optimize
            
        Returns:
            Optimized pipeline
        """
        # Simple optimization: identify parallelizable operations
        optimized_operations = []
        
        for i, operation in enumerate(pipeline.operations):
            # Check if operation can be parallelized with next operation
            if i < len(pipeline.operations) - 1:
                next_op = pipeline.operations[i + 1]
                
                # Check if operations are independent (simplified check)
                current_outputs = set(operation.spec.metadata_fields_generated)
                next_requires = set(next_op.spec.metadata_fields_required)
                
                if not current_outputs.intersection(next_requires):
                    # Operations could potentially be parallelized
                    logger.debug(
                        f"Operations {operation.spec.name} and {next_op.spec.name} "
                        f"could be parallelized"
                    )
            
            optimized_operations.append(operation)
        
        pipeline.operations = optimized_operations
        return pipeline


class MetadataDependencyResolver:
    """Resolves metadata dependencies in pipelines.
    
    This class analyzes and resolves metadata dependencies to ensure
    all required metadata is available when needed.
    """
    
    def __init__(self, registry=None):
        """Initialize the resolver.
        
        Args:
            registry: Operation registry to use
        """
        self.registry = registry or OPERATION_REGISTRY
    
    def resolve_dependencies(self, 
                            operations: List[str],
                            initial_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve metadata dependencies for a pipeline.
        
        Args:
            operations: List of operation names
            initial_metadata: Initially available metadata
            
        Returns:
            Dictionary of required metadata sources
            
        Raises:
            ValueError: If dependencies cannot be resolved
        """
        available = set(initial_metadata.keys())
        required_sources = {}
        
        for op_name in operations:
            spec = self.registry.get_spec(op_name)
            if not spec:
                raise ValueError(f"Unknown operation: {op_name}")
            
            # Check required fields
            for field in spec.metadata_fields_required:
                if field not in available:
                    # Try to find an operation that generates this field
                    generator = self._find_metadata_generator(field, operations[:operations.index(op_name)])
                    if generator:
                        required_sources[field] = generator
                    else:
                        raise ValueError(
                            f"Operation {op_name} requires metadata field '{field}' "
                            f"which is not available"
                        )
            
            # Add generated fields to available
            available.update(spec.metadata_fields_generated)
        
        return required_sources
    
    def _find_metadata_generator(self, field: str, prior_operations: List[str]) -> Optional[str]:
        """Find an operation that generates a metadata field.
        
        Args:
            field: Metadata field name
            prior_operations: Operations that come before in the pipeline
            
        Returns:
            Operation name that generates the field, or None
        """
        # Check prior operations first
        for op_name in reversed(prior_operations):
            spec = self.registry.get_spec(op_name)
            if spec and field in spec.metadata_fields_generated:
                return op_name
        
        # Check all registered operations
        for op_name, spec in self.registry._specs.items():
            if field in spec.metadata_fields_generated:
                return op_name
        
        return None
    
    def suggest_metadata_operations(self, 
                                   missing_fields: List[str]) -> List[str]:
        """Suggest operations that could provide missing metadata.
        
        Args:
            missing_fields: List of missing metadata fields
            
        Returns:
            List of operation names that could provide the metadata
        """
        suggestions = []
        
        for field in missing_fields:
            generator = self._find_metadata_generator(field, [])
            if generator and generator not in suggestions:
                suggestions.append(generator)
        
        return suggestions