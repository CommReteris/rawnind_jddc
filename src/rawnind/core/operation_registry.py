"""Central registry for all pipeline operations."""

from typing import Dict, List, Optional, Type, Any
import importlib
import logging
from .operation_spec import OperationSpec, TensorType, ConstraintType
from .pipeline_operation import PipelineOperation

logger = logging.getLogger(__name__)


class OperationRegistry:
    """Central registry for all available pipeline operations.
    
    This registry maintains a catalog of all registered operations,
    their specifications, and provides methods for discovering and
    instantiating operations.
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._specs: Dict[str, OperationSpec] = {}
        self._implementations: Dict[str, Type[PipelineOperation]] = {}
        self._categories: Dict[str, List[str]] = {}
        
    def register_spec(self, spec: OperationSpec):
        """Register an operation specification.
        
        Args:
            spec: Operation specification to register
        """
        if spec.name in self._specs:
            logger.warning(f"Overwriting existing spec for operation: {spec.name}")
        
        self._specs[spec.name] = spec
        
        # Update category index
        if spec.category not in self._categories:
            self._categories[spec.category] = []
        if spec.name not in self._categories[spec.category]:
            self._categories[spec.category].append(spec.name)
        
        logger.debug(f"Registered spec for operation: {spec.name}")
    
    def register_implementation(self, 
                               name: str, 
                               implementation: Type[PipelineOperation]):
        """Register an operation implementation.
        
        Args:
            name: Operation name
            implementation: Implementation class
        """
        if name not in self._specs:
            raise ValueError(f"No spec registered for operation: {name}")
        
        self._implementations[name] = implementation
        logger.debug(f"Registered implementation for operation: {name}")
    
    def get_spec(self, name: str) -> Optional[OperationSpec]:
        """Get specification for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Operation specification or None if not found
        """
        return self._specs.get(name)
    
    def get_implementation(self, name: str) -> Optional[Type[PipelineOperation]]:
        """Get implementation class for an operation.
        
        Args:
            name: Operation name
            
        Returns:
            Implementation class or None if not found
        """
        # Return if already loaded
        if name in self._implementations:
            return self._implementations[name]
            
        # Try to load from operations module mapping first
        if name in self._specs:
            try:
                from ..operations import OPERATION_IMPLEMENTATIONS
                if name in OPERATION_IMPLEMENTATIONS:
                    impl_class = OPERATION_IMPLEMENTATIONS[name]
                    self._implementations[name] = impl_class
                    return impl_class
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not load from operations module: {e}")
            
            # Fall back to spec-based loading
            spec = self._specs[name]
            if spec.implementation_module and spec.implementation_class:
                try:
                    module = importlib.import_module(spec.implementation_module)
                    impl_class = getattr(module, spec.implementation_class)
                    self._implementations[name] = impl_class
                    return impl_class
                except (ImportError, AttributeError) as e:
                    logger.error(f"Failed to load implementation for {name}: {e}")
        
        return None
    
    def create_operation(self, 
                        name: str, 
                        config: Optional[Dict[str, Any]] = None) -> PipelineOperation:
        """Create an instance of an operation.
        
        Args:
            name: Operation name
            config: Configuration dictionary
            
        Returns:
            Instantiated operation
            
        Raises:
            ValueError: If operation not found or cannot be instantiated
        """
        spec = self.get_spec(name)
        if not spec:
            raise ValueError(f"Unknown operation: {name}")
        
        impl_class = self.get_implementation(name)
        if not impl_class:
            raise ValueError(f"No implementation found for operation: {name}")
        
        return impl_class(spec, config)
    
    def list_operations(self, 
                       category: Optional[str] = None,
                       input_type: Optional[TensorType] = None,
                       output_type: Optional[TensorType] = None) -> List[str]:
        """List available operations with optional filtering.
        
        Args:
            category: Filter by category
            input_type: Filter by input type
            output_type: Filter by output type
            
        Returns:
            List of operation names
        """
        operations = []
        
        for name, spec in self._specs.items():
            # Category filter
            if category and spec.category != category:
                continue
            
            # Input type filter
            if input_type and input_type not in spec.input_types:
                if TensorType.ANY not in spec.input_types:
                    continue
            
            # Output type filter
            if output_type and output_type not in spec.output_types:
                if TensorType.ANY not in spec.output_types:
                    continue
            
            operations.append(name)
        
        return sorted(operations)
    
    def get_categories(self) -> List[str]:
        """Get list of all categories.
        
        Returns:
            List of category names
        """
        return sorted(self._categories.keys())
    
    def find_compatible_operations(self, 
                                  source_op: str, 
                                  position: str = "after") -> List[str]:
        """Find operations compatible with a given operation.
        
        Args:
            source_op: Source operation name
            position: Whether to find operations that can come "before" or "after"
            
        Returns:
            List of compatible operation names
        """
        source_spec = self.get_spec(source_op)
        if not source_spec:
            return []
        
        compatible = []
        
        for name, spec in self._specs.items():
            if name == source_op:
                continue
            
            if position == "after":
                # Check if source output matches target input
                if source_spec.is_compatible_with(spec):
                    compatible.append(name)
            else:  # position == "before"
                # Check if target output matches source input
                if spec.is_compatible_with(source_spec):
                    compatible.append(name)
        
        return compatible
    
    def get_pipeline_path(self, 
                         start_type: TensorType, 
                         end_type: TensorType) -> List[List[str]]:
        """Find possible operation paths between two tensor types.
        
        Args:
            start_type: Starting tensor type
            end_type: Target tensor type
            
        Returns:
            List of possible paths (each path is a list of operation names)
        """
        # Simple BFS to find paths
        from collections import deque
        
        # Find operations that accept start_type
        start_ops = []
        for name, spec in self._specs.items():
            if start_type in spec.input_types or TensorType.ANY in spec.input_types:
                start_ops.append(name)
        
        # BFS to find paths
        paths = []
        queue = deque([(op, [op]) for op in start_ops])
        visited = set()
        
        while queue and len(paths) < 10:  # Limit to 10 paths
            current_op, path = queue.popleft()
            
            if (current_op, len(path)) in visited:
                continue
            visited.add((current_op, len(path)))
            
            current_spec = self.get_spec(current_op)
            
            # Check if we've reached the target
            if end_type in current_spec.output_types:
                paths.append(path)
                continue
            
            # Find next operations
            for next_op in self.find_compatible_operations(current_op, "after"):
                if next_op not in path:  # Avoid cycles
                    queue.append((next_op, path + [next_op]))
        
        return paths
    
    def validate_pipeline(self, operations: List[str]) -> List[str]:
        """Validate a sequence of operations.
        
        Args:
            operations: List of operation names
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not operations:
            errors.append("Pipeline is empty")
            return errors
        
        # Check all operations exist
        for op in operations:
            if op not in self._specs:
                errors.append(f"Unknown operation: {op}")
        
        if errors:
            return errors
        
        # Check compatibility between consecutive operations
        for i in range(len(operations) - 1):
            current_spec = self._specs[operations[i]]
            next_spec = self._specs[operations[i + 1]]
            
            if not current_spec.is_compatible_with(next_spec):
                errors.append(
                    f"Operations {operations[i]} and {operations[i + 1]} "
                    f"are not compatible"
                )
        
        return errors
    
    def export_specs(self) -> Dict[str, Dict[str, Any]]:
        """Export all specifications as a dictionary.
        
        Returns:
            Dictionary of all specifications
        """
        return {
            name: {
                "name": spec.name,
                "category": spec.category,
                "description": spec.description,
                "input_types": [t.value for t in spec.input_types],
                "output_types": [t.value for t in spec.output_types],
                "constraints": [c.value for c in spec.constraints],
                "metadata_required": spec.metadata_fields_required,
                "metadata_generated": spec.metadata_fields_generated,
                "config_schema": spec.config_schema,
                "default_config": spec.default_config,
            }
            for name, spec in self._specs.items()
        }


# Global registry instance
OPERATION_REGISTRY = OperationRegistry()


def register_default_operations():
    """Register default operations with the global registry."""
    
    # Raw loading operations
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="raw_loader",
        category="input",
        description="Load RAW image files and extract metadata",
        input_types=[TensorType.METADATA],
        output_types=[TensorType.BAYER],
        metadata_fields_required=["filepath"],
        metadata_fields_generated=["bayer_pattern", "white_balance", "color_matrix"],
        constraints=[ConstraintType.DETERMINISTIC],
        config_schema={
            "force_rggb": bool,
            "crop_all": bool,
            "return_float": bool,
        },
        default_config={
            "force_rggb": True,
            "crop_all": True,
            "return_float": True,
        },
        implementation_module="src.rawnind.operations.raw_operations",
        implementation_class="RawLoaderOperation"
    ))
    
    # Bayer processing operations
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="white_balance",
        category="preprocessing",
        description="Apply white balance correction to Bayer mosaic",
        input_types=[TensorType.BAYER],
        output_types=[TensorType.BAYER],
        metadata_fields_required=["white_balance"],
        constraints=[ConstraintType.DETERMINISTIC, ConstraintType.PRESERVES_SHAPE],
        config_schema={
            "wb_type": str,
            "reverse": bool,
        },
        default_config={
            "wb_type": "daylight",
            "reverse": False,
        },
        implementation_module="src.rawnind.operations.bayer_operations",
        implementation_class="WhiteBalanceOperation"
    ))
    
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="bayer_to_rggb",
        category="preprocessing",
        description="Convert Bayer mosaic to 4-channel RGGB",
        input_types=[TensorType.BAYER],
        output_types=[TensorType.RGGB],
        metadata_fields_required=["bayer_pattern"],
        constraints=[ConstraintType.DETERMINISTIC, ConstraintType.CHANGES_SHAPE],
        implementation_module="src.rawnind.operations.bayer_operations",
        implementation_class="BayerToRGGBOperation"
    ))
    
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="demosaic",
        category="preprocessing",
        description="Demosaic Bayer pattern to RGB",
        input_types=[TensorType.BAYER],
        output_types=[TensorType.RGB],
        metadata_fields_required=["bayer_pattern"],
        constraints=[ConstraintType.DETERMINISTIC],
        config_schema={
            "method": str,
        },
        default_config={
            "method": "ea",  # Edge-aware
        },
        implementation_module="src.rawnind.operations.bayer_operations",
        implementation_class="DemosaicOperation"
    ))
    
    # Color transformation operations
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="color_transform",
        category="preprocessing",
        description="Apply color space transformation",
        input_types=[TensorType.RGB],
        output_types=[TensorType.RGB],
        metadata_fields_required=["color_matrix"],
        constraints=[ConstraintType.DETERMINISTIC, ConstraintType.PRESERVES_SHAPE],
        config_schema={
            "output_profile": str,
        },
        default_config={
            "output_profile": "lin_rec2020",
        },
        implementation_module="src.rawnind.operations.color_operations",
        implementation_class="ColorTransformOperation"
    ))
    
    # Model operations
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="encoder",
        category="model",
        description="Encode image to latent representation",
        input_types=[TensorType.RGB, TensorType.RGGB],
        output_types=[TensorType.LATENT],
        constraints=[ConstraintType.REQUIRES_TRAINING, ConstraintType.GPU_REQUIRED],
        config_schema={
            "model_path": str,
            "latent_dim": int,
        },
        default_config={
            "latent_dim": 512,
        },
        implementation_module="src.rawnind.operations.model_operations",
        implementation_class="EncoderOperation"
    ))
    
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="decoder",
        category="model",
        description="Decode latent representation to image",
        input_types=[TensorType.LATENT],
        output_types=[TensorType.RGB],
        constraints=[ConstraintType.REQUIRES_TRAINING, ConstraintType.GPU_REQUIRED],
        config_schema={
            "model_path": str,
        },
        implementation_module="src.rawnind.operations.model_operations",
        implementation_class="DecoderOperation"
    ))
    
    # Output operations
    OPERATION_REGISTRY.register_spec(OperationSpec(
        name="save_hdr",
        category="output",
        description="Save image as HDR file (EXR or TIFF)",
        input_types=[TensorType.RGB],
        output_types=[TensorType.METADATA],
        metadata_fields_required=["filepath"],
        constraints=[ConstraintType.DETERMINISTIC],
        config_schema={
            "format": str,
            "bit_depth": int,
            "color_profile": str,
        },
        default_config={
            "format": "exr",
            "bit_depth": 16,
            "color_profile": "lin_rec2020",
        },
        implementation_module="src.rawnind.operations.io_operations",
        implementation_class="SaveHDROperation"
    ))


# Initialize default operations when module is imported
register_default_operations()