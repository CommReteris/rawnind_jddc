"""Operation specification dataclass and related types."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union


class TensorType(Enum):
    """Enumeration of supported tensor types."""
    RAW = "raw"  # Raw sensor data
    BAYER = "bayer"  # Bayer pattern mosaic
    RGGB = "rggb"  # 4-channel RGGB
    RGB = "rgb"  # Standard RGB
    LATENT = "latent"  # Latent space representation
    COMPRESSED = "compressed"  # Compressed representation
    METADATA = "metadata"  # Metadata only
    ANY = "any"  # Accepts any type


class ConstraintType(Enum):
    """Types of constraints that operations can have."""
    REQUIRES_METADATA = "requires_metadata"
    MODIFIES_METADATA = "modifies_metadata"
    PRESERVES_SHAPE = "preserves_shape"
    CHANGES_SHAPE = "changes_shape"
    REQUIRES_TRAINING = "requires_training"
    DETERMINISTIC = "deterministic"
    PROBABILISTIC = "probabilistic"
    GPU_REQUIRED = "gpu_required"
    CPU_ONLY = "cpu_only"


@dataclass
class OperationSpec:
    """Specification for a pipeline operation.
    
    This dataclass defines the complete specification for any pipeline operation,
    including its input/output types, constraints, and performance characteristics.
    """
    
    name: str
    category: str  # e.g., "preprocessing", "augmentation", "model", "postprocessing"
    description: str
    
    # Input/Output specification
    input_types: List[TensorType]
    output_types: List[TensorType]
    input_count: int = 1
    output_count: int = 1
    
    # Metadata handling
    metadata_fields_required: List[str] = field(default_factory=list)
    metadata_fields_generated: List[str] = field(default_factory=list)
    metadata_fields_modified: List[str] = field(default_factory=list)
    
    # Constraints and characteristics
    constraints: List[ConstraintType] = field(default_factory=list)
    
    # Performance hints
    estimated_memory_mb: Optional[float] = None
    estimated_compute_ms: Optional[float] = None
    parallelizable: bool = True
    
    # Configuration schema
    config_schema: Dict[str, Any] = field(default_factory=dict)
    default_config: Dict[str, Any] = field(default_factory=dict)
    
    # Compatibility and dependencies
    compatible_with: List[str] = field(default_factory=list)
    incompatible_with: List[str] = field(default_factory=list)
    requires_operations: List[str] = field(default_factory=list)
    
    # Optional implementation hints
    implementation_class: Optional[str] = None
    implementation_module: Optional[str] = None
    
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate a configuration against the schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check required fields
        for key, schema_type in self.config_schema.items():
            if isinstance(schema_type, dict) and schema_type.get("required", False):
                if key not in config:
                    errors.append(f"Required field '{key}' missing")
        
        # Check types (simplified type checking)
        for key, value in config.items():
            if key in self.config_schema:
                expected = self.config_schema[key]
                if isinstance(expected, type):
                    if not isinstance(value, expected):
                        errors.append(
                            f"Field '{key}' should be {expected.__name__}, "
                            f"got {type(value).__name__}"
                        )
        
        return errors
    
    def get_effective_config(self, user_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge user configuration with defaults.
        
        Args:
            user_config: User-provided configuration
            
        Returns:
            Merged configuration dictionary
        """
        if user_config is None:
            return self.default_config.copy()
        
        # Start with defaults
        config = self.default_config.copy()
        # Override with user config
        config.update(user_config)
        
        return config
    
    def is_compatible_with(self, other: "OperationSpec") -> bool:
        """Check if this operation is compatible with another.
        
        Args:
            other: Another operation specification
            
        Returns:
            True if operations are compatible
        """
        # Check explicit incompatibilities
        if other.name in self.incompatible_with:
            return False
        if self.name in other.incompatible_with:
            return False
        
        # Check type compatibility (output of this feeds into other)
        if self.output_types and other.input_types:
            # Check if any output type matches any input type
            for out_type in self.output_types:
                for in_type in other.input_types:
                    if out_type == in_type or in_type == TensorType.ANY:
                        return True
            return False
        
        return True
    
    def estimate_resources(self, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Estimate resource requirements for given input shape.
        
        Args:
            input_shape: Shape of input tensor
            
        Returns:
            Dictionary with estimated memory and compute requirements
        """
        # Simple estimation based on input size
        num_elements = 1
        for dim in input_shape:
            num_elements *= dim
        
        # Base estimates (can be overridden by specific operations)
        memory_mb = (num_elements * 4) / (1024 * 1024)  # Assume float32
        if self.estimated_memory_mb:
            memory_mb = self.estimated_memory_mb
        
        compute_ms = num_elements / 1000000  # Simple linear estimate
        if self.estimated_compute_ms:
            compute_ms = self.estimated_compute_ms
        
        return {
            "memory_mb": memory_mb,
            "compute_ms": compute_ms,
            "parallelizable": self.parallelizable
        }