"""Base class for all pipeline operations."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import torch
import numpy as np
from .operation_spec import OperationSpec


class PipelineOperation(ABC):
    """Universal interface for all pipeline operations.
    
    This abstract base class defines the standard interface that all pipeline
    operations must implement, ensuring compatibility and composability across
    the entire system.
    """
    
    def __init__(self, spec: OperationSpec, config: Optional[Dict[str, Any]] = None):
        """Initialize the operation.
        
        Args:
            spec: Operation specification
            config: Configuration dictionary (uses defaults if not provided)
        """
        self.spec = spec
        self.config = spec.get_effective_config(config)
        
        # Validate configuration
        errors = spec.validate_config(self.config)
        if errors:
            raise ValueError(f"Invalid configuration for {spec.name}: {', '.join(errors)}")
        
        # Initialize the operation
        self._initialized = False
        self._device = None
        
    @abstractmethod
    def forward(self, 
                data: Union[torch.Tensor, np.ndarray, Dict[str, Any]], 
                metadata: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Execute the operation.
        
        Args:
            data: Input data (tensor, array, or dictionary)
            metadata: Optional metadata dictionary
            
        Returns:
            Tuple of (output_data, updated_metadata)
        """
        pass
    
    def __call__(self, 
                 data: Union[torch.Tensor, np.ndarray, Dict[str, Any]], 
                 metadata: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
        """Make the operation callable.
        
        Args:
            data: Input data
            metadata: Optional metadata
            
        Returns:
            Tuple of (output_data, updated_metadata)
        """
        # Ensure operation is initialized
        if not self._initialized:
            self.initialize()
        
        # Validate metadata requirements
        if metadata is None:
            metadata = {}
        
        missing_fields = []
        for field in self.spec.metadata_fields_required:
            if field not in metadata:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(
                f"Operation {self.spec.name} requires metadata fields: {missing_fields}"
            )
        
        # Execute the operation
        output_data, output_metadata = self.forward(data, metadata)
        
        # Update metadata with generated fields
        for field in self.spec.metadata_fields_generated:
            if field not in output_metadata:
                raise RuntimeError(
                    f"Operation {self.spec.name} failed to generate required "
                    f"metadata field: {field}"
                )
        
        return output_data, output_metadata
    
    def initialize(self, device: Optional[str] = None):
        """Initialize the operation (load models, allocate resources, etc.).
        
        Args:
            device: Device to use (e.g., 'cuda', 'cpu')
        """
        self._device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialized = True
    
    def cleanup(self):
        """Clean up resources used by the operation."""
        self._initialized = False
        self._device = None
    
    @property
    def device(self) -> Optional[str]:
        """Get the current device."""
        return self._device or "cpu"
    
    def to(self, device: str) -> "PipelineOperation":
        """Move operation to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self._device = device
        return self
    
    def estimate_resources(self, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """Estimate resource requirements.
        
        Args:
            input_shape: Shape of input tensor
            
        Returns:
            Dictionary with resource estimates
        """
        return self.spec.estimate_resources(input_shape)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the operation.
        
        Returns:
            Dictionary with operation information
        """
        return {
            "name": self.spec.name,
            "category": self.spec.category,
            "description": self.spec.description,
            "config": self.config,
            "initialized": self._initialized,
            "device": self._device,
            "input_types": [t.value for t in self.spec.input_types],
            "output_types": [t.value for t in self.spec.output_types],
        }
    
    def __repr__(self) -> str:
        """String representation of the operation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.spec.name}', "
            f"category='{self.spec.category}', "
            f"initialized={self._initialized})"
        )