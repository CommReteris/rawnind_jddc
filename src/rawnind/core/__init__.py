"""Core pipeline architecture for RawNind.

This package implements the Function-Based Composable Pipeline Architecture,
providing a universal interface for all pipeline operations that is agnostic
to implementation details.

Key Components:
- PipelineOperation: Universal interface for all operations
- OperationSpec: Specification dataclass for operations
- OperationRegistry: Central registry for all available operations
- PipelineAssembler: Validates and assembles pipeline configurations
- OperationResolver: Converts simple names to full specifications
"""

from .pipeline_operation import PipelineOperation
from .operation_spec import OperationSpec, TensorType, ConstraintType
from .operation_registry import OperationRegistry, OPERATION_REGISTRY
from .pipeline_assembler import (
    OperationPipeline,
    SmartPipelineAssembler,
    MetadataDependencyResolver
)
from .operation_resolver import OperationResolver

__all__ = [
    'PipelineOperation',
    'OperationSpec',
    'TensorType',
    'ConstraintType',
    'OperationRegistry',
    'OPERATION_REGISTRY',
    'OperationPipeline',
    'SmartPipelineAssembler',
    'MetadataDependencyResolver',
    'OperationResolver',
]