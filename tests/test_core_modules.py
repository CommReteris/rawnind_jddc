"""Test the core pipeline modules."""

import pytest
from typing import Dict, Any, Tuple
import torch
import numpy as np

from src.rawnind.core import (
    PipelineOperation,
    OperationSpec,
    TensorType,
    ConstraintType,
    OperationRegistry,
    OPERATION_REGISTRY,
    OperationPipeline,
    SmartPipelineAssembler,
    MetadataDependencyResolver,
    OperationResolver,
)


class TestOperationSpec:
    """Test the OperationSpec dataclass."""
    
    def test_basic_spec_creation(self):
        """Test creating a basic operation specification."""
        spec = OperationSpec(
            name="test_op",
            category="test",
            description="A test operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.LATENT],
        )
        
        assert spec.name == "test_op"
        assert spec.category == "test"
        assert TensorType.RGB in spec.input_types
        assert TensorType.LATENT in spec.output_types
    
    def test_config_validation(self):
        """Test configuration validation."""
        spec = OperationSpec(
            name="test_op",
            category="test",
            description="A test operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.RGB],
            config_schema={
                "scale": float,
                "method": str,
            },
            default_config={
                "scale": 1.0,
                "method": "bilinear",
            }
        )
        
        # Valid config
        errors = spec.validate_config({"scale": 2.0, "method": "cubic"})
        assert len(errors) == 0
        
        # Invalid type
        errors = spec.validate_config({"scale": "not_a_float"})
        assert len(errors) > 0
    
    def test_compatibility_check(self):
        """Test operation compatibility checking."""
        spec1 = OperationSpec(
            name="op1",
            category="test",
            description="First operation",
            input_types=[TensorType.BAYER],
            output_types=[TensorType.RGB],
        )
        
        spec2 = OperationSpec(
            name="op2",
            category="test",
            description="Second operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.LATENT],
        )
        
        spec3 = OperationSpec(
            name="op3",
            category="test",
            description="Third operation",
            input_types=[TensorType.BAYER],
            output_types=[TensorType.RGGB],
        )
        
        # spec1 output matches spec2 input
        assert spec1.is_compatible_with(spec2)
        
        # spec1 output doesn't match spec3 input
        assert not spec1.is_compatible_with(spec3)


class TestOperationRegistry:
    """Test the operation registry."""
    
    def test_register_and_retrieve_spec(self):
        """Test registering and retrieving specifications."""
        registry = OperationRegistry()
        
        spec = OperationSpec(
            name="test_op",
            category="test",
            description="A test operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.LATENT],
        )
        
        registry.register_spec(spec)
        
        retrieved = registry.get_spec("test_op")
        assert retrieved is not None
        assert retrieved.name == "test_op"
    
    def test_list_operations(self):
        """Test listing operations with filters."""
        registry = OperationRegistry()
        
        # Register multiple specs
        specs = [
            OperationSpec(
                name="op1",
                category="preprocess",
                description="Op 1",
                input_types=[TensorType.BAYER],
                output_types=[TensorType.RGB],
            ),
            OperationSpec(
                name="op2",
                category="model",
                description="Op 2",
                input_types=[TensorType.RGB],
                output_types=[TensorType.LATENT],
            ),
            OperationSpec(
                name="op3",
                category="preprocess",
                description="Op 3",
                input_types=[TensorType.BAYER],
                output_types=[TensorType.RGGB],
            ),
        ]
        
        for spec in specs:
            registry.register_spec(spec)
        
        # List all
        all_ops = registry.list_operations()
        assert len(all_ops) == 3
        
        # Filter by category
        preprocess_ops = registry.list_operations(category="preprocess")
        assert len(preprocess_ops) == 2
        
        # Filter by input type
        bayer_ops = registry.list_operations(input_type=TensorType.BAYER)
        assert len(bayer_ops) == 2
    
    def test_find_compatible_operations(self):
        """Test finding compatible operations."""
        registry = OperationRegistry()
        
        specs = [
            OperationSpec(
                name="bayer_input",
                category="input",
                description="Bayer input",
                input_types=[TensorType.METADATA],
                output_types=[TensorType.BAYER],
            ),
            OperationSpec(
                name="bayer_to_rgb",
                category="preprocess",
                description="Convert to RGB",
                input_types=[TensorType.BAYER],
                output_types=[TensorType.RGB],
            ),
            OperationSpec(
                name="rgb_encoder",
                category="model",
                description="Encode RGB",
                input_types=[TensorType.RGB],
                output_types=[TensorType.LATENT],
            ),
        ]
        
        for spec in specs:
            registry.register_spec(spec)
        
        # Find operations that can come after bayer_input
        after_bayer = registry.find_compatible_operations("bayer_input", "after")
        assert "bayer_to_rgb" in after_bayer
        assert "rgb_encoder" not in after_bayer
        
        # Find operations that can come before rgb_encoder
        before_encoder = registry.find_compatible_operations("rgb_encoder", "before")
        assert "bayer_to_rgb" in before_encoder
        assert "bayer_input" not in before_encoder


class TestPipelineOperation:
    """Test the PipelineOperation base class."""
    
    def test_pipeline_operation_initialization(self):
        """Test initializing a pipeline operation."""
        spec = OperationSpec(
            name="test_op",
            category="test",
            description="A test operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.RGB],
            default_config={"scale": 2.0}
        )
        
        # Create a simple test implementation
        class TestOperation(PipelineOperation):
            def forward(self, data, metadata):
                return data, metadata
        
        op = TestOperation(spec, {"scale": 3.0})
        
        assert op.spec.name == "test_op"
        assert op.config["scale"] == 3.0
        assert not op._initialized
    
    def test_device_management(self):
        """Test device management methods."""
        spec = OperationSpec(
            name="test_op",
            category="test",
            description="A test operation",
            input_types=[TensorType.RGB],
            output_types=[TensorType.RGB],
        )
        
        class TestOperation(PipelineOperation):
            def forward(self, data, metadata):
                return data, metadata
        
        op = TestOperation(spec)
        
        # Check initial device
        assert op.device == "cpu"
        
        # Move to CUDA (if available)
        if torch.cuda.is_available():
            op.to("cuda")
            assert op.device == "cuda"


class TestSmartPipelineAssembler:
    """Test the smart pipeline assembler."""
    
    def test_assemble_from_names(self):
        """Test assembling a pipeline from operation names."""
        # Use the global registry with default operations
        assembler = SmartPipelineAssembler(OPERATION_REGISTRY)
        
        # Try to assemble a simple pipeline
        pipeline = assembler.assemble_from_names(
            ["raw_loader", "white_balance", "demosaic"],
            name="test_pipeline",
            description="A test pipeline"
        )
        
        assert pipeline.name == "test_pipeline"
        assert len(pipeline.operations) == 3
        assert pipeline.operations[0].spec.name == "raw_loader"
    
    def test_invalid_pipeline_detection(self):
        """Test that invalid pipelines are detected."""
        assembler = SmartPipelineAssembler(OPERATION_REGISTRY)
        
        # Try to assemble an incompatible pipeline
        with pytest.raises(ValueError) as excinfo:
            assembler.assemble_from_names(
                ["raw_loader", "encoder"]  # encoder needs RGB/RGGB, not BAYER
            )
        assert "not compatible" in str(excinfo.value).lower()
    
    def test_metadata_flow_analysis(self):
        """Test metadata flow analysis."""
        assembler = SmartPipelineAssembler(OPERATION_REGISTRY)
        
        pipeline = assembler.assemble_from_names(
            ["raw_loader", "white_balance"]
        )
        
        # Check metadata requirements
        required = pipeline.get_metadata_requirements()
        assert "filepath" in required  # raw_loader needs filepath
        
        # Check metadata flow
        assert pipeline.metadata_flow is not None


class TestOperationResolver:
    """Test the operation resolver."""
    
    def test_resolve_aliases(self):
        """Test resolving operation aliases."""
        resolver = OperationResolver(OPERATION_REGISTRY)
        
        # Test alias resolution
        name, config = resolver.resolve_operation("wb")
        assert name == "white_balance"
        
        name, config = resolver.resolve_operation("load")
        assert name == "raw_loader"
    
    def test_resolve_pipeline_string(self):
        """Test resolving pipeline specification strings."""
        resolver = OperationResolver(OPERATION_REGISTRY)
        
        # Simple pipeline string
        operations = resolver.resolve_pipeline("load -> wb -> demosaic")
        assert len(operations) == 3
        assert operations[0][0] == "raw_loader"
        assert operations[1][0] == "white_balance"
        assert operations[2][0] == "demosaic"
        
        # Pipeline with configuration
        operations = resolver.resolve_pipeline("load -> wb(reverse=true) -> demosaic")
        assert operations[1][1]["reverse"] is True
    
    def test_preset_pipelines(self):
        """Test preset pipeline resolution."""
        resolver = OperationResolver(OPERATION_REGISTRY)
        
        # Test a preset
        operations = resolver.resolve_pipeline("raw_to_rgb")
        assert len(operations) == 4
        assert operations[0][0] == "raw_loader"
        assert operations[-1][0] == "color_transform"
    
    def test_fuzzy_matching(self):
        """Test fuzzy matching of operation names."""
        resolver = OperationResolver(OPERATION_REGISTRY)
        
        # Should find "demosaic" even with slight typo
        matches = resolver._fuzzy_match("demosic")
        assert "demosaic" in matches
    
    def test_pipeline_validation(self):
        """Test pipeline string validation."""
        resolver = OperationResolver(OPERATION_REGISTRY)
        
        # Valid pipeline
        errors = resolver.validate_pipeline_string("load -> wb -> demosaic")
        assert len(errors) == 0
        
        # Invalid pipeline (incompatible operations)
        errors = resolver.validate_pipeline_string("load -> encoder")
        assert len(errors) > 0


class TestMetadataDependencyResolver:
    """Test the metadata dependency resolver."""
    
    def test_resolve_dependencies(self):
        """Test resolving metadata dependencies."""
        resolver = MetadataDependencyResolver(OPERATION_REGISTRY)
        
        # Initial metadata
        initial_metadata = {"filepath": "/path/to/image.raw"}
        
        # Pipeline that needs metadata
        operations = ["raw_loader", "white_balance"]
        
        # Resolve dependencies
        sources = resolver.resolve_dependencies(operations, initial_metadata)
        
        # white_balance needs white_balance metadata which raw_loader provides
        assert len(sources) == 0 or "white_balance" in sources
    
    def test_missing_metadata_detection(self):
        """Test detection of missing metadata."""
        resolver = MetadataDependencyResolver(OPERATION_REGISTRY)
        
        # No initial metadata
        initial_metadata = {}
        
        # raw_loader needs filepath
        with pytest.raises(ValueError) as excinfo:
            resolver.resolve_dependencies(["raw_loader"], initial_metadata)
        assert "filepath" in str(excinfo.value)
    
    def test_suggest_metadata_operations(self):
        """Test suggesting operations for missing metadata."""
        resolver = MetadataDependencyResolver(OPERATION_REGISTRY)
        
        # Find operation that generates white_balance metadata
        suggestions = resolver.suggest_metadata_operations(["white_balance"])
        assert "raw_loader" in suggestions


def test_global_registry_initialized():
    """Test that the global registry is properly initialized."""
    assert OPERATION_REGISTRY is not None
    
    # Check some default operations are registered
    assert OPERATION_REGISTRY.get_spec("raw_loader") is not None
    assert OPERATION_REGISTRY.get_spec("white_balance") is not None
    assert OPERATION_REGISTRY.get_spec("demosaic") is not None
    assert OPERATION_REGISTRY.get_spec("encoder") is not None
    assert OPERATION_REGISTRY.get_spec("decoder") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])