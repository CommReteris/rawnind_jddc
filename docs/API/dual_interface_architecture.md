# Function-Based Composable Pipeline - Dual Interface System

## Core Concept

The dual interface provides **two levels of abstraction** for the same operations:

### Simple Interface (Beginner-Friendly)

```python
# Ultra-simple - just operation names
simple_pipeline = ['denoise', 'sharpen', 'tone_map']

# Simple with parameters
simple_pipeline = [
    {'operation': 'denoise', 'strength': 0.3},
    {'operation': 'sharpen', 'amount': 1.2},
    {'operation': 'tone_map', 'method': 'filmic'}
]
```

### Low-Level Interface (Power Users)

```python
# Full specification with explicit control
advanced_pipeline = [
    {
        'operation': 'utnet2',
        'category': 'denoising_operations',
        'params': {'checkpoint': 'models/utnet2_best.pth'},
        'constraints': {'gpu_memory': '8GB'},
        'metadata_requirements': ['noise_profile'],
        'fallback_operation': 'bilateral'
    }
]
```

## Auto-Resolution System

```python
class OperationResolver:
    """Converts simple names to full specifications."""

    def __init__(self, registry: Dict):
        self.operation_lookup = {
            'denoise': ('denoising_operations', 'bilateral'),
            'sharpen': ('enhancement_operations', 'unsharp_mask'),
            'tone_map': ('tone_mapping_operations', 'sigmoid'),
            # ... more mappings
        }

    def resolve_operation(self, name: str) -> Tuple[str, OperationSpec]:
        """Simple name -> (category, full_spec)"""
        if name not in self.operation_lookup:
            raise ValueError(f"Unknown operation: {name}")
        category, full_name = self.operation_lookup[name]
        return category, self.registry[category][full_name]
```

## Integration with RawNind

```python
@dataclass
class EnhancedDatasetConfig(DatasetConfig):
    # Existing fields unchanged
    augmentations: List[str] = field(default_factory=lambda: ['flip', 'rotate'])

    # New dual interface field
    pipeline_operations: Union[List[str], List[Dict]] = field(default_factory=list)

    def create_pipeline(self) -> OperationPipeline:
        resolver = OperationResolver(OPERATION_REGISTRY)
        resolved_config = resolver.resolve_config(self.pipeline_operations)
        return OperationPipeline(resolved_config)


# Usage
config = EnhancedDatasetConfig(
    pipeline_operations=['rawprepare', 'denoise', 'tone_map']  # Simple!
)
```

## Key Benefits

1. **Natural Learning Curve** - Start simple, add complexity gradually
2. **Backward Compatibility** - Existing configs continue working
3. **Same Operations** - Complexity only in configuration, not architecture
4. **Progressive Disclosure** - Advanced features don't overwhelm beginners

The dual interface maintains the elegance of your augmentations pipeline while enabling sophisticated workflows.