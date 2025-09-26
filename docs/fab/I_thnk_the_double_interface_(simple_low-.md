### Dual Interface System Retrieved

The dual interface concept provided **two levels of abstraction** for the same operations:

#### ### Simple Interface (Beginner-Friendly)

```python
# Ultra-simple - just operation names
simple_pipeline = ['denoise', 'sharpen', 'tone_map']

# Simple with parameters
simple_pipeline = [
    {'operation': 'denoise', 'strength': 0.3},
    {'operation': 'sharpen', 'amount': 1.2}
]
```

#### ### Low-Level Interface (Power Users)

```python
# Full specification with explicit control
advanced_pipeline = [
    {
        'operation': 'utnet2',
        'category': 'denoising_operations',
        'params': {'checkpoint': 'models/utnet2_best.pth'},
        'constraints': {'gpu_memory': '8GB'},
        'metadata_requirements': ['noise_profile']
    }
]
```

#### ### Auto-Resolution System

```python
class OperationResolver:
    def resolve_operation(self, name: str) -> Tuple[str, OperationSpec]:
        """Simple name -> (category, full_spec)"""
        operation_lookup = {
            'denoise': ('denoising_operations', 'bilateral'),
            'sharpen': ('enhancement_operations', 'unsharp_mask'),
            'tone_map': ('tone_mapping_operations', 'sigmoid')
        }
        category, full_name = operation_lookup[name]
        return category, self.registry[category][full_name]
```

#### ### Integration with RawNind

```python
@dataclass 
class EnhancedDatasetConfig(DatasetConfig):
    # New dual interface field
    pipeline_operations: Union[List[str], List[Dict]] = field(default_factory=list)

# Usage - maintains elegance
config = EnhancedDatasetConfig(
    pipeline_operations=['rawprepare', 'denoise', 'tone_map']  # Simple!
)
```

#### ### Key Benefits

1. **Natural Learning Curve** - Start simple, add complexity gradually
2. **Backward Compatibility** - Existing configs continue working
3. **Progressive Disclosure** - Advanced features don't overwhelm beginners
4. **Same Architecture** - Complexity only in configuration, not implementation

The dual interface maintained the elegance of your augmentations pipeline while enabling sophisticated workflows. Users
could start with one-line configs and naturally progress to full specifications without changing the underlying system.