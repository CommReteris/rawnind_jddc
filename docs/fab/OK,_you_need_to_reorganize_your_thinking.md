### Corrected Function-Based Composable Pipeline Architecture

You're absolutely right about the architectural grouping issues. Here's the corrected view organized by logical
packaging:

#### ### Core Architecture (Actual Modules)

**Main Pipeline System:**

- `PipelineOperation` (abstract base class)
- `OperationPipeline` (execution engine)
- `SmartPipelineAssembler` (validation/assembly)

**Single Operation Registry** (grouped by function, not implementation):

```python
OPERATION_REGISTRY = {
    'denoising_operations': {
        'utnet2': UTNet2Wrapper(),  # ML-based
        'bm3d': BM3DWrapper(),  # Classical
        'bilateral': BilateralWrapper(),  # Could use Kornia or OpenCV
        'nlmeans': NLMeansWrapper(),  # Classical
    },
    'enhancement_operations': {
        'sharpen': SharpenWrapper(),
        'unsharp_mask': UnsharpWrapper(),  # Could use Kornia
        'adjust_contrast': ContrastWrapper(),  # Could use Kornia
        'gamma_correct': GammaWrapper(),  # Could use Kornia
    },
    'geometric_operations': {
        'rotate': RotateWrapper(),  # Could use Kornia
        'perspective': PerspectiveWrapper(),  # Could use Kornia
        'crop': CropWrapper(),
    },
    # ... other functional categories
}
```

#### ### Operation Implementations (Implementation Details)

**Wrapper Classes** that adapt different backends:

- `UTNet2Wrapper` - uses PyTorch models
- `BilateralWrapper` - could use Kornia, OpenCV, or custom implementation
- `RotateWrapper` - could use Kornia geometry transforms
- `HDRMergeWrapper` - custom multi-image implementation

#### ### Framework Integration (Not Architectural Components)

**Implementation Patterns** (affect HOW modules are written):

- **PyTorch Lightning**: Affects training strategy implementations
- **Hydra**: Affects configuration loading/parsing
- **Kornia**: Backend library for GPU-accelerated operations
- **OpenCV**: Backend library for classical computer vision

#### ### Registry Extensions

Additional specialized registries:

- `ModelRegistry` - for ML model instantiation
- `QualityChecksRegistry` - for assessment pipelines
- `PreprocessingRegistry` - for preprocessing steps

#### ### Key Corrections

1. **Kornia operations integrated** into functional categories, not separate registry
2. **Frameworks shown as implementation details**, not architectural components
3. **Single operation registry** organized by functional intent
4. **Clear module boundaries** based on actual packaging logic

The architecture maintains the core principle: **operations are grouped by what they do, not how they're implemented**.