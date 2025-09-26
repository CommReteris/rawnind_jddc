# Function-Based Composable Pipeline Architecture - Complete Visual

Based on our comprehensive discussions and architectural refinements, here's the detailed visual representation of our
Function-Based Composable Pipeline Architecture:

## ### Core Architectural Diagram

```
Function-Based Composable Pipeline Architecture
================================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE PIPELINE SYSTEM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PipelineOperation (Abstract Base Class)                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ Universal Interface: (tensor, **kwargs) → (tensor, metadata)               │
│  │ +spec: OperationSpec                                                       │
│  │ +process_tensors(data, metadata, **kwargs)*                               │
│  │ +validate_inputs(inputs, metadata)                                         │
│  │ +__call__(data, metadata, **kwargs)                                       │
│  │ +operation_type: "trainable" | "non_trainable"                            │
│  │ +get_parameters() → Optional[torch.nn.Module]                             │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OperationPipeline (Main Execution Engine)                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +operations: List[PipelineOperation]                                       │
│  │ +metadata_history: List[Dict[str, Any]]                                    │
│  │ +__call__(data) → (processed_data, metadata_history)                      │
│  │ +get_trainable_operations() → List[torch.nn.Module]                       │
│  │ +execute_optimized(data, metadata)                                         │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SmartPipelineAssembler (Validation & Assembly)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +validate_pipeline_compatibility(config) → List[warnings]                 │
│  │ +suggest_missing_operations(input_type, target_type) → List[suggestions]  │
│  │ +generate_auto_fixes(warnings, config) → Dict[fixes]                      │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SINGLE COMPREHENSIVE OPERATION REGISTRY                     │
│                        (Grouped by Functional Intent)                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OPERATION_REGISTRY = {                                                        │
│                                                                                 │
│    'denoising_operations': {                                                   │
│      'utnet2': UTNet2Wrapper(),              # ML-based (PyTorch)             │
│      'bm3d': BM3DWrapper(),                  # Classical algorithm            │
│      'bilateral': BilateralWrapper(),        # Can use Kornia/OpenCV/custom   │
│      'nlmeans': NLMeansWrapper(),            # Classical                      │
│      'temporal_denoise': TemporalDenoiseWrapper(), # Multi-frame              │
│      'rawdenoise': RawDenoiseWrapper(),      # Raw domain specific            │
│    },                                                                          │
│                                                                                 │
│    'enhancement_operations': {                                                 │
│      'sharpen': SharpenWrapper(),            # Can use Kornia/OpenCV          │
│      'unsharp_mask': UnsharpWrapper(),       # Backend-agnostic               │
│      'adjust_contrast': ContrastWrapper(),   # Kornia/custom implementation   │
│      'gamma_correct': GammaWrapper(),        # Simple function                │
│      'tone_mapping': ToneMappingWrapper(),   # Complex algorithms             │
│      'filmicrgb': FilmicRGBWrapper(),        # Scene-referred processing      │
│    },                                                                          │
│                                                                                 │
│    'geometric_operations': {                                                   │
│      'rotate': RotateWrapper(),              # Kornia/OpenCV backend          │
│      'perspective': PerspectiveWrapper(),    # Kornia geometric transforms    │
│      'crop': CropWrapper(),                  # Simple tensor operations       │
│      'elastic_transform': ElasticWrapper(),  # Advanced Kornia operations     │
│      'thin_plate_spline': TPSWrapper(),      # Kornia geometry functions      │
│    },                                                                          │
│                                                                                 │
│    'color_processing_operations': {                                            │
│      'white_balance': WhiteBalanceWrapper(), # Raw processing specific        │
│      'colorin': ColorInWrapper(),            # ICC profile transformation     │
│      'colorout': ColorOutWrapper(),          # Output color management        │
│      'channelmixerrgb': ChannelMixerWrapper(), # Color grading                │
│      'rgb_to_lab': RGBToLabWrapper(),        # Kornia color conversions       │
│    },                                                                          │
│                                                                                 │
│    'burst_processing_operations': {                                            │
│      'hdr_merge': HDRMergeWrapper(),         # Multi-image → single           │
│      'focus_stack': FocusStackWrapper(),    # Multi-image → single + mask    │
│      'panorama_stitch': PanoramaWrapper(),  # Multi-image → single           │
│      'super_resolution': SuperResWrapper(), # Multi-frame enhancement         │
│    },                                                                          │
│                                                                                 │
│    'raw_processing_operations': {                                              │
│      'rawprepare': RawPrepareWrapper(),      # Sensor data preparation        │
│      'demosaic': DemosaicWrapper(),          # Bayer → RGB conversion         │
│      'hotpixel': HotPixelWrapper(),          # Sensor defect correction       │
│      'temperature': TemperatureWrapper(),    # White balance adjustment       │
│    },                                                                          │
│                                                                                 │
│    'quality_assessment_operations': {                                          │
│      'overexposure': OverexposureWrapper(),  # Clipping detection            │
│      'noise_estimation': NoiseWrapper(),     # Noise level analysis          │
│      'blur_detection': BlurWrapper(),        # Sharpness assessment          │
│      'exposure_analysis': ExposureWrapper(), # Dynamic range analysis        │
│    }                                                                           │
│  }                                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      REGISTRY PATTERN EXTENSIONS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ModelRegistry                  QualityChecksRegistry                          │
│  ┌─────────────────────┐        ┌──────────────────────────────────────┐       │
│  │ ML Model Factory    │        │ Assessment Pipeline Factory          │       │
│  │ ─────────────────── │        │ ──────────────────────────────────── │       │
│  │ +utnet2: UTNet2     │        │ +overexposure: check_overexposure    │       │
│  │ +utnet3: UTNet3     │        │ +underexposure: check_underexposure  │       │
│  │ +bm3d: BM3DDenoiser │        │ +noise_level: check_noise_level      │       │
│  │ +learned_denoise    │        │ +create_pipeline(config)             │       │
│  │ +get_model(name)    │        │ +apply_all_checks(image)             │       │
│  └─────────────────────┘        └──────────────────────────────────────┘       │
│                                                                                 │
│  PreprocessingRegistry          TrainingStrategyRegistry                       │
│  ┌─────────────────────┐        ┌──────────────────────────────────────┐       │
│  │ Raw Processing      │        │ Training Pattern Factory             │       │
│  │ ─────────────────── │        │ ──────────────────────────────────── │       │
│  │ +normalize          │        │ +supervised: SupervisedStrategy      │       │
│  │ +gamma_correction   │        │ +self_supervised: SelfSupStrategy    │       │
│  │ +white_balance      │        │ +adversarial: AdversarialStrategy    │       │
│  │ +demosaic          │        │ +multi_task: MultiTaskStrategy       │       │
│  │ +create_pipeline()  │        │ +create_strategy(name, operations)   │       │
│  └─────────────────────┘        └──────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        IMPLEMENTATION PATTERNS                                 │
│              (Frameworks that influence HOW modules are written)               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PyTorch Lightning Integration Pattern                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ • ImageProcessingTask(LightningModule) wraps OperationPipeline             │
│  │ • training_step() → pipeline(batch) → loss computation                     │
│  │ • configure_optimizers() → separate optimizers for each trainable op      │
│  │ • Callbacks: PipelineVisualization, QualityMetrics                        │
│  │ • Multi-GPU support, automatic mixed precision, distributed training      │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Hydra Configuration Pattern                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ • Hierarchical YAML configs for pipeline assembly                          │
│  │ • defaults: [pipeline: denoising, model: utnet2, training: adam]          │
│  │ • Command-line overrides: python train.py model=utnet3 data.batch_size=16 │
│  │ • Compose configurations at runtime with validation                        │
│  │ • Experiment tracking and reproducible research workflows                  │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Backend Library Integration (Implementation Details)                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ • Kornia: GPU-accelerated computer vision (65+ operations available)      │
│  │ • OpenCV: Classical computer vision algorithms                             │
│  │ • NumPy/SciPy: CPU-based numerical processing                             │
│  │ • PyTorch: Deep learning operations and tensor manipulation               │
│  │ • LibRaw/rawpy: RAW file format decoding and metadata extraction          │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```

## ### Key Architectural Principles

### **1. Universal Interface Principle**

Every operation, regardless of implementation (classical algorithm, ML model, Kornia function), conforms to the same
interface:

```python
def __call__(self, data: Union[torch.Tensor, List[torch.Tensor]],
             metadata: Dict[str, Any] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]
```

### **2. Function-Based Organization**

Operations are grouped by **what they do**, not **how they do it**:

- `bilateral` operation could use Kornia, OpenCV, or custom implementation
- `utnet2` and `bm3d` both live in `denoising_operations` despite different architectures
- `rotate` operation abstracts away whether it uses Kornia geometry or OpenCV transforms

### **3. Implementation Agnostic Pipeline**

The pipeline executor doesn't know or care about implementation details:

```python
pipeline = OperationPipeline([
    {'category': 'raw_processing_operations', 'operation': 'demosaic'},
    {'category': 'denoising_operations', 'operation': 'bilateral'},  # Could be Kornia
    {'category': 'enhancement_operations', 'operation': 'sharpen'},  # Could be OpenCV
    {'category': 'denoising_operations', 'operation': 'utnet2'},  # PyTorch model
])
```

### **4. Smart Validation System**

- Automatic compatibility checking between operation inputs/outputs
- Suggestions for missing operations (e.g., "Need `colorin` to convert RAW_4CH → RGB")
- Type safety validation (RGB → LAB → RGB conversion chains)

## ### Data Flow Architecture

```
Configuration (YAML/JSON)
         ↓
   Hydra Parser & Validation
         ↓
   SmartPipelineAssembler
   • Validates compatibility
   • Suggests missing ops
   • Creates operation sequence
         ↓
   OperationPipeline
   • Chains operations seamlessly
   • Handles multi-tensor inputs
   • Propagates metadata
         ↓
   Individual Operations
   • Classical algorithms (BM3D, bilateral)
   • ML models (UTNet2, learned denoise)
   • GPU operations (Kornia transforms)
   • Multi-image processing (HDR, focus stack)
         ↓
   Quality Assessment & Validation
         ↓
   Output (processed tensors + metadata)
```

## ### Integration with Your RawNind Project

### **Current Codebase Integration:**

```python
# Your existing DatasetConfig enhanced:
@dataclass
class EnhancedDatasetConfig(DatasetConfig):
    # Existing fields remain unchanged
    augmentations: List[str] = field(default_factory=lambda: ['flip', 'rotate'])
    preprocessing_steps: List[str] = field(default_factory=lambda: ['normalize', 'demosaic'])
    quality_checks: List[str] = field(default_factory=lambda: ['overexposure', 'noise_level'])

    # New pipeline-based fields
    pipeline_operations: List[Dict[str, Any]] = field(default_factory=list)
    training_strategy: str = 'supervised'
    model_registry_config: Dict[str, Dict] = field(default_factory=dict)


# Usage maintains your existing patterns:
config = EnhancedDatasetConfig(
    dataset_name="raw_bayer_dataset",
    pipeline_operations=[
        {'category': 'raw_processing_operations', 'operation': 'rawprepare'},
        {'category': 'denoising_operations', 'operation': 'utnet2'},
        {'category': 'enhancement_operations', 'operation': 'filmicrgb'},
    ]
)
```

## ### Benefits of This Architecture

### **1. Maintains Your Elegant Philosophy**

- **Single registry** organized by functional intent
- **Implementation hiding** - pipeline doesn't care if denoising uses ML or classical
- **Configuration-driven** - same YAML-based approach as your augmentations

### **2. Production-Ready Features**

- **65+ operations** from comprehensive research (darktable, vkdt, Kornia)
- **Multi-image processing** for HDR, focus stacking, panoramas
- **Smart validation** prevents configuration errors
- **Professional ML workflow** through Lightning integration

### **3. Extensibility**

- **Add new operations** to any category without changing core architecture
- **Switch backends** (Kornia → OpenCV → custom) without pipeline changes
- **Compose complex workflows** through simple configuration

### **4. Real-World Applicability**

- **Raw image processing** from Bayer patterns to final RGB
- **Burst processing** for computational photography
- **Quality assessment** for automated pipeline validation
- **Training integration** for ML-based operations

This architecture successfully transforms your elegant augmentations pipeline concept into a comprehensive,
production-ready system that handles everything from basic raw processing to advanced multi-image workflows while
maintaining the core principle: **the pipeline doesn't care about implementation details, only functional intent**.