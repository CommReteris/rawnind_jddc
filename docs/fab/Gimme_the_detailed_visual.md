# Function-Based Composable Pipeline Architecture - Complete Detailed Visual

Based on our comprehensive architectural development, here's the complete detailed visual representation of the
Function-Based Composable Pipeline Architecture:

## ### Complete System Architecture Diagram

```
Function-Based Composable Pipeline Architecture - Complete System
================================================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           CORE ARCHITECTURAL FOUNDATION                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PipelineOperation (Universal Abstract Base Class)                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ Universal Contract: (tensor|List[tensor], metadata, **kwargs) → (result, metadata) │
│  │                                                                             │
│  │ +spec: OperationSpec                                                       │
│  │ +metadata_cache: Dict[str, Any]                                            │
│  │ +process_tensors(data, metadata, **kwargs)*                               │
│  │ +validate_inputs(inputs, metadata) → bool                                 │
│  │ +__call__(data, metadata, **kwargs) → (result, metadata)                 │
│  │ +operation_type: "trainable" | "non_trainable"                            │
│  │ +get_parameters() → Optional[torch.nn.Module]                             │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OperationSpec (Operation Specification System)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +name: str                                                                 │
│  │ +supported_modes: List[ProcessingMode]                                     │
│  │   • SINGLE_IMAGE, BURST_PROCESSING, BATCH_PROCESSING, GROUP_PROCESSING    │
│  │ +input_types: List[InputOutputType]                                        │
│  │   • RAW_BAYER, RAW_4CH, RGB, LAB, GRAYSCALE, MULTI_EXPOSURE, MASK         │
│  │ +output_types: List[InputOutputType]                                       │
│  │ +input_count: Tuple[int, Optional[int]]  # (min, max)                     │
│  │ +output_count: int                                                         │
│  │ +requires_metadata: List[str]                                              │
│  │ +produces_metadata: List[str]                                              │
│  │ +constraints: Dict[str, Any]                                               │
│  │ +description: str                                                          │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DUAL INTERFACE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Simple Interface (Beginner-Friendly)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ # Ultra-simple configuration                                               │
│  │ simple_pipeline = ['denoise', 'sharpen', 'tone_map']                      │
│  │                                                                             │
│  │ # Simple with parameters                                                   │
│  │ simple_pipeline = [                                                        │
│  │     {'operation': 'denoise', 'strength': 0.3},                           │
│  │     {'operation': 'sharpen', 'amount': 1.2},                             │
│  │     {'operation': 'tone_map', 'method': 'filmic'}                         │
│  │ ]                                                                          │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Low-Level Interface (Power Users)                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ # Full specification with explicit control                                │
│  │ advanced_pipeline = [                                                      │
│  │     {                                                                      │
│  │         'operation': 'utnet2',                                            │
│  │         'category': 'denoising_operations',                               │
│  │         'params': {'checkpoint': 'models/utnet2_best.pth'},               │
│  │         'constraints': {'gpu_memory': '8GB'},                             │
│  │         'metadata_requirements': ['noise_profile']                        │
│  │     }                                                                      │
│  │ ]                                                                          │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Operation Resolver (Auto-Resolution System)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +resolve_operation(name) → (category, full_spec)                          │
│  │ +operation_lookup: Dict[str, Tuple[str, str]]                             │
│  │   • 'denoise' → ('denoising_operations', 'bilateral')                    │
│  │   • 'sharpen' → ('enhancement_operations', 'unsharp_mask')               │
│  │   • 'tone_map' → ('tone_mapping_operations', 'sigmoid')                  │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPREHENSIVE OPERATION REGISTRY                            │
│                        (65+ Operations Grouped by Function)                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  COMPREHENSIVE_OPERATION_REGISTRY = {                                          │
│                                                                                 │
│    'raw_processing_operations': {                                              │
│      'rawprepare': RawPrepareWrapper(),      # Sensor data preparation        │
│      'hotpixels': HotPixelWrapper(),         # Sensor defect correction       │
│      'temperature': TemperatureWrapper(),    # White balance adjustment       │
│      'rawdenoise': RawDenoiseWrapper(),      # Raw domain noise reduction     │
│      'demosaic': DemosaicWrapper(),          # Bayer → RGB (AMaZE, VNG, etc.) │
│    },                                                                          │
│                                                                                 │
│    'color_processing_operations': {                                            │
│      'colorin': ColorInWrapper(),            # ICC profile transformation     │
│      'colorout': ColorOutWrapper(),          # Output color management        │
│      'channelmixerrgb': ChannelMixerWrapper(), # RGB channel mixing           │
│      'colorbalancergb': ColorBalanceWrapper(), # RGB-aware color balance      │
│      'primaries': PrimariesWrapper(),        # Color primaries adjustment     │
│    },                                                                          │
│                                                                                 │
│    'tone_mapping_operations': {                                                │
│      'exposure': ExposureWrapper(),          # Linear exposure compensation   │
│      'filmicrgb': FilmicRGBWrapper(),        # Filmic tone mapping           │
│      'sigmoid': SigmoidWrapper(),            # Sigmoid tone mapping          │
│      'toneequal': ToneEqualWrapper(),        # Tone equalizer                │
│      'highlights': HighlightsWrapper(),      # Highlight recovery            │
│    },                                                                          │
│                                                                                 │
│    'enhancement_operations': {                                                 │
│      'sharpen': SharpenWrapper(),            # Image sharpening              │
│      'diffuse': DiffuseWrapper(),            # Diffusion-based enhancement   │
│      'blurs': BlurWrapper(),                 # Edge-preserving blur          │
│      'defringe': DefringeWrapper(),          # Purple fringing removal       │
│      'ashift': AshiftWrapper(),              # Perspective correction        │
│    },                                                                          │
│                                                                                 │
│    'denoising_operations': {                                                   │
│      'utnet2': UTNet2Wrapper(),              # Deep learning denoiser        │
│      'bm3d': BM3DWrapper(),                  # Classical BM3D algorithm      │
│      'bilateral': BilateralWrapper(),        # Bilateral filtering           │
│      'nlmeans': NLMeansWrapper(),            # Non-local means denoising     │
│      'denoiseprofile': DenoiseProfileWrapper(), # Profile-based denoising   │
│    },                                                                          │
│                                                                                 │
│    'burst_processing_operations': {                                            │
│      'hdr_merge': HDRMergeWrapper(),         # Multi-exposure → single       │
│      'focus_stack': FocusStackWrapper(),    # Multi-focus → extended DOF     │
│      'panorama_stitch': PanoramaWrapper(),  # Multi-image → panorama        │
│      'temporal_denoise': TemporalDenoiseWrapper(), # Multi-frame NR          │
│      'super_resolution': SuperResWrapper(), # Multi-frame SR                │
│    },                                                                          │
│                                                                                 │
│    'geometric_operations': {                                                   │
│      'crop': CropWrapper(),                  # Cropping with guides          │
│      'flip': FlipWrapper(),                  # Horizontal/vertical flipping  │
│      'rotatepixels': RotatePixelsWrapper(),  # Pixel-level rotation         │
│      'scalepixels': ScalePixelsWrapper(),    # Scaling operations           │
│      'liquify': LiquifyWrapper(),            # Local geometric distortions   │
│    },                                                                          │
│                                                                                 │
│    'quality_assessment_operations': {                                          │
│      'overexposed': OverexposedWrapper(),    # Overexposure detection        │
│      'rawoverexposed': RawOverexposedWrapper(), # Raw-level overexposure    │
│      'noise_estimation': NoiseEstimationWrapper(), # Noise level analysis   │
│      'blur_detection': BlurDetectionWrapper(), # Sharpness assessment       │
│      'exposure_analysis': ExposureAnalysisWrapper(), # Dynamic range        │
│    },                                                                          │
│                                                                                 │
│    'creative_operations': {                                                    │
│      'grain': GrainWrapper(),                # Film grain simulation         │
│      'borders': BordersWrapper(),            # Border and frame effects      │
│      'watermark': WatermarkWrapper(),        # Watermark overlay            │
│      'vignette': VignetteWrapper(),          # Vignette effects             │
│      'bloom': BloomWrapper(),                # Bloom effects for highlights  │
│    }                                                                           │
│  }                                                                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         KORNIA INTEGRATION BACKEND                             │
│                    (65+ GPU-Accelerated Computer Vision Operations)            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Filter Operations                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +bilateral_filter, +gaussian_blur2d, +sobel, +laplacian                   │
│  │ +box_blur, +median_blur, +motion_blur, +unsharp_mask                      │
│  │ +canny, +spatial_gradient                                                 │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Color Operations                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +rgb_to_grayscale, +rgb_to_hsv, +hsv_to_rgb                              │
│  │ +rgb_to_lab, +lab_to_rgb, +rgb_to_yuv, +yuv_to_rgb                       │
│  │ +rgb_to_xyz, +xyz_to_rgb, +sepia                                          │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Enhancement Operations                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +adjust_brightness, +adjust_contrast, +adjust_gamma                       │
│  │ +adjust_hue, +adjust_saturation, +normalize, +denormalize                 │
│  │ +equalize_hist, +invert, +posterize, +sharpness, +solarize               │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Geometry Operations                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +rotate, +translate, +scale, +shear, +resize                             │
│  │ +crop_by_boxes, +center_crop, +crop_and_resize                           │
│  │ +hflip, +vflip, +warp_perspective, +warp_affine                          │
│  │ +elastic_transform2d, +thin_plate_spline                                  │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Augmentation Operations                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +random_crop, +random_resized_crop, +random_rotation, +random_affine      │
│  │ +random_perspective, +random_elastic_transform                            │
│  │ +color_jitter, +random_brightness, +random_contrast, +random_gamma        │
│  │ +random_gaussian_noise, +random_gaussian_blur, +random_motion_blur        │
│  │ +random_solarize, +random_posterize, +random_erasing                      │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Kornia Loss & Metrics Operations                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +ssim_loss, +ms_ssim_loss, +lpips_loss, +psnr_loss                       │
│  │ +total_variation, +focal_loss, +dice_loss, +tversky_loss                  │
│  │ +psnr, +ssim, +ms_ssim, +lpips, +mean_iou, +accuracy                     │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      REGISTRY PATTERN EXTENSIONS                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ModelRegistry (ML Model Factory)                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +models: Dict[str, Type]                                                   │
│  │ • 'utnet2': UTNet2Wrapper()                                               │
│  │ • 'utnet3': UTNet3Wrapper()                                               │
│  │ • 'bm3d': BM3DWrapper()                                                   │
│  │ • 'learned_denoise': LearnedDenoiseNet()                                  │
│  │ • 'balle_encoder': BalleEncoderWrapper()                                  │
│  │ • 'balle_decoder': BalleDecoderWrapper()                                  │
│  │ +get_model(name, **params) → PipelineOperation                           │
│  │ +register_model(name, model_class)                                        │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  QualityChecksRegistry (Assessment Pipeline Factory)                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +checks: Dict[str, Callable]                                               │
│  │ • 'overexposure': check_overexposure                                      │
│  │ • 'underexposure': check_underexposure                                    │
│  │ • 'noise_level': check_noise_level                                        │
│  │ • 'dynamic_range': check_dynamic_range                                    │
│  │ • 'color_accuracy': check_color_accuracy                                  │
│  │ +create_quality_pipeline(config) → QualityChecksPipeline                 │
│  │ +apply_all_checks(image) → Dict[str, Any]                                │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PreprocessingRegistry (Raw Processing Pipeline)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +steps: Dict[str, Callable]                                                │
│  │ • 'normalize': normalize_image                                             │
│  │ • 'gamma_correction': gamma_correction                                     │
│  │ • 'white_balance': white_balance                                           │
│  │ • 'demosaic': demosaic_bilinear                                           │
│  │ +create_preprocessing_pipeline(config) → PreprocessingPipeline            │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  TrainingStrategyRegistry (Training Pattern Factory)                           │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +strategies: Dict[str, Type]                                               │
│  │ • 'supervised': SupervisedTrainingStrategy                                │
│  │ • 'self_supervised': SelfSupervisedStrategy                               │
│  │ • 'adversarial': AdversarialTrainingStrategy                              │
│  │ • 'multi_task': MultiTaskTrainingStrategy                                 │
│  │ • 'few_shot': FewShotTrainingStrategy                                     │
│  │ +create_strategy(name, operations) → TrainingStrategy                     │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE EXECUTION SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OperationPipeline (Main Execution Engine)                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +operations: List[PipelineOperation]                                       │
│  │ +metadata_history: List[Dict[str, Any]]                                    │
│  │ +execution_plan: Dict[str, Any]                                            │
│  │ +__init__(config: List[Dict[str, Any]])                                   │
│  │ +__call__(data) → (processed_data, metadata_history)                      │
│  │ +get_trainable_operations() → List[Tuple[str, torch.nn.Module]]           │
│  │ +execute_optimized(data, metadata) → (result, metadata)                   │
│  │ +_create_operation(config) → PipelineOperation                            │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  SmartPipelineAssembler (Validation & Assembly)                               │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +registry: Dict[str, Dict[str, OperationSpec]]                            │
│  │ +validate_pipeline_compatibility(config) → List[warnings]                 │
│  │ +suggest_missing_operations(input_type, target_type) → List[suggestions]  │
│  │ +generate_auto_fixes(warnings, config) → Dict[fixes]                      │
│  │ +_get_operation_spec(op_config) → OperationSpec                           │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MetadataDependencyResolver (Dependency Analysis)                              │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +operations: List[OperationSpec]                                           │
│  │ +dependency_graph: Dict[str, Any]                                          │
│  │ +validate_pipeline(initial_metadata) → List[warnings]                     │
│  │ +suggest_providers(missing_metadata) → List[providers]                    │
│  │ +_build_dependency_graph() → Dict[str, Any]                               │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  OptimizedPipelineExecutor (Performance Optimizations)                        │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +operations: List[PipelineOperation]                                       │
│  │ +execution_plan: Dict[str, Any]                                            │
│  │   • batching_groups: List[List[int]]                                      │
│  │   • memory_checkpoints: List[int]                                         │
│  │   • device_assignments: Dict[int, str]                                    │
│  │   • parallelization_opportunities: List[int]                             │
│  │ +_create_execution_plan() → Dict[str, Any]                                │
│  │ +execute_optimized(data, metadata) → (result, metadata)                   │
│  │ +_execute_batched_group(group, data, metadata)                            │
│  │ +_execute_with_memory_management(operation, data, metadata)               │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      FRAMEWORK INTEGRATION PATTERNS                            │
│              (How External Frameworks Influence Module Design)                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  PyTorch Lightning Integration Pattern                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ ImageProcessingTask(LightningModule):                                      │
│  │   +pipeline: OperationPipeline                                             │
│  │   +loss_functions: List[Tuple[Callable, float]]                          │
│  │   +training_step(batch, batch_idx) → Dict[str, Any]                       │
│  │   +validation_step(batch, batch_idx) → Dict[str, Any]                     │
│  │   +configure_optimizers() → List[torch.optim.Optimizer]                  │
│  │   +_compute_adaptive_loss(outputs, targets, metadata) → torch.Tensor     │
│  │                                                                             │
│  │ Production Callbacks:                                                       │
│  │   • PipelineVisualizationCallback - saves intermediate results            │
│  │   • QualityMetricsCallback - logs quality assessments                     │
│  │   • Multi-GPU support, automatic mixed precision, distributed training    │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Hydra Configuration Pattern                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ LayeredConfigurationSystem:                                                │
│  │   +registry: Dict                                                          │
│  │   +operation_resolver: OperationResolver                                   │
│  │   +resolve_simple_config(simple_config) → List[Dict]                      │
│  │   +validate_and_suggest(config) → Dict[str, Any]                          │
│  │                                                                             │
│  │ HydraPipelineFactory:                                                      │
│  │   +create_pipeline_from_config(config_name, overrides)                    │
│  │   +_initialize_config_store()                                             │
│  │   +_compose_configuration(config_name, overrides)                         │
│  │                                                                             │
│  │ Configuration Structure:                                                    │
│  │   • conf/pipeline/scene_referred_workflow.yaml                            │
│  │   • conf/model/utnet2.yaml                                                │
│  │   • conf/training/adam_scheduler.yaml                                     │
│  │   • Command-line overrides: python train.py model=utnet3 data.batch_size=16 │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  AdaptiveOperationTrainer (Task-Agnostic Training)                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ +operation: PipelineOperation                                              │
│  │ +batch_strategy: str                                                       │
│  │   • "multi_image_grouping" for burst operations                           │
│  │   • "temporal_batching" for sequence operations                           │
│  │   • "standard_batching" for single-image operations                       │
│  │ +loss_strategy: str                                                        │
│  │ +_determine_batch_strategy() → str                                         │
│  │ +_create_training_batch(dataset_batch) → batch                           │
│  │ +_compute_adaptive_loss(outputs, targets, metadata) → torch.Tensor       │
│  │ +train(epochs) → None                                                     │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                        RAWNIND PROJECT INTEGRATION                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Enhanced DatasetConfig Integration                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ @dataclass                                                                 │
│  │ class EnhancedDatasetConfig(DatasetConfig):                               │
│  │     # Existing fields remain unchanged                                     │
│  │     augmentations: List[str] = field(default_factory=list)               │
│  │     preprocessing_steps: List[str] = field(default_factory=list)          │
│  │     quality_checks: List[str] = field(default_factory=list)               │
│  │                                                                             │
│  │     # New dual interface field                                             │
│  │     pipeline_operations: Union[List[str], List[Dict]] = field(default_factory=list) │
│  │     training_strategy: str = 'supervised'                                  │
│  │     model_registry_config: Dict[str, Dict] = field(default_factory=dict)  │
│  │                                                                             │
│  │ Usage Examples:                                                            │
│  │   # Simple interface                                                       │
│  │   config = EnhancedDatasetConfig(                                         │
│  │       pipeline_operations=['rawprepare', 'denoise', 'tone_map']           │
│  │   )                                                                        │
│  │                                                                             │
│  │   # Advanced interface                                                     │
│  │   config = EnhancedDatasetConfig(                                         │
│  │       pipeline_operations=[                                               │
│  │           {'operation': 'rawprepare', 'category': 'raw_processing_operations'}, │
│  │           {'operation': 'utnet2', 'category': 'denoising_operations'},    │
│  │           {'operation': 'filmicrgb', 'category': 'tone_mapping_operations'} │
│  │       ]                                                                    │
│  │   )                                                                        │
│  └─────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Scene-Referred Workflow Integration                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┤
│  │ Standard Raw Processing Pipeline:                                          │
│  │   1. rawprepare → sensor data preparation                                 │
│  │   2. temperature → white balance adjustment                               │
│  │   3. demosaic → Bayer pattern to RGB conversion                           │
│  │   4. colorin → input color profile transformation                         │
│  │   5. exposure → linear exposure compensation                              │
│  │   6. filmicrgb → scene-referred tone mapping                             │
│  │   7. colorbalancergb → color balance and grading                         │
│  │   8. colorout → output color profile transformation                       │
│  └─────────────────────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────────────────────┘
```

## ### Data Flow Architecture

```
Configuration Flow:
==================
YAML/JSON Config → Hydra Parser → Operation Resolver → SmartPipelineAssembler
                                      ↓
Simple Interface: ['denoise', 'sharpen'] → Auto-resolved to full specs
Advanced Interface: Full operation specifications → Direct validation
                                      ↓
Pipeline Assembly → Compatibility validation → Dependency resolution → Execution plan

Execution Flow:
==============
Input Data → OperationPipeline → Individual Operations → Output + Metadata
    ↓              ↓                      ↓
Multi-tensor   Optimized batching    Classical/ML/Kornia
processing     Memory management     Backend abstraction
    ↓              ↓                      ↓
Quality        Performance           Universal interface
Assessment     Monitoring            Implementation hiding

Training Flow:
=============
Pipeline → ImageProcessingTask → Lightning Trainer → Multi-GPU/AMP
    ↓           ↓                     ↓
Trainable   Adaptive loss         Professional ML
Operations  computation           workflow patterns
    ↓           ↓                     ↓
Model       Quality metrics       Experiment tracking
Registry    callbacks             and reproducibility
```

## ### Key Architectural Benefits

### **1. Universal Abstraction**

- Every operation follows the same interface regardless of implementation
- Classical algorithms (BM3D), ML models (UTNet2), and GPU operations (Kornia) work identically
- Pipeline executor is completely implementation-agnostic

### **2. Dual Interface System**

- **Simple**: `['denoise', 'sharpen', 'tone_map']` for beginners
- **Advanced**: Full specifications with constraints and metadata for power users
- **Progressive disclosure**: Users can start simple and add complexity gradually

### **3. Comprehensive Operation Coverage**

- **65+ operations** covering complete raw development workflows
- **Multi-image processing** for HDR, focus stacking, panoramas, super-resolution
- **Quality assessment** for automated pipeline validation
- **Creative effects** for artistic workflows

### **4. Production-Ready Architecture**

- **Smart validation** with compatibility checking and auto-fixes
- **Performance optimization** with batching and memory management
- **Professional ML workflows** through Lightning integration
- **Experiment management** through Hydra configuration system
- **GPU acceleration** through Kornia backend integration

### **5. RawNind Integration**

- **Seamless integration** with existing DatasetConfig structure
- **Backward compatibility** with current augmentations pipeline
- **Enhancement of existing patterns** rather than replacement
- **Natural evolution** of the elegant registry-based approach

This complete architecture successfully transforms the elegant augmentations pipeline concept into a comprehensive,
production-ready system that handles everything from basic raw processing to advanced computational photography
workflows while maintaining the core principle: **the pipeline doesn't care about implementation details, only
functional intent**.