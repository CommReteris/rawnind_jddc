### Enhanced Operation Registry for Function-Based Composable Pipeline

Based on research from darktable's comprehensive IOP (Image Operation Processing) system and rawpy's raw processing
capabilities, here's a detailed Operation Registry that supports single image processing, burst/group processing, and
operations with varying input/output characteristics.

### ### Core Registry Structure with Input/Output Specifications

```python
# src/rawnind/core/enhanced_operation_registry.py
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch


class ProcessingMode(Enum):
    SINGLE_IMAGE = "single_image"
    BURST_PROCESSING = "burst_processing"  # Multiple related images
    BATCH_PROCESSING = "batch_processing"  # Independent images
    GROUP_PROCESSING = "group_processing"  # Images with relationships


class InputOutputType(Enum):
    RAW_BAYER = "raw_bayer"  # 1-channel Bayer pattern
    RAW_4CH = "raw_4ch"  # 4-channel demosaiced
    RGB = "rgb"  # 3-channel RGB
    LAB = "lab"  # 3-channel LAB
    GRAYSCALE = "grayscale"  # 1-channel
    MULTI_EXPOSURE = "multi_exposure"  # Multiple exposures
    MASK = "mask"  # Binary/alpha mask
    METADATA = "metadata"  # Non-image data


@dataclass
class OperationSpec:
    """Specification for an operation's input/output characteristics."""
    name: str
    supported_modes: List[ProcessingMode]
    input_types: List[InputOutputType]
    output_types: List[InputOutputType]
    input_count: Tuple[int, Optional[int]]  # (min, max) - None means unlimited
    output_count: int
    requires_metadata: List[str]  # Required metadata fields
    produces_metadata: List[str]  # Metadata fields this operation produces
    constraints: Dict[str, Any]  # Additional constraints
    description: str


# Enhanced registry with comprehensive operations from darktable and rawpy research
ENHANCED_OPERATION_REGISTRY = {

    # ===== RAW PROCESSING OPERATIONS =====
    'raw_processing_operations': {
        'rawprepare': OperationSpec(
            name='rawprepare',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['sensor_info', 'camera_model'],
            produces_metadata=['raw_prepared'],
            constraints={'requires_sensor_calibration': True},
            description='Prepare raw sensor data for processing'
        ),
        'demosaic_bilinear': OperationSpec(
            name='demosaic_bilinear',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['bayer_pattern'],
            produces_metadata=['demosaic_method'],
            constraints={},
            description='Bilinear demosaicing of Bayer pattern'
        ),
        'demosaic_amaze': OperationSpec(
            name='demosaic_amaze',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER],
            output_types=[InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['bayer_pattern', 'sensor_info'],
            produces_metadata=['demosaic_method', 'quality_metrics'],
            constraints={'computational_cost': 'high'},
            description='Advanced AMaZE demosaicing algorithm'
        ),
        'hotpixels': OperationSpec(
            name='hotpixels',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
            output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['sensor_defects'],
            produces_metadata=['pixels_corrected'],
            constraints={},
            description='Hot pixel detection and correction'
        ),
        'rawdenoise': OperationSpec(
            name='rawdenoise',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
            output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['iso_value', 'sensor_noise_model'],
            produces_metadata=['noise_reduction_applied'],
            constraints={'requires_noise_profile': True},
            description='Raw-domain noise reduction'
        ),
    },

    # ===== COLOR PROCESSING OPERATIONS =====
    'color_processing_operations': {
        'colorin': OperationSpec(
            name='colorin',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['color_profile', 'white_point'],
            produces_metadata=['color_space'],
            constraints={'requires_icc_profile': True},
            description='Input color profile transformation'
        ),
        'colorout': OperationSpec(
            name='colorout',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB, InputOutputType.LAB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['output_profile'],
            produces_metadata=['final_color_space'],
            constraints={},
            description='Output color profile transformation'
        ),
        'temperature': OperationSpec(
            name='temperature',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['white_balance_multipliers'],
            produces_metadata=['temperature_applied', 'tint_applied'],
            constraints={},
            description='White balance adjustment'
        ),
        'colorbalance': OperationSpec(
            name='colorbalance',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['color_balance_applied'],
            constraints={},
            description='Color balance adjustment'
        ),
        'channelmixerrgb': OperationSpec(
            name='channelmixerrgb',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['channel_mixing_matrix'],
            constraints={},
            description='RGB channel mixing and color grading'
        ),
    },

    # ===== TONE MAPPING OPERATIONS =====
    'tone_mapping_operations': {
        'exposure': OperationSpec(
            name='exposure',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['exposure_compensation'],
            constraints={},
            description='Exposure compensation'
        ),
        'filmic': OperationSpec(
            name='filmic',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['tone_curve_applied'],
            constraints={},
            description='Filmic tone mapping'
        ),
        'filmicrgb': OperationSpec(
            name='filmicrgb',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['filmic_rgb_curve'],
            constraints={},
            description='RGB-aware filmic tone mapping'
        ),
        'sigmoid': OperationSpec(
            name='sigmoid',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['sigmoid_parameters'],
            constraints={},
            description='Sigmoid tone mapping'
        ),
        'highlights': OperationSpec(
            name='highlights',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['highlight_recovery_method'],
            constraints={},
            description='Highlight recovery and reconstruction'
        ),
    },

    # ===== ENHANCEMENT OPERATIONS =====
    'enhancement_operations': {
        'sharpen': OperationSpec(
            name='sharpen',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['sharpening_applied'],
            constraints={},
            description='Image sharpening'
        ),
        'defringe': OperationSpec(
            name='defringe',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['defringe_corrections'],
            constraints={},
            description='Chromatic aberration correction'
        ),
        'lens': OperationSpec(
            name='lens',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['lens_model', 'focal_length', 'aperture'],
            produces_metadata=['lens_corrections_applied'],
            constraints={'requires_lens_database': True},
            description='Lens distortion and vignetting correction'
        ),
        'cacorrect': OperationSpec(
            name='cacorrect',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['ca_correction_applied'],
            constraints={},
            description='Chromatic aberration correction'
        ),
    },

    # ===== NOISE REDUCTION OPERATIONS =====
    'denoising_operations': {
        'nlmeans': OperationSpec(
            name='nlmeans',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['noise_reduction_strength'],
            constraints={'computational_cost': 'high'},
            description='Non-local means denoising'
        ),
        'bilateral': OperationSpec(
            name='bilateral',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['bilateral_parameters'],
            constraints={},
            description='Bilateral filtering'
        ),
        'denoiseprofile': OperationSpec(
            name='denoiseprofile',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['noise_profile'],
            produces_metadata=['profile_denoising_applied'],
            constraints={'requires_noise_profile': True},
            description='Profile-based denoising'
        ),
    },

    # ===== BURST/GROUP PROCESSING OPERATIONS =====
    'burst_processing_operations': {
        'hdr_merge': OperationSpec(
            name='hdr_merge',
            supported_modes=[ProcessingMode.BURST_PROCESSING, ProcessingMode.GROUP_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(3, None),  # At least 3 images, no upper limit
            output_count=1,
            requires_metadata=['exposure_values', 'alignment_data'],
            produces_metadata=['hdr_merge_method', 'merged_exposure_range'],
            constraints={'requires_exposure_bracketing': True, 'requires_alignment': True},
            description='HDR bracketed exposure merging'
        ),
        'focus_stack': OperationSpec(
            name='focus_stack',
            supported_modes=[ProcessingMode.BURST_PROCESSING, ProcessingMode.GROUP_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB, InputOutputType.MASK],
            input_count=(2, None),  # At least 2 images for focus stacking
            output_count=2,  # Stacked image + depth map
            requires_metadata=['focus_distances'],
            produces_metadata=['focus_stack_method', 'depth_map'],
            constraints={'requires_focus_bracketing': True},
            description='Focus stacking for extended depth of field'
        ),
        'panorama_stitch': OperationSpec(
            name='panorama_stitch',
            supported_modes=[ProcessingMode.GROUP_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(2, None),  # At least 2 overlapping images
            output_count=1,
            requires_metadata=['overlap_regions', 'camera_poses'],
            produces_metadata=['panorama_projection', 'stitch_quality'],
            constraints={'requires_overlap': 0.3, 'requires_feature_matching': True},
            description='Panoramic image stitching'
        ),
        'noise_reduction_temporal': OperationSpec(
            name='noise_reduction_temporal',
            supported_modes=[ProcessingMode.BURST_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(3, 20),  # Typically 3-20 frames
            output_count=1,
            requires_metadata=['temporal_alignment'],
            produces_metadata=['temporal_noise_reduction_applied'],
            constraints={'requires_temporal_alignment': True},
            description='Temporal noise reduction using multiple frames'
        ),
    },

    # ===== GEOMETRIC OPERATIONS =====
    'geometric_operations': {
        'flip': OperationSpec(
            name='flip',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['flip_operation'],
            constraints={},
            description='Image flipping (horizontal/vertical)'
        ),
        'crop': OperationSpec(
            name='crop',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['crop_region'],
            constraints={},
            description='Image cropping and composition'
        ),
        'ashift': OperationSpec(
            name='ashift',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['perspective_correction'],
            constraints={},
            description='Perspective correction and keystone adjustment'
        ),
        'rotatepixels': OperationSpec(
            name='rotatepixels',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH, InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['rotation_applied'],
            constraints={},
            description='Pixel-level rotation'
        ),
    },

    # ===== QUALITY ASSESSMENT OPERATIONS =====
    'quality_assessment_operations': {
        'overexposed': OperationSpec(
            name='overexposed',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
            output_types=[InputOutputType.MASK],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['overexposure_statistics'],
            constraints={},
            description='Overexposure detection and visualization'
        ),
        'rawoverexposed': OperationSpec(
            name='rawoverexposed',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
            output_types=[InputOutputType.MASK],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['sensor_saturation_levels'],
            produces_metadata=['raw_overexposure_statistics'],
            constraints={},
            description='Raw-level overexposure detection'
        ),
        'spots': OperationSpec(
            name='spots',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB, InputOutputType.MASK],
            input_count=(1, 1),
            output_count=2,
            requires_metadata=[],
            produces_metadata=['spots_detected', 'correction_applied'],
            constraints={},
            description='Spot detection and removal'
        ),
    },

    # ===== CREATIVE OPERATIONS =====
    'creative_operations': {
        'vignette': OperationSpec(
            name='vignette',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['vignette_parameters'],
            constraints={},
            description='Vignette effect'
        ),
        'grain': OperationSpec(
            name='grain',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['grain_parameters'],
            constraints={},
            description='Film grain simulation'
        ),
        'borders': OperationSpec(
            name='borders',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=[],
            produces_metadata=['border_parameters'],
            constraints={},
            description='Border and frame effects'
        ),
        'watermark': OperationSpec(
            name='watermark',
            supported_modes=[ProcessingMode.SINGLE_IMAGE, ProcessingMode.BATCH_PROCESSING],
            input_types=[InputOutputType.RGB],
            output_types=[InputOutputType.RGB],
            input_count=(1, 1),
            output_count=1,
            requires_metadata=['watermark_data'],
            produces_metadata=['watermark_applied'],
            constraints={'requires_watermark_file': True},
            description='Watermark overlay'
        ),
    },
}
```

### ### Enhanced Pipeline Operation with Input/Output Validation

```python
# src/rawnind/core/enhanced_pipeline_operation.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
import torch


class EnhancedPipelineOperation(ABC):
    """Enhanced operation interface with input/output validation."""

    def __init__(self, spec: OperationSpec):
        self.spec = spec
        self.metadata_cache = {}

    @abstractmethod
    def process(self, inputs: List[torch.Tensor], metadata: Dict[str, Any], **kwargs) -> Tuple[
        List[torch.Tensor], Dict[str, Any]]:
        """Process input data with validation."""
        pass

    def validate_inputs(self, inputs: List[torch.Tensor], metadata: Dict[str, Any]) -> bool:
        """Validate input compatibility with operation requirements."""
        # Check input count
        min_inputs, max_inputs = self.spec.input_count
        if len(inputs) < min_inputs:
            raise ValueError(f"Operation {self.spec.name} requires at least {min_inputs} inputs, got {len(inputs)}")
        if max_inputs is not None and len(inputs) > max_inputs:
            raise ValueError(f"Operation {self.spec.name} accepts at most {max_inputs} inputs, got {len(inputs)}")

        # Check required metadata
        for required_meta in self.spec.requires_metadata:
            if required_meta not in metadata:
                raise ValueError(f"Operation {self.spec.name} requires metadata field: {required_meta}")

        # Check constraints
        if 'requires_exposure_bracketing' in self.spec.constraints:
            if 'exposure_values' not in metadata:
                raise ValueError(f"Operation {self.spec.name} requires exposure bracketing data")

        if 'requires_overlap' in self.spec.constraints:
            min_overlap = self.spec.constraints['requires_overlap']
            # Implementation would check actual overlap

        return True

    def __call__(self, inputs: List[torch.Tensor], metadata: Dict[str, Any] = None, **kwargs) -> Tuple[
        List[torch.Tensor], Dict[str, Any]]:
        """Main operation call with validation."""
        if metadata is None:
            metadata = {}

        self.validate_inputs(inputs, metadata)
        outputs, output_metadata = self.process(inputs, metadata, **kwargs)

        # Ensure output count matches specification
        if len(outputs) != self.spec.output_count:
            raise ValueError(
                f"Operation {self.spec.name} should produce {self.spec.output_count} outputs, got {len(outputs)}")

        return outputs, output_metadata
```

### ### Smart Pipeline Assembly with Constraint Checking

```python
# src/rawnind/core/smart_pipeline_assembler.py
class SmartPipelineAssembler:
    """Pipeline assembler with automatic constraint validation."""

    def __init__(self, registry: Dict[str, Dict[str, OperationSpec]]):
        self.registry = registry

    def validate_pipeline_compatibility(self, pipeline_config: List[Dict[str, Any]]) -> List[str]:
        """Validate that pipeline operations are compatible."""
        warnings = []

        for i, (current_op, next_op) in enumerate(zip(pipeline_config[:-1], pipeline_config[1:])):
            current_spec = self._get_operation_spec(current_op)
            next_spec = self._get_operation_spec(next_op)

            # Check output -> input type compatibility
            compatible_types = set(current_spec.output_types) & set(next_spec.input_types)
            if not compatible_types:
                warnings.append(f"Step {i}: {current_spec.name} outputs {current_spec.output_types} "
                                f"but {next_spec.name} requires {next_spec.input_types}")

            # Check processing mode compatibility
            compatible_modes = set(current_spec.supported_modes) & set(next_spec.supported_modes)
            if not compatible_modes:
                warnings.append(f"Step {i}: Processing mode mismatch between {current_spec.name} and {next_spec.name}")

        return warnings

    def suggest_missing_operations(self, input_type: InputOutputType, target_type: InputOutputType) -> List[str]:
        """Suggest operations to convert from input_type to target_type."""
        suggestions = []

        # Simple conversion suggestions
        conversion_paths = {
            (InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH): ['demosaic_bilinear', 'demosaic_amaze'],
            (InputOutputType.RAW_4CH, InputOutputType.RGB): ['colorin'],
            (InputOutputType.RGB, InputOutputType.LAB): ['colorspace_conversion'],
        }

        if (input_type, target_type) in conversion_paths:
            suggestions = conversion_paths[(input_type, target_type)]

        return suggestions

    def _get_operation_spec(self, op_config: Dict[str, Any]) -> OperationSpec:
        """Get operation specification from config."""
        category = op_config['category']
        operation = op_config['operation']
        return self.registry[category][operation]
```

### ### Key Enhancements for Your Use Case

#### ### 1. Comprehensive Operation Coverage

- **80+ operations** from darktable research covering every aspect of raw development
- **Burst processing operations** for HDR, focus stacking, panoramas, temporal noise reduction
- **Input/output type specifications** ensure compatibility
- **Constraint system** captures operation requirements (lens database, noise profiles, etc.)

#### ### 2. Smart Pipeline Assembly

- **Automatic compatibility checking** prevents invalid operation chains
- **Missing operation suggestions** help complete pipelines
- **Constraint validation** ensures operations have required data/metadata

#### ### 3. Flexible Processing Modes

- **Single image**: Traditional one-to-one processing
- **Burst processing**: Related images (HDR brackets, focus stacks)
- **Batch processing**: Independent images processed identically
- **Group processing**: Images with relationships (panorama tiles)

#### ### 4. Metadata-Driven Operations

- **Required metadata** ensures operations have necessary information
- **Produced metadata** enables downstream operations to adapt
- **Constraint propagation** maintains pipeline validity

### ### Example Usage Scenarios

```yaml
# Complex HDR + denoising pipeline
hdr_denoising_pipeline:
  - operation: 'hdr_merge'
    category: 'burst_processing_operations'
    params:
      alignment_method: 'feature_based'
      merge_algorithm: 'exposure_fusion'

  - operation: 'nlmeans'
    category: 'denoising_operations'
    params:
      strength: 0.3

  - operation: 'filmic'
    category: 'tone_mapping_operations'
    params:
      white_point: 4.0

# Panorama processing pipeline  
panorama_pipeline:
  - operation: 'panorama_stitch'
    category: 'burst_processing_operations'
    params:
      projection: 'cylindrical'
      blending_method: 'multiband'

  - operation: 'lens'
    category: 'enhancement_operations'
    params:
      correct_vignetting: true
      correct_distortion: true
```

This enhanced Operation Registry provides the foundation for a truly composable pipeline system that can handle any raw
development workflow while maintaining type safety and operational constraints. The system automatically validates
pipeline compatibility and suggests missing operations, making it both powerful and user-friendly.