### Complete Function-Based Composable Pipeline Architecture

Here's the comprehensive Function-Based Composable Pipeline Architecture with unnecessary qualifiers removed and
complete Kornia operation integration:

### ### Core Architecture Foundation

#### ### Universal Operation Interface

The system uses a universal interface that handles both single and multi-tensor operations:

```python
# src/rawnind/core/pipeline_operation.py
from typing import Union, List, Tuple, Dict, Any
import torch


class PipelineOperation(ABC):
    """Universal interface for all pipeline operations."""

    def __init__(self, spec: Optional[OperationSpec] = None):
        self.spec = spec or self._create_default_spec()
        self.metadata_cache = {}

    @abstractmethod
    def process_tensors(
            self,
            data: Union[torch.Tensor, List[torch.Tensor]],
            metadata: Dict[str, Any],
            **kwargs
    ) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Dict[str, Any]]:
        """Process data with universal interface."""
        pass

    def __call__(self, data, metadata=None, **kwargs):
        """Main entry point with automatic validation."""
        metadata = metadata or {}
        input_data = [data] if isinstance(data, torch.Tensor) else data
        self.validate_inputs(input_data, metadata)
        outputs, output_metadata = self.process_tensors(input_data, metadata, **kwargs)

        if self.spec.output_count == 1 and isinstance(outputs, list):
            outputs = outputs[0]
        return outputs, output_metadata
```

### ### Operation Registry (65+ Operations)

#### **Raw Processing Operations**

```python
'raw_processing_operations': {
    'rawprepare': OperationSpec(
        name='rawprepare',
        input_types=[InputOutputType.RAW_BAYER],
        output_types=[InputOutputType.RAW_4CH],
        requires_metadata=['sensor_info', 'camera_model'],
        description='Prepare raw sensor data, black level subtraction'
    ),
    'hotpixels': OperationSpec(
        name='hotpixels',
        input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
        output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
        requires_metadata=['sensor_defects'],
        description='Hot pixel detection and correction'
    ),
    'temperature': OperationSpec(
        name='temperature',
        input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        requires_metadata=['white_balance_multipliers'],
        description='White balance adjustment'
    ),
    'rawdenoise': OperationSpec(
        name='rawdenoise',
        input_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
        output_types=[InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH],
        requires_metadata=['iso_value', 'sensor_noise_model'],
        description='Raw domain noise reduction'
    ),
    'demosaic': OperationSpec(
        name='demosaic',
        input_types=[InputOutputType.RAW_BAYER],
        output_types=[InputOutputType.RAW_4CH],
        requires_metadata=['bayer_pattern'],
        description='Advanced demosaicing (AMaZE, VNG, etc.)'
    ),
}
```

#### **Color Processing Operations**

```python
'color_processing_operations': {
    'colorin': OperationSpec(
        name='colorin',
        input_types=[InputOutputType.RAW_4CH],
        output_types=[InputOutputType.RGB],
        requires_metadata=['color_profile', 'white_point'],
        description='Input color profile transformation'
    ),
    'colorout': OperationSpec(
        name='colorout',
        input_types=[InputOutputType.RGB, InputOutputType.LAB],
        output_types=[InputOutputType.RGB],
        requires_metadata=['output_profile'],
        description='Output color profile transformation'
    ),
    'channelmixerrgb': OperationSpec(
        name='channelmixerrgb',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='RGB channel mixer with color grading'
    ),
    'colorbalancergb': OperationSpec(
        name='colorbalancergb',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='RGB-aware color balance'
    ),
    'primaries': OperationSpec(
        name='primaries',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Color primaries adjustment for wide gamut'
    ),
}
```

#### **Scene-Referred Tone Mapping Operations**

```python  
'tone_mapping_operations': {
    'exposure': OperationSpec(
        name='exposure',
        input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        description='Linear exposure compensation with highlight protection'
    ),
    'filmicrgb': OperationSpec(
        name='filmicrgb',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Filmic tone mapping with RGB processing'
    ),
    'sigmoid': OperationSpec(
        name='sigmoid',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Sigmoid tone mapping for smooth transitions'
    ),
    'toneequal': OperationSpec(
        name='toneequal',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Tone equalizer for advanced tone control'
    ),
    'highlights': OperationSpec(
        name='highlights',
        input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        description='Highlight recovery and reconstruction'
    ),
}
```

### ### Comprehensive Kornia Operation Registry

#### **Kornia Filters Operations**

```python
'kornia_filter_operations': {
    'bilateral_filter': KorniaOperationWrapper(kornia.filters.bilateral),
    'gaussian_blur2d': KorniaOperationWrapper(kornia.filters.gaussian_blur2d),
    'sobel': KorniaOperationWrapper(kornia.filters.sobel),
    'laplacian': KorniaOperationWrapper(kornia.filters.laplacian),
    'box_blur': KorniaOperationWrapper(kornia.filters.box_blur),
    'median_blur': KorniaOperationWrapper(kornia.filters.median_blur),
    'motion_blur': KorniaOperationWrapper(kornia.filters.motion_blur),
    'unsharp_mask': KorniaOperationWrapper(kornia.filters.unsharp_mask),
    'canny': KorniaOperationWrapper(kornia.filters.canny),
    'spatial_gradient': KorniaOperationWrapper(kornia.filters.spatial_gradient),
}
```

#### **Kornia Color Operations**

```python
'kornia_color_operations': {
    'rgb_to_grayscale': KorniaOperationWrapper(kornia.color.rgb_to_grayscale),
    'rgb_to_hsv': KorniaOperationWrapper(kornia.color.rgb_to_hsv),
    'hsv_to_rgb': KorniaOperationWrapper(kornia.color.hsv_to_rgb),
    'rgb_to_lab': KorniaOperationWrapper(kornia.color.rgb_to_lab),
    'lab_to_rgb': KorniaOperationWrapper(kornia.color.lab_to_rgb),
    'rgb_to_yuv': KorniaOperationWrapper(kornia.color.rgb_to_yuv),
    'yuv_to_rgb': KorniaOperationWrapper(kornia.color.yuv_to_rgb),
    'rgb_to_xyz': KorniaOperationWrapper(kornia.color.rgb_to_xyz),
    'xyz_to_rgb': KorniaOperationWrapper(kornia.color.xyz_to_rgb),
    'sepia': KorniaOperationWrapper(kornia.color.sepia),
}
```

#### **Kornia Enhancement Operations**

```python
'kornia_enhance_operations': {
    'adjust_brightness': KorniaOperationWrapper(kornia.enhance.adjust_brightness),
    'adjust_contrast': KorniaOperationWrapper(kornia.enhance.adjust_contrast),
    'adjust_gamma': KorniaOperationWrapper(kornia.enhance.adjust_gamma),
    'adjust_hue': KorniaOperationWrapper(kornia.enhance.adjust_hue),
    'adjust_saturation': KorniaOperationWrapper(kornia.enhance.adjust_saturation),
    'normalize': KorniaOperationWrapper(kornia.enhance.normalize),
    'denormalize': KorniaOperationWrapper(kornia.enhance.denormalize),
    'equalize_hist': KorniaOperationWrapper(kornia.enhance.equalize_hist),
    'invert': KorniaOperationWrapper(kornia.enhance.invert),
    'posterize': KorniaOperationWrapper(kornia.enhance.posterize),
    'sharpness': KorniaOperationWrapper(kornia.enhance.sharpness),
    'solarize': KorniaOperationWrapper(kornia.enhance.solarize),
}
```

#### **Kornia Geometry Operations**

```python
'kornia_geometry_operations': {
    'rotate': KorniaOperationWrapper(kornia.geometry.rotate),
    'translate': KorniaOperationWrapper(kornia.geometry.translate),
    'scale': KorniaOperationWrapper(kornia.geometry.scale),
    'shear': KorniaOperationWrapper(kornia.geometry.shear),
    'resize': KorniaOperationWrapper(kornia.geometry.resize),
    'crop_by_boxes': KorniaOperationWrapper(kornia.geometry.crop_by_boxes),
    'center_crop': KorniaOperationWrapper(kornia.geometry.center_crop),
    'crop_and_resize': KorniaOperationWrapper(kornia.geometry.crop_and_resize),
    'hflip': KorniaOperationWrapper(kornia.geometry.hflip),
    'vflip': KorniaOperationWrapper(kornia.geometry.vflip),
    'warp_perspective': KorniaOperationWrapper(kornia.geometry.warp_perspective),
    'warp_affine': KorniaOperationWrapper(kornia.geometry.warp_affine),
    'elastic_transform2d': KorniaOperationWrapper(kornia.geometry.elastic_transform2d),
    'thin_plate_spline': KorniaOperationWrapper(kornia.geometry.thin_plate_spline),
}
```

#### **Kornia Camera and 3D Operations**

```python
'kornia_camera_operations': {
    'project_points': KorniaOperationWrapper(kornia.geometry.camera.project_points),
    'unproject_points': KorniaOperationWrapper(kornia.geometry.camera.unproject_points),
    'depth_to_3d': KorniaOperationWrapper(kornia.geometry.depth.depth_to_3d),
    'depth_to_normals': KorniaOperationWrapper(kornia.geometry.depth.depth_to_normals),
    'warp_frame_depth': KorniaOperationWrapper(kornia.geometry.depth.warp_frame_depth),
}
```

#### **Kornia Feature Detection Operations**

```python
'kornia_feature_operations': {
    'harris_response': KorniaOperationWrapper(kornia.feature.harris_response),
    'corner_detection': KorniaOperationWrapper(kornia.feature.corner_detection),
    'gftt_response': KorniaOperationWrapper(kornia.feature.gftt_response),
    'hessian_response': KorniaOperationWrapper(kornia.feature.hessian_response),
    'dog_response': KorniaOperationWrapper(kornia.feature.dog_response),
}
```

#### **Kornia Augmentation Operations**

```python
'kornia_augmentation_operations': {
    # Geometric augmentations
    'random_crop': KorniaAugmentationWrapper(kornia.augmentation.RandomCrop),
    'random_resized_crop': KorniaAugmentationWrapper(kornia.augmentation.RandomResizedCrop),
    'center_crop': KorniaAugmentationWrapper(kornia.augmentation.CenterCrop),
    'random_rotation': KorniaAugmentationWrapper(kornia.augmentation.RandomRotation),
    'random_affine': KorniaAugmentationWrapper(kornia.augmentation.RandomAffine),
    'random_perspective': KorniaAugmentationWrapper(kornia.augmentation.RandomPerspective),
    'random_elastic_transform': KorniaAugmentationWrapper(kornia.augmentation.RandomElasticTransform),
    'random_thin_plate_spline': KorniaAugmentationWrapper(kornia.augmentation.RandomThinPlateSpline),
    'random_horizontal_flip': KorniaAugmentationWrapper(kornia.augmentation.RandomHorizontalFlip),
    'random_vertical_flip': KorniaAugmentationWrapper(kornia.augmentation.RandomVerticalFlip),

    # Photometric augmentations
    'color_jitter': KorniaAugmentationWrapper(kornia.augmentation.ColorJitter),
    'random_brightness': KorniaAugmentationWrapper(kornia.augmentation.RandomBrightness),
    'random_contrast': KorniaAugmentationWrapper(kornia.augmentation.RandomContrast),
    'random_gamma': KorniaAugmentationWrapper(kornia.augmentation.RandomGamma),
    'random_hue': KorniaAugmentationWrapper(kornia.augmentation.RandomHue),
    'random_saturation': KorniaAugmentationWrapper(kornia.augmentation.RandomSaturation),
    'random_gaussian_noise': KorniaAugmentationWrapper(kornia.augmentation.RandomGaussianNoise),
    'random_gaussian_blur': KorniaAugmentationWrapper(kornia.augmentation.RandomGaussianBlur),
    'random_motion_blur': KorniaAugmentationWrapper(kornia.augmentation.RandomMotionBlur),
    'random_solarize': KorniaAugmentationWrapper(kornia.augmentation.RandomSolarize),
    'random_posterize': KorniaAugmentationWrapper(kornia.augmentation.RandomPosterize),
    'random_erasing': KorniaAugmentationWrapper(kornia.augmentation.RandomErasing),
}
```

#### **Kornia Loss Functions**

```python
'kornia_loss_operations': {
    'ssim_loss': KorniaLossWrapper(kornia.losses.SSIMLoss),
    'ms_ssim_loss': KorniaLossWrapper(kornia.losses.MS_SSIMLoss),
    'lpips_loss': KorniaLossWrapper(kornia.losses.LPIPSLoss),
    'psnr_loss': KorniaLossWrapper(kornia.losses.PSNRLoss),
    'total_variation': KorniaLossWrapper(kornia.losses.TotalVariation),
    'focal_loss': KorniaLossWrapper(kornia.losses.FocalLoss),
    'dice_loss': KorniaLossWrapper(kornia.losses.DiceLoss),
    'tversky_loss': KorniaLossWrapper(kornia.losses.TverskyLoss),
    'lovasz_hinge_loss': KorniaLossWrapper(kornia.losses.LovaszHingeLoss),
    'lovasz_softmax_loss': KorniaLossWrapper(kornia.losses.LovaszSoftmaxLoss),
}
```

#### **Kornia Metrics Operations**

```python
'kornia_metrics_operations': {
    'psnr': KorniaMetricWrapper(kornia.metrics.psnr),
    'ssim': KorniaMetricWrapper(kornia.metrics.ssim),
    'ms_ssim': KorniaMetricWrapper(kornia.metrics.ms_ssim),
    'lpips': KorniaMetricWrapper(kornia.metrics.lpips),
    'mean_iou': KorniaMetricWrapper(kornia.metrics.mean_iou),
    'accuracy': KorniaMetricWrapper(kornia.metrics.accuracy),
}
```

### ### Multi-Image Processing Operations

#### **Burst Processing Operations**

```python
'burst_processing_operations': {
    'hdr_merge': OperationSpec(
        name='hdr_merge',
        supported_modes=[ProcessingMode.BURST_PROCESSING],
        input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        input_count=(3, None),
        output_count=1,
        requires_metadata=['exposure_values', 'alignment_data'],
        description='HDR bracketed exposure merging'
    ),
    'focus_stack': OperationSpec(
        name='focus_stack',
        supported_modes=[ProcessingMode.BURST_PROCESSING],
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB, InputOutputType.MASK],
        input_count=(2, None),
        output_count=2,
        requires_metadata=['focus_distances'],
        description='Focus stacking for extended depth of field'
    ),
    'panorama_stitch': OperationSpec(
        name='panorama_stitch',
        supported_modes=[ProcessingMode.GROUP_PROCESSING],
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        input_count=(2, None),
        output_count=1,
        requires_metadata=['overlap_regions', 'camera_poses'],
        description='Panoramic image stitching'
    ),
    'temporal_denoise': OperationSpec(
        name='temporal_denoise',
        supported_modes=[ProcessingMode.BURST_PROCESSING],
        input_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        output_types=[InputOutputType.RAW_4CH, InputOutputType.RGB],
        input_count=(3, 20),
        output_count=1,
        requires_metadata=['temporal_alignment'],
        description='Temporal noise reduction using multiple frames'
    ),
}
```

### ### Registry Pattern Extensions

#### **Model Registry Pattern**

```python
# src/rawnind/dependencies/model_registry.py
MODEL_REGISTRY = {
    'utnet2': UTNet2Wrapper(),
    'utnet3': UTNet3Wrapper(),
    'bm3d': BM3DWrapper(),
    'learned_denoise': LearnedDenoiseNet(),
    'balle_encoder': BalleEncoderWrapper(),
    'balle_decoder': BalleDecoderWrapper(),
}


class ModelPipeline:
    def create_model(self, model_name: str, **params) -> PipelineOperation:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        return MODEL_REGISTRY[model_name](**params)
```

#### **Quality Checks Registry**

```python
QUALITY_CHECKS_REGISTRY = {
    'overexposure': OverexposureCheckWrapper(),
    'noise_estimation': NoiseEstimationWrapper(),
    'blur_detection': BlurDetectionWrapper(),
    'exposure_analysis': ExposureAnalysisWrapper(),
    'dynamic_range': DynamicRangeWrapper(),
    'color_accuracy': ColorAccuracyWrapper(),
}
```

### ### PyTorch Lightning Integration

#### **Universal Task Wrapper**

```python
# src/rawnind/core/lightning_task.py
class ImageProcessingTask(L.LightningModule):
    def __init__(self, operation_pipeline: OperationPipeline, loss_config: Dict):
        super().__init__()
        self.pipeline = operation_pipeline
        self.loss_functions = self._create_loss_functions(loss_config)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs, metadata = self.pipeline(inputs)
        loss = self._compute_adaptive_loss(outputs, targets, metadata)
        self.log_dict({
            'train_loss': loss,
            'pipeline_metadata': metadata
        })
        return loss

    def configure_optimizers(self):
        optimizers = []
        for operation in self.pipeline.operations:
            if operation.operation_type == "trainable":
                params = operation.get_parameters()
                optimizers.append(torch.optim.Adam(params.parameters(), lr=1e-4))
        return optimizers
```

### ### Hydra Configuration System

#### **Pipeline Configuration Example**

```yaml
# conf/pipeline/scene_referred_workflow.yaml
operations:
  - category: "raw_processing_operations"
    operation: "rawprepare"
    params:
      sensor_calibration: true

  - category: "raw_processing_operations"
    operation: "temperature"
    params:
      auto_wb: true

  - category: "raw_processing_operations"
    operation: "demosaic"
    params:
      algorithm: "amaze"

  - category: "kornia_color_operations"
    operation: "rgb_to_lab"
    params: { }

  - category: "kornia_filter_operations"
    operation: "bilateral_filter"
    params:
      kernel_size: [ 5, 5 ]
      sigma_color: 0.1
      sigma_space: 1.0

  - category: "tone_mapping_operations"
    operation: "exposure"
    params:
      compensation: 0.0

  - category: "tone_mapping_operations"
    operation: "filmicrgb"
    params:
      white_point: 4.0

  - category: "kornia_enhance_operations"
    operation: "adjust_contrast"
    params:
      contrast_factor: 1.1
```

### ### Smart Pipeline Assembly

#### **Compatibility Checking**

```python
class SmartPipelineAssembler:
    def validate_pipeline_compatibility(self, config: List[Dict]) -> List[str]:
        warnings = []
        for i, (current_op, next_op) in enumerate(zip(config[:-1], config[1:])):
            current_spec = self._get_operation_spec(current_op)
            next_spec = self._get_operation_spec(next_op)

            compatible_types = set(current_spec.output_types) & set(next_spec.input_types)
            if not compatible_types:
                warnings.append(f"Step {i}: {current_spec.name} → {next_spec.name} type mismatch")
        return warnings

    def suggest_missing_operations(self, input_type: InputOutputType, target_type: InputOutputType) -> List[str]:
        conversion_paths = {
            (InputOutputType.RAW_BAYER, InputOutputType.RAW_4CH): ['demosaic'],
            (InputOutputType.RAW_4CH, InputOutputType.RGB): ['colorin'],
            (InputOutputType.RGB, InputOutputType.GRAYSCALE): ['rgb_to_grayscale'],
        }
        return conversion_paths.get((input_type, target_type), [])
```

### ### Complete Integration Example

```python
def create_enhanced_pipeline(config_name: str = "scene_referred_workflow"):
    # Load configuration with Hydra
    with initialize_config_store():
        cfg = compose(config_name=config_name)

    # Create operation pipeline from registry
    pipeline = OperationPipeline(cfg.operations)

    # Wrap in Lightning task
    task = ImageProcessingTask(pipeline, cfg.training)

    # Add callbacks for monitoring
    callbacks = [
        PipelineVisualizationCallback(),
        QualityMetricsCallback()
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        precision=16
    )

    return trainer, task, cfg
```

### ### Key Benefits

#### **1. Production-Grade Architecture**

- **65+ operations** from darktable/vkdt/rawpy + comprehensive Kornia integration
- **Scene-referred workflow** with proper tone mapping
- **Multi-image processing** (HDR, focus stacking, panoramas)
- **GPU acceleration** through Kornia and Lightning

#### **2. Research Flexibility**

- **Configuration-driven experiments** with Hydra
- **Modular operation composition** with automatic validation
- **Multiple training strategies** through Lightning integration
- **Professional monitoring** with callback system

#### **3. Developer Experience**

- **Implementation agnostic** - classical algorithms, ML models work identically
- **Smart error handling** with pipeline compatibility checking
- **Comprehensive debugging** with operation introspection
- **Elegant composition** maintaining the original philosophy

#### **4. Complete Computer Vision Coverage**

- **Kornia filters**: bilateral, Gaussian, Sobel, Laplacian, motion blur, unsharp mask
- **Kornia color**: RGB/HSV/LAB/XYZ conversions, sepia effects
- **Kornia enhancement**: brightness, contrast, gamma, hue, saturation adjustments
- **Kornia geometry**: rotation, translation, scaling, warping, elastic transforms
- **Kornia augmentation**: comprehensive geometric and photometric augmentations
- **Kornia losses**: SSIM, MS-SSIM, LPIPS, focal, dice losses
- **Kornia metrics**: PSNR, SSIM, accuracy measurements
- **Kornia features**: Harris, corner detection, local features

This complete enhancement suite transforms the elegant Function-Based Composable Pipeline Architecture into a
production-ready system with comprehensive computer vision capabilities while maintaining the core principle: **the
pipeline doesn't care about implementation details, only functional intent**.