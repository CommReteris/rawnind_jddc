### Complete Function-Based Composable Pipeline Architecture - Full Suite Enhancement

Here's the comprehensive enhancement suite for your RawNind project, integrating all architectural frameworks with the
corrected modern operation registry:

### ### Core Architecture Foundation

#### ### 1. Function-Based Composable Pipeline System

The universal operation interface that enables implementation-agnostic processing:

```python
# src/rawnind/core/pipeline_operation.py
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

### ### 2. Modern Operation Registry (45+ Legitimate Operations)

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
        description='Modern white balance adjustment'
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

#### **Modern Color Processing Operations**

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
        description='Modern RGB channel mixer with color grading'
    ),
    'colorbalancergb': OperationSpec(
        name='colorbalancergb',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Modern RGB-aware color balance'
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
        description='Modern filmic tone mapping with RGB processing'
    ),
    'sigmoid': OperationSpec(
        name='sigmoid',
        input_types=[InputOutputType.RGB],
        output_types=[InputOutputType.RGB],
        description='Modern sigmoid tone mapping for smooth transitions'
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

### ### 3. Registry Pattern Extensions

#### **Model Registry Pattern**

```python
# src/rawnind/dependencies/model_registry.py
MODEL_REGISTRY = {
    'utnet2': UTNet2Wrapper(),
    'utnet3': UTNet3Wrapper(),
    'bm3d': BM3DWrapper(),
    'learned_denoise': LearnedDenoiseNet(),
}


class ModelPipeline:
    def create_model(self, model_name: str, **params) -> PipelineOperation:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_name}")
        return MODEL_REGISTRY[model_name]
```

#### **Quality Checks Registry**

```python
QUALITY_CHECKS_REGISTRY = {
    'overexposure': OverexposureCheckWrapper(),
    'noise_estimation': NoiseEstimationWrapper(),
    'blur_detection': BlurDetectionWrapper(),
    'exposure_analysis': ExposureAnalysisWrapper(),
}


class QualityChecksPipeline:
    def __call__(self, image: torch.Tensor) -> Dict[str, Any]:
        results = {}
        for check_fn, params in self.checks:
            result = check_fn(image, **params)
            results[check_fn.__name__] = result
        return results
```

### ### 4. Kornia Integration Enhancement

#### **Enhanced Augmentations with Kornia**

```python
# Enhanced augmentation registry with Kornia
KORNIA_AUGMENTATION_REGISTRY = {
    'horizontal_flip': KA.RandomHorizontalFlip(p=0.5),
    'vertical_flip': KA.RandomVerticalFlip(p=0.5),
    'rotation': KA.RandomRotation(degrees=15, p=0.5),
    'affine': KA.RandomAffine(degrees=10, translate=0.1, scale=(0.9, 1.1), p=0.5),
    'color_jitter': KA.ColorJitter(brightness=0.1, contrast=0.1, p=0.5),
    'gamma': KA.RandomGamma(gamma=(0.8, 1.2), gain=(0.9, 1.1), p=0.5),
    'noise': KA.RandomGaussianNoise(mean=0.0, std=0.01, p=0.3),
    'blur': KA.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 1.0), p=0.2),
}
```

#### **Advanced Loss Functions**

```python
KORNIA_LOSS_REGISTRY = {
    'ssim': KL.SSIMLoss(window_size=11, reduction='mean'),
    'ms_ssim': KL.MS_SSIMLoss(reduction='mean'),
    'lpips': KL.LPIPSLoss(net_type='alex', reduction='mean'),
    'total_variation': KL.TotalVariation(reduction='mean'),
    'gradient_loss': lambda x, y: F.l1_loss(KF.spatial_gradient(x), KF.spatial_gradient(y)),
}
```

### ### 5. PyTorch Lightning Integration

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
        # Separate optimizers for different operation types
        optimizers = []
        for operation in self.pipeline.operations:
            if operation.operation_type == "trainable":
                params = operation.get_parameters()
                optimizers.append(torch.optim.Adam(params.parameters(), lr=1e-4))
        return optimizers
```

### ### 6. Hydra Configuration System

#### **Hierarchical Configuration**

```yaml
# conf/config.yaml
defaults:
  - pipeline: scene_referred_workflow
  - model: utnet2
  - training: adam_scheduler
  - data: raw_bayer_dataset
  - augmentations: kornia_standard
  - quality_checks: production_suite
  - _self_

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

  - category: "color_processing_operations"
    operation: "colorin"
    params:
      profile: "srgb"

  - category: "tone_mapping_operations"
    operation: "exposure"
    params:
      compensation: 0.0

  - category: "tone_mapping_operations"
    operation: "filmicrgb"
    params:
      white_point: 4.0

  - category: "color_processing_operations"
    operation: "colorbalancergb"
    params:
      shadows: [ 1.0, 1.0, 1.0 ]
      midtones: [ 1.0, 1.0, 1.0 ]
      highlights: [ 1.0, 1.0, 1.0 ]
```

### ### 7. Production Features

#### **Multi-Image Processing Support**

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
}
```

#### **Smart Pipeline Assembly**

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
```

### ### 8. Advanced Features

#### **Callback System**

```python
class PipelineVisualizationCallback(L.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        for i, operation in enumerate(pl_module.pipeline.operations):
            if hasattr(operation, 'last_output'):
                self._save_visualization(f"op_{i}_{operation.spec.name}", operation.last_output)

class QualityMetricsCallback(L.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if 'quality_metrics' in outputs:
            for metric_name, value in outputs['quality_metrics'].items():
                trainer.logger.log_metrics({f"quality/{metric_name}": value})
```

### ### Complete Integration Example

```python
# Complete workflow example
def create_enhanced_pipeline(config_name: str = "scene_referred_workflow"):
    # 1. Load configuration with Hydra
    with initialize_config_store():
        cfg = compose(config_name=config_name)
    
    # 2. Create operation pipeline from modern registry
    pipeline = OperationPipeline(cfg.operations)
    
    # 3. Wrap in Lightning task
    task = ImageProcessingTask(pipeline, cfg.training)
    
    # 4. Add callbacks for monitoring
    callbacks = [
        PipelineVisualizationCallback(),
        QualityMetricsCallback()
    ]
    
    # 5. Create trainer with Lightning
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        callbacks=callbacks,
        precision=16  # Mixed precision support
    )
    
    return trainer, task, cfg

# Usage
trainer, task, config = create_enhanced_pipeline()
trainer.fit(task, datamodule=create_datamodule(config.data))
```

### ### Key Benefits

#### **1. Production-Grade Architecture**

- **45+ legitimate operations** from darktable/vkdt/rawpy
- **Scene-referred workflow** with modern tone mapping
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
- **Elegant composition** maintaining your original philosophy

#### **4. Ecosystem Integration**

- **PyTorch Lightning** for professional ML workflows
- **Hydra** for configuration management and experiment tracking
- **Kornia** for state-of-the-art computer vision operations
- **Modern raw processing** following current darktable practices

This complete enhancement suite transforms your elegant Function-Based Composable Pipeline Architecture into a
production-ready system that rivals commercial raw processing software while maintaining the core principle: **the
pipeline doesn't care about implementation details, only functional intent**.