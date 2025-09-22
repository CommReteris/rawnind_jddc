"""
Clean programmatic API for inference package.

This module provides clean, modern interfaces for model inference without CLI dependencies.
It implements the API specifications defined in test_e2e_inference_clean_api.py.

The clean API provides:
1. Factory functions for creating model instances
2. Model loading utilities
3. Metrics computation utilities
4. Configuration classes for explicit parameter specification
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field

import torch
import yaml

from .base_inference import ImageToImageNN, BayerImageToImageNN
from .model_factory import Denoiser, BayerDenoiser, DenoiseCompress, BayerDenoiseCompress
from ..dependencies.pt_losses import losses, metrics as metrics_dict_from_module
from ..dependencies.pytorch_helpers import get_device


@dataclass
class InferenceConfig:
    """Configuration class for inference operations."""
    
    # Core required parameters
    architecture: str
    input_channels: int
    device: str = "cpu"
    
    # Model architecture parameters
    filter_units: int = 48  # funit in configs
    match_gain: str = "never"
    
    # Bayer-specific parameters
    enable_preupsampling: bool = False  # preupsample in configs
    
    # Model loading parameters
    use_best_checkpoint: bool = True
    loss_function: Optional[str] = None  # loss in configs
    
    # Processing parameters
    memory_efficient: bool = False
    transfer_function: Optional[str] = None
    
    # Validation and testing parameters
    crop_size: Optional[int] = None
    test_crop_size: Optional[int] = None
    val_crop_size: Optional[int] = None
    
    # Metrics and evaluation
    metrics_to_compute: List[str] = field(default_factory=list)
    
    # Compression-specific parameters
    encoder_arch: str = "Balle"
    decoder_arch: str = "Balle"
    hidden_out_channels: int = 192
    bitstream_out_channels: int = 64
    num_distributions: int = 64
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Expand supported architectures to include all actually used ones
        supported_archs = [
            'unet', 'utnet3', 'identity', 'bm3d',
            'ManyPriors', 'DenoiseThenCompress',
            'JPEGXL', 'JPEG', 'Passthrough',
            'standard', 'autoencoder'  # Add missing architectures
        ]
        if self.architecture not in supported_archs:
            raise ValueError(f"Unsupported architecture: {self.architecture}. Supported: {supported_archs}")
        
        if self.input_channels not in [3, 4]:
            raise ValueError(f"Input channels must be 3 or 4, got {self.input_channels}")
            
        if self.match_gain not in ["input", "output", "never"]:
            raise ValueError(f"Invalid match_gain option: {self.match_gain}")
            
        # Validate Bayer-specific constraints
        if self.enable_preupsampling and self.input_channels != 4:
            raise ValueError("Preupsampling can only be used with 4-channel (Bayer) input")


@dataclass
class ModelCheckpoint:
    """Information about a model checkpoint."""
    
    checkpoint_path: str
    step_number: int
    model_config: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def from_directory(cls, model_dir: str, metric_name: str = "val_msssim") -> 'ModelCheckpoint':
        """Load checkpoint info from model directory."""
        model_dir = Path(model_dir)
        
        # Load training results to find best step
        trainres_path = model_dir / "trainres.yaml"
        if not trainres_path.exists():
            raise FileNotFoundError(f"Training results not found: {trainres_path}")
            
        with open(trainres_path, 'r') as f:
            results = yaml.safe_load(f)
        
        if 'best_step' not in results or metric_name not in results['best_step']:
            raise KeyError(f"Metric {metric_name} not found in training results")
            
        best_step = results['best_step'][metric_name]
        checkpoint_path = model_dir / "saved_models" / f"iter_{best_step}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model configuration
        args_path = model_dir / "args.yaml"
        model_config = {}
        if args_path.exists():
            with open(args_path, 'r') as f:
                model_config = yaml.safe_load(f)
        
        return cls(
            checkpoint_path=str(checkpoint_path),
            step_number=best_step,
            model_config=model_config,
            metrics=results.get(str(best_step), {})
        )


class CleanDenoiser:
    """Clean denoiser interface without CLI dependencies."""
    
    def __init__(self, model: torch.nn.Module, config: InferenceConfig):
        """Initialize clean denoiser.
        
        Args:
            model: PyTorch model instance
            config: Inference configuration
        """
        self.model = model.eval()
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Expose configuration properties for easy access
        self.architecture = config.architecture
        self.input_channels = config.input_channels
        self.filter_units = config.filter_units
        
    def denoise_batch(self, batch_images: torch.Tensor) -> torch.Tensor:
        """Denoise a batch of images.
        
        Args:
            batch_images: Batch of images [B,C,H,W]
            
        Returns:
            Denoised batch of images [B,C,H,W]
        """
        return self.denoise(batch_images)
        
    def denoise(self, image: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """Denoise an input image.
        
        Args:
            image: Input image tensor [C,H,W] or [B,C,H,W]
            return_dict: If True, return dict with additional info
            
        Returns:
            Denoised image tensor or dict with additional information
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
                
            # Verify input channels
            if image.shape[1] != self.config.input_channels:
                raise ValueError(f"Expected {self.config.input_channels} channels, got {image.shape[1]}")
            
            # Move to device and run inference
            image = image.to(self.device)
            output = self.model(image)
            
            # Handle different output formats
            if isinstance(output, dict):
                denoised = output.get('reconstructed_image', output.get('output', output))
            else:
                denoised = output
            
            # Remove batch dimension if we added it
            if squeeze_output:
                denoised = denoised.squeeze(0)
            
            if return_dict:
                result = {'denoised_image': denoised}
                if isinstance(output, dict):
                    result.update({k: v for k, v in output.items() if k != 'reconstructed_image'})
                return result
            
            return denoised


class CleanBayerDenoiser(CleanDenoiser):
    """Clean Bayer denoiser interface with color space processing."""
    
    def __init__(self, model: torch.nn.Module, config: InferenceConfig):
        """Initialize clean Bayer denoiser.
        
        Args:
            model: PyTorch model instance
            config: Inference configuration
        """
        super().__init__(model, config)
        
        # Bayer-specific properties
        self.supports_bayer = True
        
        # Add demosaic_fn attribute expected by training code
        from ..dependencies.raw_processing import demosaic
        self.demosaic_fn = demosaic
    
    def denoise_bayer(
        self, 
        bayer_image: torch.Tensor, 
        rgb_xyz_matrix: torch.Tensor,
        return_dict: bool = False
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Denoise Bayer pattern image and convert to RGB.
        
        Args:
            bayer_image: Bayer pattern image [4,H,W] or [B,4,H,W]
            rgb_xyz_matrix: Color transformation matrix [3,3] or [B,3,3]
            return_dict: If True, return dict with additional info
            
        Returns:
            RGB denoised image or dict with additional information
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(bayer_image.shape) == 3:
                bayer_image = bayer_image.unsqueeze(0)
                rgb_xyz_matrix = rgb_xyz_matrix.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
            
            # Verify input format
            if bayer_image.shape[1] != 4:
                raise ValueError(f"Bayer image must have 4 channels, got {bayer_image.shape[1]}")
            if rgb_xyz_matrix.shape[-2:] != (3, 3):
                raise ValueError(f"RGB XYZ matrix must be 3x3, got {rgb_xyz_matrix.shape}")
            
            # Move to device
            bayer_image = bayer_image.to(self.device)
            rgb_xyz_matrix = rgb_xyz_matrix.to(self.device)
            
            # Run inference
            output = self.model(bayer_image)
            
            # Handle model output
            if isinstance(output, dict):
                denoised = output.get('reconstructed_image', output.get('output', output))
            else:
                denoised = output
            
            # Apply color space conversion for Bayer processing
            from ..dependencies import raw_processing as rawproc
            processed_output = rawproc.camRGB_to_lin_rec2020_images(denoised, rgb_xyz_matrix)
            
            # Remove batch dimension if we added it
            if squeeze_output:
                processed_output = processed_output.squeeze(0)
                
            if return_dict:
                result = {'denoised_image': processed_output}
                if isinstance(output, dict):
                    result.update({k: v for k, v in output.items() if k != 'reconstructed_image'})
                return result
                
            return processed_output


class CleanCompressor:
    """Clean compressor interface for denoise+compress models."""
    
    def __init__(self, model: torch.nn.Module, config: InferenceConfig):
        """Initialize clean compressor.
        
        Args:
            model: PyTorch compression model instance
            config: Inference configuration
        """
        self.model = model.eval()
        self.config = config
        self.device = torch.device(config.device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Expose configuration properties for easy access
        self.architecture = config.architecture
        self.input_channels = config.input_channels
        self.filter_units = config.filter_units
        
    def compress(
        self, 
        image: torch.Tensor, 
        target_bpp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Alias for compress_and_denoise method.
        
        Args:
            image: Input image tensor [C,H,W] or [B,C,H,W]
            target_bpp: Target bits per pixel (optional)
            
        Returns:
            Dict containing compression results
        """
        return self.compress_and_denoise(image, target_bpp)
        
    def decompress(self, compressed_data: torch.Tensor) -> torch.Tensor:
        """Decompress compressed image data.
        
        Args:
            compressed_data: Compressed image tensor
            
        Returns:
            Decompressed image tensor
        """
        # For joint denoise+compress models, the forward pass does both
        # This is a simplified implementation - real decompression might be more complex
        with torch.no_grad():
            if len(compressed_data.shape) == 3:
                compressed_data = compressed_data.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
                
            compressed_data = compressed_data.to(self.device)
            output = self.model(compressed_data)
            
            if isinstance(output, dict):
                decompressed = output.get('reconstructed_image', output.get('output', output))
            else:
                decompressed = output
            
            if squeeze_output:
                decompressed = decompressed.squeeze(0)
                
            return decompressed
        
    def compress_and_denoise(
        self, 
        image: torch.Tensor, 
        target_bpp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Apply joint denoising and compression.
        
        Args:
            image: Input image tensor [C,H,W] or [B,C,H,W]
            target_bpp: Target bits per pixel (optional)
            
        Returns:
            Dict containing 'denoised_image', 'bpp', and other compression info
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
                squeeze_output = True
            else:
                squeeze_output = False
                
            # Move to device and run inference
            image = image.to(self.device)
            output = self.model(image)
            
            # Extract results
            if isinstance(output, dict):
                denoised = output['reconstructed_image']
                bpp = output.get('bpp', 0.0)
            else:
                denoised = output
                bpp = 0.0
            
            # Remove batch dimension if we added it
            if squeeze_output:
                denoised = denoised.squeeze(0)
                if isinstance(bpp, torch.Tensor) and bpp.dim() > 0:
                    bpp = bpp.squeeze(0)
            
            return {
                'denoised_image': denoised,
                'bpp': float(bpp) if isinstance(bpp, torch.Tensor) else bpp,
                'compression_ratio': float(24.0 / bpp) if bpp > 0 else float('inf')
            }


def create_rgb_denoiser(
    architecture: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    filter_units: int = 48,
    **kwargs
) -> CleanDenoiser:
    """Create RGB denoiser with clean API.
    
    Args:
        architecture: Model architecture ('unet', 'utnet3', 'bm3d', etc.)
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        filter_units: Number of filter units in the model
        **kwargs: Additional configuration parameters
        
    Returns:
        CleanDenoiser instance ready for inference
    """
    config = InferenceConfig(
        architecture=architecture,
        input_channels=3,
        device=device,
        filter_units=filter_units,
        **kwargs
    )
    
    # Convert device string to format expected by legacy code
    if device == "cpu":
        device_param = -1
    elif device.startswith("cuda"):
        if ":" in device:
            device_param = int(device.split(":")[1])
        else:
            device_param = 0  # cuda without number = cuda:0
    else:
        device_param = device  # Pass through other formats
    
    # Create model instance without CLI dependencies
    denoiser_instance = Denoiser(
        test_only=True,
        use_cli=False,
        arch=architecture,
        in_channels=3,
        funit=filter_units,
        device=device_param,
        load_path=checkpoint_path,
        match_gain=config.match_gain,
        metrics=config.metrics_to_compute,
        debug_options=[],
        save_dpath='/tmp/rawnind_inference'
    )
    
    return CleanDenoiser(denoiser_instance.model, config)


def create_bayer_denoiser(
    architecture: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    filter_units: int = 48,
    **kwargs
) -> CleanBayerDenoiser:
    """Create Bayer denoiser with clean API.
    
    Args:
        architecture: Model architecture ('unet', 'utnet3', 'bm3d', etc.)
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        filter_units: Number of filter units in the model
        **kwargs: Additional configuration parameters including enable_preupsampling
        
    Returns:
        CleanBayerDenoiser instance ready for inference
    """
    config = InferenceConfig(
        architecture=architecture,
        input_channels=4,
        device=device,
        filter_units=filter_units,
        **kwargs
    )
    
    # Convert device string to format expected by legacy code
    if device == "cpu":
        device_param = -1
    elif device.startswith("cuda"):
        if ":" in device:
            device_param = int(device.split(":")[1])
        else:
            device_param = 0  # cuda without number = cuda:0
    else:
        device_param = device  # Pass through other formats
    
    # Create model instance without CLI dependencies
    denoiser_instance = BayerDenoiser(
        test_only=True,
        use_cli=False,
        arch=architecture,
        in_channels=4,
        funit=filter_units,
        device=device_param,
        load_path=checkpoint_path,
        match_gain=config.match_gain,
        metrics=config.metrics_to_compute,
        debug_options=[],
        save_dpath='/tmp/rawnind_inference'
    )
    
    # Set preupsample attribute for model instantiation (vars(self).get("preupsample", False))
    denoiser_instance.preupsample = config.enable_preupsampling
    
    return CleanBayerDenoiser(denoiser_instance.model, config)


def create_compressor(
    architecture: str,
    encoder_arch: str = "Balle",
    decoder_arch: str = "Balle",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    input_channels: int = 3,
    **kwargs
) -> CleanCompressor:
    """Create compressor with clean API.
    
    Args:
        architecture: Compression architecture ('ManyPriors', 'DenoiseThenCompress', etc.)
        encoder_arch: Encoder architecture ('Balle', etc.)
        decoder_arch: Decoder architecture ('Balle', 'BayerPS', 'BayerTC')
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.)
        input_channels: Number of input channels (3 for RGB, 4 for Bayer)
        **kwargs: Additional configuration parameters
        
    Returns:
        CleanCompressor instance ready for inference
    """
    config = InferenceConfig(
        architecture=architecture,
        input_channels=input_channels,
        device=device,
        **kwargs
    )
    
    # Convert device string to format expected by legacy code
    if device == "cpu":
        device_param = -1
    elif device.startswith("cuda"):
        if ":" in device:
            device_param = int(device.split(":")[1])
        else:
            device_param = 0  # cuda without number = cuda:0
    else:
        device_param = device  # Pass through other formats
    
    # Determine if this is Bayer or RGB compression
    if input_channels == 4:
        compressor_class = BayerDenoiseCompress
    else:
        compressor_class = DenoiseCompress
    
    # Create model instance without CLI dependencies
    compressor_instance = compressor_class(
        test_only=True,
        use_cli=False,
        arch=architecture,
        arch_enc=encoder_arch,
        arch_dec=decoder_arch,
        in_channels=input_channels,
        device=device_param,
        load_path=checkpoint_path,
        match_gain=config.match_gain,
        hidden_out_channels=kwargs.get('hidden_out_channels', 192),
        bitstream_out_channels=kwargs.get('bitstream_out_channels', 64),
        save_dpath='/tmp/rawnind_inference'
    )
    
    return CleanCompressor(compressor_instance.model, config)


def load_model_from_checkpoint(
    checkpoint_path: str,
    architecture: Optional[str] = None,
    input_channels: Optional[int] = None,
    device: str = "cpu",
    **model_kwargs
) -> Dict[str, Any]:
    """Load model from checkpoint with clean API.
    
    Args:
        checkpoint_path: Path to checkpoint file or model directory
        architecture: Model architecture name (auto-detected if None)
        input_channels: Number of input channels (auto-detected if None)
        device: Device to load model on
        **model_kwargs: Additional model configuration
        
    Returns:
        Dict containing:
        - 'model': Loaded PyTorch model
        - 'config': Model configuration
        - 'checkpoint_info': Information about the checkpoint
    """    
    # Handle directory vs file path
    checkpoint_path = Path(checkpoint_path)
    
    if checkpoint_path.is_dir():
        # Find best checkpoint in directory and load model config
        try:
            checkpoint_info = ModelCheckpoint.from_directory(str(checkpoint_path))
            actual_checkpoint_path = checkpoint_info.checkpoint_path
            model_config = checkpoint_info.model_config
        except (FileNotFoundError, KeyError) as e:
            logging.warning(f"Could not find best checkpoint, looking for latest: {e}")
            # Fallback to finding latest checkpoint
            saved_models_dir = checkpoint_path / "saved_models"
            if saved_models_dir.exists():
                checkpoint_files = list(saved_models_dir.glob("iter_*.pt"))
                if checkpoint_files:
                    # Sort by iteration number
                    latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
                    actual_checkpoint_path = str(latest_checkpoint)
                    step_number = int(latest_checkpoint.stem.split('_')[1])
                    
                    # Try to load model config from args.yaml
                    args_path = checkpoint_path / "args.yaml"
                    model_config = {}
                    if args_path.exists():
                        with open(args_path, 'r') as f:
                            model_config = yaml.safe_load(f)
                    
                    checkpoint_info = ModelCheckpoint(
                        checkpoint_path=actual_checkpoint_path,
                        step_number=step_number,
                        model_config=model_config
                    )
                else:
                    raise FileNotFoundError(f"No checkpoint files found in {saved_models_dir}")
            else:
                raise FileNotFoundError(f"Saved models directory not found: {saved_models_dir}")
    else:
        actual_checkpoint_path = str(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # For individual checkpoint files, try to find args.yaml in parent directory
        parent_dir = checkpoint_path.parent.parent  # Go up from saved_models/iter_xxx.pt
        args_path = parent_dir / "args.yaml"
        model_config = {}
        if args_path.exists():
            with open(args_path, 'r') as f:
                model_config = yaml.safe_load(f)
        
        # Extract step number from filename
        try:
            step_number = int(checkpoint_path.stem.split('_')[1])
        except (IndexError, ValueError):
            step_number = 0
            
        checkpoint_info = ModelCheckpoint(
            checkpoint_path=actual_checkpoint_path,
            step_number=step_number,
            model_config=model_config
        )
    
    # Auto-detect architecture and input channels from model config if not provided
    if architecture is None:
        architecture = model_config.get('arch')
        if architecture is None:
            raise ValueError("Architecture not provided and could not be auto-detected from checkpoint")
    
    if input_channels is None:
        input_channels = model_config.get('in_channels', 3)
    
    # Convert device string to format expected by legacy code
    if device == "cpu":
        device_param = -1
    elif device.startswith("cuda"):
        if ":" in device:
            device_param = int(device.split(":")[1])
        else:
            device_param = 0  # cuda without number = cuda:0
    else:
        device_param = device  # Pass through other formats
    
    # Extract other model parameters from config
    filter_units = model_config.get('funit', model_kwargs.get('filter_units', 48))
    
    # Create appropriate model class
    config = InferenceConfig(
        architecture=architecture,
        input_channels=input_channels,
        device=device,
        filter_units=filter_units,
        **model_kwargs
    )
    
    # Create model without loading checkpoint first
    if architecture in Denoiser.ARCHS:
        # For denoiser models, we need loss function from config
        loss_function = model_config.get('loss', model_kwargs.get('loss', 'msssim'))
        
        if input_channels == 4:
            model_instance = BayerDenoiser(
                test_only=True,
                use_cli=False,
                arch=architecture,
                in_channels=input_channels,
                funit=filter_units,
                loss=loss_function,
                device=device_param,
                load_path=None,  # Don't auto-load
                **model_kwargs
            )
        else:
            model_instance = Denoiser(
                test_only=True,
                use_cli=False,
                arch=architecture,
                in_channels=input_channels,
                funit=filter_units,
                loss=loss_function,
                device=device_param,
                load_path=None,  # Don't auto-load
                **model_kwargs
            )
    elif architecture in DenoiseCompress.ARCHS:
        # For compression models, we need encoder/decoder architectures
        encoder_arch = model_config.get('arch_enc', model_kwargs.get('encoder_arch', 'Balle'))
        decoder_arch = model_config.get('arch_dec', model_kwargs.get('decoder_arch', 'Balle'))
        hidden_channels = model_config.get('hidden_out_channels', model_kwargs.get('hidden_out_channels', 192))
        bitstream_channels = model_config.get('bitstream_out_channels', model_kwargs.get('bitstream_out_channels', 64))
        
        if input_channels == 4:
            model_instance = BayerDenoiseCompress(
                test_only=True,
                use_cli=False,
                arch=architecture,
                arch_enc=encoder_arch,
                arch_dec=decoder_arch,
                in_channels=input_channels,
                funit=filter_units,
                hidden_out_channels=hidden_channels,
                bitstream_out_channels=bitstream_channels,
                device=device_param,
                load_path=None,  # Don't auto-load
                **model_kwargs
            )
        else:
            model_instance = DenoiseCompress(
                test_only=True,
                use_cli=False,
                arch=architecture,
                arch_enc=encoder_arch,
                arch_dec=decoder_arch,
                in_channels=input_channels,
                funit=filter_units,
                hidden_out_channels=hidden_channels,
                bitstream_out_channels=bitstream_channels,
                device=device_param,
                load_path=None,  # Don't auto-load
                **model_kwargs
            )
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Load checkpoint manually
    ImageToImageNN.load_model(model_instance.model, actual_checkpoint_path, device=config.device)
    
    # Return appropriate clean interface based on architecture
    if architecture in Denoiser.ARCHS:
        if input_channels == 4:
            return CleanBayerDenoiser(model_instance.model, config)
        else:
            return CleanDenoiser(model_instance.model, config)
    elif architecture in DenoiseCompress.ARCHS:
        return CleanCompressor(model_instance.model, config)
    else:
        # Fallback to dictionary format for unknown architectures
        return {
            'model': model_instance.model,
            'config': config,
            'checkpoint_info': checkpoint_info,
            'model_instance': model_instance
        }


def compute_image_metrics(
    predicted_image: torch.Tensor,
    ground_truth_image: torch.Tensor,
    metrics: List[str],
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Compute image quality metrics between predicted and ground truth images.
    
    Args:
        predicted_image: Model prediction [C,H,W] or [B,C,H,W]
        ground_truth_image: Ground truth image [C,H,W] or [B,C,H,W]
        metrics_list: List of metrics to compute ('mse', 'psnr', 'ms_ssim', etc.)
        mask: Optional mask for valid pixels [C,H,W] or [B,C,H,W]
        
    Returns:
        Dict mapping metric names to computed values
    """
    # Ensure tensors are the same shape
    if predicted_image.shape != ground_truth_image.shape:
        raise ValueError(f"Image shapes must match: {predicted_image.shape} vs {ground_truth_image.shape}")
    
    # Add batch dimension if needed
    if len(predicted_image.shape) == 3:
        predicted_image = predicted_image.unsqueeze(0)
        ground_truth_image = ground_truth_image.unsqueeze(0)
        if mask is not None:
            mask = mask.unsqueeze(0)
    
    # Apply mask if provided
    if mask is not None:
        predicted_image = predicted_image * mask
        ground_truth_image = ground_truth_image * mask
    
    results = {}
    
    for metric_name in metrics:
        if metric_name not in metrics_dict_from_module:
            logging.warning(f"Unknown metric: {metric_name}")
            continue
            
        try:
            # Check image size constraints for MS-SSIM
            if metric_name in ['msssim', 'msssim_loss']:
                min_size = min(predicted_image.shape[-2:])  # Get minimum of H,W
                if min_size < 162:  # MS-SSIM requires at least 162x162 due to 4 downsamplings
                    logging.warning(f"Skipping {metric_name}: image size {min_size} too small (need ≥162)")
                    results[metric_name] = float('nan')
                    continue
            
            metric_fn = metrics_dict_from_module[metric_name]()
            with torch.no_grad():
                metric_value = metric_fn(predicted_image, ground_truth_image)
                
            # Convert to float if it's a tensor
            if isinstance(metric_value, torch.Tensor):
                metric_value = float(metric_value.item())
            
            results[metric_name] = metric_value
            
        except Exception as e:
            logging.error(f"Error computing {metric_name}: {e}")
            results[metric_name] = float('nan')
    
    return results


def list_available_models(models_base_path: Optional[str] = None) -> Dict[str, List[str]]:
    """List available pre-trained models.
    
    Args:
        models_base_path: Base path to search for models (optional)
        
    Returns:
        Dict mapping model types to lists of available model names
    """
    if models_base_path is None:
        # Use default model paths
        models_base_path = Path(__file__).parent.parent / "models"
    
    models_base_path = Path(models_base_path)
    available_models = {
        'denoisers': [],
        'compressors': [],
        'bayer_denoisers': [],
        'bayer_compressors': []
    }
    
    if not models_base_path.exists():
        return available_models
    
    # Look for model directories with weight files
    for model_dir in models_base_path.glob("*"):
        if model_dir.is_dir() and (model_dir / "saved_models").exists():
            model_name = model_dir.name
            
            # Try to determine model type from args.yaml or directory name
            args_file = model_dir / "args.yaml"
            if args_file.exists():
                try:
                    with open(args_file, 'r') as f:
                        args = yaml.safe_load(f)
                    
                    in_channels = args.get('in_channels', 3)
                    arch = args.get('arch', '')
                    
                    # Categorize based on architecture and channels
                    if arch in DenoiseCompress.ARCHS:
                        if in_channels == 4:
                            available_models['bayer_compressors'].append(model_name)
                        else:
                            available_models['compressors'].append(model_name)
                    elif arch in Denoiser.ARCHS:
                        if in_channels == 4:
                            available_models['bayer_denoisers'].append(model_name)
                        else:
                            available_models['denoisers'].append(model_name)
                            
                except Exception as e:
                    logging.warning(f"Could not parse {args_file}: {e}")
    
    return available_models


def find_best_model_in_directory(
    model_directory: str,
    metric_name: str = "val_msssim"
) -> str:
    """Find the best model checkpoint in a directory based on a metric.
    
    Args:
        model_directory: Path to model experiment directory
        metric_name: Metric to use for finding best model
        
    Returns:
        Path to the best model checkpoint file
    """
    try:
        checkpoint_info = ModelCheckpoint.from_directory(model_directory, metric_name)
        return checkpoint_info.checkpoint_path
    except (FileNotFoundError, KeyError) as e:
        # Fallback to latest checkpoint
        model_dir = Path(model_directory)
        saved_models_dir = model_dir / "saved_models"
        
        if not saved_models_dir.exists():
            raise FileNotFoundError(f"No saved models directory: {saved_models_dir}")
        
        checkpoint_files = list(saved_models_dir.glob("iter_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {saved_models_dir}")
        
        # Return latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[1]))
        return str(latest_checkpoint)


def convert_device_format(device: Union[str, int, torch.device]) -> Union[str, int]:
    """Convert device to format expected by legacy code.
    
    Args:
        device: Device specification
        
    Returns:
        Union[str, int]: Device in legacy format (-1 for CPU, device number for CUDA)
    """
    if isinstance(device, torch.device):
        if device.type == 'cpu':
            return -1
        elif device.type == 'cuda':
            return device.index if device.index is not None else 0
        else:
            return str(device)
    elif isinstance(device, int):
        return device  # Already in legacy format
    elif isinstance(device, str):
        if device == "cpu":
            return -1
        elif device.startswith("cuda"):
            if ":" in device:
                return int(device.split(":")[1])
            else:
                return 0  # cuda without number = cuda:0
        else:
            return device  # Pass through other formats
    else:
        raise ValueError(f"Unsupported device specification: {device}")