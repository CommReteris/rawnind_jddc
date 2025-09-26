"""Model operations for the pipeline."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from ..core import PipelineOperation


class Encoder(PipelineOperation):
    """Encode image data to latent representation."""
    
    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.encoder_net = None
    
    def initialize(self):
        """Initialize the encoder network."""
        super().initialize()
        
        # Create a simple encoder network
        self.encoder_net = nn.Sequential(
            nn.Conv2d(3 if self.spec.input_types[0].value == "rgb" else 4, 
                     32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.config.get("latent_dim", 256), 
                     kernel_size=3, stride=2, padding=1),
        )
        self.encoder_net = self.encoder_net.to(self.device)
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Encode input to latent space.
        
        Args:
            data: Input tensor (RGB or RGGB)
            metadata: Metadata
            
        Returns:
            Tuple of (latent tensor, metadata)
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure batch dimension
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        # Encode
        latent = self.encoder_net(data)
        
        metadata["encoded"] = True
        metadata["latent_shape"] = list(latent.shape)
        
        # Remove batch dimension if it wasn't there originally
        if latent.shape[0] == 1:
            latent = latent.squeeze(0)
        
        return latent, metadata
    
    def to(self, device: str):
        """Move operation to device."""
        super().to(device)
        if self.encoder_net is not None:
            self.encoder_net = self.encoder_net.to(device)
        return self


class Decoder(PipelineOperation):
    """Decode latent representation back to image."""
    
    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.decoder_net = None
    
    def initialize(self):
        """Initialize the decoder network."""
        super().initialize()
        
        # Create a simple decoder network
        latent_dim = self.config.get("latent_dim", 256)
        output_channels = 3 if self.spec.output_types[0].value == "rgb" else 4
        
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 
                              kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        self.decoder_net = self.decoder_net.to(self.device)
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Decode latent to image space.
        
        Args:
            data: Latent tensor
            metadata: Metadata
            
        Returns:
            Tuple of (decoded tensor, metadata)
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure batch dimension
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        # Decode
        decoded = self.decoder_net(data)
        
        metadata["decoded"] = True
        
        # Remove batch dimension if it wasn't there originally
        if decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        
        return decoded, metadata
    
    def to(self, device: str):
        """Move operation to device."""
        super().to(device)
        if self.decoder_net is not None:
            self.decoder_net = self.decoder_net.to(device)
        return self


class Denoiser(PipelineOperation):
    """Denoise image data."""
    
    def __init__(self, spec, config=None):
        super().__init__(spec, config)
        self.denoiser_net = None
    
    def initialize(self):
        """Initialize the denoiser network."""
        super().initialize()
        
        # Create a simple denoising network (U-Net style)
        in_channels = 3 if self.spec.input_types[0].value == "rgb" else 4
        
        self.denoiser_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=3, padding=1),
        )
        self.denoiser_net = self.denoiser_net.to(self.device)
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Denoise input data.
        
        Args:
            data: Input tensor
            metadata: Metadata
            
        Returns:
            Tuple of (denoised tensor, metadata)
        """
        if not self._initialized:
            self.initialize()
        
        # Ensure batch dimension
        if len(data.shape) == 3:
            data = data.unsqueeze(0)
        
        # Denoise (residual learning)
        noise = self.denoiser_net(data)
        denoised = data - noise
        denoised = torch.clamp(denoised, 0, 1)
        
        metadata["denoised"] = True
        metadata["noise_level"] = float(noise.abs().mean())
        
        # Remove batch dimension if it wasn't there originally
        if denoised.shape[0] == 1:
            denoised = denoised.squeeze(0)
        
        return denoised, metadata
    
    def to(self, device: str):
        """Move operation to device."""
        super().to(device)
        if self.denoiser_net is not None:
            self.denoiser_net = self.denoiser_net.to(device)
        return self


class Compressor(PipelineOperation):
    """Compress data to reduced representation."""
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compress input data.
        
        Args:
            data: Input tensor
            metadata: Metadata
            
        Returns:
            Tuple of (compressed tensor, metadata)
        """
        # Simple compression via quantization and downsampling
        compression_ratio = self.config.get("ratio", 0.5)
        
        # Downsample
        if compression_ratio < 1.0:
            scale = int(1.0 / compression_ratio)
            if len(data.shape) == 3:
                compressed = data[:, ::scale, ::scale]
            else:
                compressed = data[::scale, ::scale]
        else:
            compressed = data
        
        # Quantize to reduce precision
        bits = self.config.get("bits", 8)
        levels = 2 ** bits
        compressed = torch.round(compressed * levels) / levels
        
        metadata["compressed"] = True
        metadata["compression_ratio"] = compression_ratio
        metadata["quantization_bits"] = bits
        metadata["original_shape"] = list(data.shape)
        
        return compressed, metadata