"""Postprocessing operations for the pipeline."""

from typing import Tuple, Dict, Any, Optional
import torch
import numpy as np
from ..core import PipelineOperation


class ToneMapper(PipelineOperation):
    """Apply tone mapping to image data."""
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply tone mapping.
        
        Args:
            data: Input tensor
            metadata: Metadata
            
        Returns:
            Tuple of (tone-mapped tensor, metadata)
        """
        method = self.config.get("method", "reinhard")
        
        if method == "reinhard":
            # Reinhard tone mapping
            luminance = 0.2126 * data[0] + 0.7152 * data[1] + 0.0722 * data[2]
            key = self.config.get("key", 0.18)
            
            # Compute log average luminance
            log_avg = torch.exp(torch.log(luminance + 1e-8).mean())
            
            # Scale luminance
            scaled_lum = key * luminance / log_avg
            
            # Compress
            white = self.config.get("white", 2.0)
            compressed_lum = scaled_lum * (1.0 + scaled_lum / (white * white)) / (1.0 + scaled_lum)
            
            # Apply back to color channels
            scale = compressed_lum / (luminance + 1e-8)
            output = data * scale.unsqueeze(0)
            
        elif method == "filmic":
            # Filmic tone mapping
            a = self.config.get("a", 2.51)
            b = self.config.get("b", 0.03)
            c = self.config.get("c", 2.43)
            d = self.config.get("d", 0.59)
            e = self.config.get("e", 0.14)
            
            output = ((data * (a * data + b)) / 
                     (data * (c * data + d) + e))
            
        else:
            # Simple exposure and gamma
            exposure = self.config.get("exposure", 1.0)
            output = data * exposure
        
        output = torch.clamp(output, 0, 1)
        metadata["tone_mapped"] = True
        metadata["tone_map_method"] = method
        
        return output, metadata


class GammaCorrection(PipelineOperation):
    """Apply gamma correction to image data."""
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply gamma correction.
        
        Args:
            data: Input tensor
            metadata: Metadata
            
        Returns:
            Tuple of (gamma-corrected tensor, metadata)
        """
        gamma = self.config.get("gamma", 2.2)
        
        # Apply inverse gamma (sRGB approximation)
        if self.config.get("inverse", False):
            output = torch.pow(data, gamma)
        else:
            output = torch.pow(data, 1.0 / gamma)
        
        metadata["gamma_corrected"] = True
        metadata["gamma_value"] = gamma
        
        return output, metadata


class Saver(PipelineOperation):
    """Save processed image to file."""
    
    def forward(self,
                data: torch.Tensor,
                metadata: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Save image data.
        
        Args:
            data: Input tensor
            metadata: Metadata
            
        Returns:
            Tuple of (data unchanged, metadata with save path)
        """
        import os
        
        # Get output path from config or metadata
        output_path = self.config.get("output_path", 
                                      metadata.get("output_path", "output.png"))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", 
                   exist_ok=True)
        
        # Convert to numpy and save (in real implementation)
        # For now, just update metadata
        metadata["saved_to"] = output_path
        metadata["save_format"] = os.path.splitext(output_path)[1]
        
        # In a real implementation, we would save here:
        # if data is torch tensor, convert to numpy
        # if len(data.shape) == 3 and data.shape[0] in [3, 4]:
        #     # CHW to HWC
        #     img_data = data.permute(1, 2, 0).cpu().numpy()
        #     # Save using PIL or cv2
        
        return data, metadata