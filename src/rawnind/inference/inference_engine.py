"""Inference engine for running model predictions.

This module provides the core inference functionality for running
trained models on input data. It handles batch processing, device
management, and output formatting.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

from typing import Any, Dict, Union

import torch


# Import from dependencies package (will be moved later)


class InferenceEngine:
    """Handles model inference operations and result processing.

    This class provides methods for running inference on trained models,
    handling different input formats, and processing model outputs.
    """

    def __init__(self, model: torch.nn.Module, device: Union[str, torch.device] = None):
        """Initialize the inference engine.

        Args:
            model: PyTorch model to use for inference
            device: Device to run inference on (defaults to model's device)
        """
        self.model = model
        self.device = device or next(model.parameters()).device

    def infer(
            self,
            img: torch.Tensor,
            return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """Perform inference with the model on input images.

        This method runs the model in evaluation mode on the input images, performing
        the image-to-image transformation (such as denoising or compression/decompression).
        It handles batched or single images and performs appropriate device transfers.

        Args:
            img: Input image tensor with shape [C,H,W] or [B,C,H,W], where:
                 B = batch size (optional, will be added if missing)
                 C = number of channels (must match model's expected input channels)
                 H, W = height and width dimensions
            return_dict: If True, returns a dictionary containing model outputs
                        (e.g., {"reconstructed_image": tensor, "bpp": value} for compression models);
                        if False, returns just the reconstructed image tensor

        Returns:
            If return_dict=False: torch.Tensor containing the processed image(s)
            If return_dict=True: dict containing model outputs (always includes "reconstructed_image")

        Raises:
            AssertionError: If the input image channels don't match the model's expected input channels

        Notes:
            - Input image is automatically converted to a batch if it's a single image
            - Inference is performed with torch.no_grad() for efficiency
            - The model is automatically set to evaluation mode during inference
            - Dictionary output may contain additional metrics like bits-per-pixel (bpp) for compression models
        """
        with torch.no_grad():
            # Track if we need to squeeze output
            squeeze_output = False
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
                squeeze_output = True

            # Get input channels from the tensor
            in_channels = img.shape[1]

            # Check if model has in_channels attribute (for compatibility)
            if hasattr(self.model, 'in_channels'):
                expected_channels = self.model.in_channels
                assert in_channels == expected_channels, (
                    f"{in_channels=}, {expected_channels=}; model configuration does not match input image."
                )

            img = img.to(self.device)
            output = self.model.eval()(img)
            
            # Extract the main result
            if isinstance(output, dict):
                result = output["reconstructed_image"]
            else:
                result = output
            
            # Remove batch dimension if we added it
            if squeeze_output:
                result = result.squeeze(0)
                if isinstance(output, dict):
                    output["reconstructed_image"] = result

            if return_dict:
                if isinstance(output, torch.Tensor):
                    return {"reconstructed_image": result}
                return output
            return result

    def process(self, img: torch.Tensor, return_dict: bool = False) -> Union[torch.Tensor, Dict[str, Any]]:
        """Alias for infer method to provide consistent API.
        
        Args:
            img: Input image tensor
            return_dict: If True, return dict with additional info
            
        Returns:
            Processed image tensor or dict
        """
        return self.infer(img, return_dict=return_dict)

    @staticmethod
    def get_transfer_function(fun_name: str) -> callable:
        """Get a transfer function for image pixel value transformation.

        This method provides a centralized way to access various transfer functions
        used for image processing. Transfer functions transform pixel values, often
        to convert between different color spaces or to apply non-linear corrections.

        Available transfer functions:
        - "None": Identity function (returns input unchanged)
        - "pq": Perceptual Quantizer encoding (converts from scene-linear to PQ encoding)
        - "gamma22": Gamma correction with γ=2.2 (standard sRGB gamma)

        Args:
            fun_name: String identifier for the desired transfer function

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: A function that applies the
            requested transfer to tensor inputs

        Raises:
            ValueError: If the requested transfer function name is not recognized

        Examples:
            >>> transfer_fn = get_transfer_function("gamma22")
            >>> corrected_image = transfer_fn(linear_image)
        """
        # Import rawproc from dependencies (will be moved later)
        from ..dependencies.raw_processing import scenelin_to_pq, gamma

        if str(fun_name) == "None":
            return lambda img: img
        elif fun_name == "pq":
            return scenelin_to_pq
        elif fun_name == "gamma22":
            return lambda img: gamma(img, gamma_val=2.2, in_place=True)
        else:
            raise ValueError(fun_name)
