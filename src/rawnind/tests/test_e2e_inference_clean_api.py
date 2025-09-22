"""End-to-end tests for clean inference package API.

This module demonstrates and tests the clean, programmatic API for the inference package.
These tests serve as the specification for how the inference package should work
without any CLI dependencies.

The clean API design principles:
1. Factory functions for creating models by type and purpose
2. Direct model loading from checkpoint paths  
3. Simple inference methods with explicit parameters
4. Separate metrics computation utilities
5. Zero CLI dependencies - pure programmatic interfaces
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml

from rawnind.inference import create_rgb_denoiser, create_bayer_denoiser, create_compressor
from rawnind.inference import load_model_from_checkpoint, compute_image_metrics
from rawnind.inference import InferenceEngine
from rawnind.dependencies.pytorch_helpers import fpath_to_tensor


class TestCleanInferenceAPI:
    """Test the clean, programmatic inference API without CLI dependencies."""

    def test_create_rgb_denoiser_basic(self):
        """Test creating an RGB denoiser with minimal parameters."""
        # Should create a basic RGB denoiser ready for inference
        denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu'
        )
        
        assert denoiser is not None
        assert denoiser.device.type == 'cpu'
        assert denoiser.input_channels == 3  # RGB
        assert hasattr(denoiser, 'denoise')
        
    def test_create_bayer_denoiser_basic(self):
        """Test creating a Bayer denoiser with minimal parameters."""
        # Should create a basic Bayer denoiser ready for inference
        denoiser = create_bayer_denoiser(
            architecture='unet',
            device='cpu'
        )
        
        assert denoiser is not None
        assert denoiser.device.type == 'cpu'
        assert denoiser.input_channels == 4  # Bayer
        assert hasattr(denoiser, 'denoise')
        
    def test_create_compressor_basic(self):
        """Test creating a compression model with minimal parameters."""
        # Should create a basic compressor ready for inference
        compressor = create_compressor(
            architecture='standard',
            device='cpu'
        )
        
        assert compressor is not None
        assert compressor.device.type == 'cpu'
        assert hasattr(compressor, 'compress')
        assert hasattr(compressor, 'decompress')

    def test_load_model_from_checkpoint_rgb(self):
        """Test loading an RGB model from a real checkpoint."""
        # Find a real RGB model checkpoint
        weights_dir = Path("src/rawnind/models/weights")
        rgb_model_dirs = list(weights_dir.glob("*ProfiledRGBToProfiledRGB*"))
        
        if not rgb_model_dirs:
            pytest.skip("No RGB model checkpoints available")
            
        model_dir = rgb_model_dirs[0]
        checkpoint_files = list(model_dir.glob("saved_models/*.pt"))
        
        if not checkpoint_files:
            pytest.skip("No checkpoint files found")
            
        checkpoint_path = str(checkpoint_files[0])
        
        # Should load the model with clean interface
        denoiser = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        
        assert denoiser is not None
        assert hasattr(denoiser, 'denoise')
        assert denoiser.device.type == 'cpu'

    def test_load_model_from_checkpoint_bayer(self):
        """Test loading a Bayer model from a real checkpoint."""
        # Find a real Bayer model checkpoint  
        weights_dir = Path("src/rawnind/models/weights")
        bayer_model_dirs = list(weights_dir.glob("*BayerToProfiledRGB*"))
        
        if not bayer_model_dirs:
            pytest.skip("No Bayer model checkpoints available")
            
        model_dir = bayer_model_dirs[0]
        checkpoint_files = list(model_dir.glob("saved_models/*.pt"))
        
        if not checkpoint_files:
            pytest.skip("No checkpoint files found")
            
        checkpoint_path = str(checkpoint_files[0])
        
        # Should load the model with clean interface
        denoiser = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device='cpu'
        )
        
        assert denoiser is not None
        assert hasattr(denoiser, 'denoise')
        assert denoiser.device.type == 'cpu'

    def test_rgb_denoising_inference(self):
        """Test RGB image denoising with clean API."""
        # Create RGB denoiser
        denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu'
        )
        
        # Create test RGB image (3 channels)
        rgb_image = torch.rand(3, 128, 128)
        
        # Should denoise the image directly
        denoised_image = denoiser.denoise(rgb_image)
        
        assert denoised_image.shape == rgb_image.shape
        assert denoised_image.shape[0] == 3  # RGB channels
        assert torch.all(torch.isfinite(denoised_image))

    def test_bayer_denoising_inference(self):
        """Test Bayer image denoising with clean API."""
        # Create Bayer denoiser  
        denoiser = create_bayer_denoiser(
            architecture='unet',
            device='cpu'
        )
        
        # Create test Bayer image (4 channels)
        bayer_image = torch.rand(4, 128, 128)
        
        # Should denoise and convert to RGB
        rgb_image = denoiser.denoise(bayer_image)
        
        assert rgb_image.shape[0] == 3  # Should output RGB
        assert rgb_image.shape[1:] == (256, 256)  # Should upsample 2x
        assert torch.all(torch.isfinite(rgb_image))

    def test_batch_inference(self):
        """Test batch processing with clean API."""
        denoiser = create_rgb_denoiser(
            architecture='unet', 
            device='cpu'
        )
        
        # Create batch of RGB images
        batch_images = torch.rand(4, 3, 128, 128)  # 4 images
        
        # Should handle batch processing
        denoised_batch = denoiser.denoise_batch(batch_images)
        
        assert denoised_batch.shape == batch_images.shape
        assert torch.all(torch.isfinite(denoised_batch))

    def test_compute_image_metrics_standalone(self):
        """Test standalone metrics computation without model dependencies."""
        # Create test images
        original = torch.rand(3, 162, 162)  # MS-SSIM needs 161x161 minimum
        processed = original + torch.randn_like(original) * 0.1
        
        # Should compute metrics without any model object
        metrics = compute_image_metrics(
            predicted_image=processed,
            ground_truth_image=original,
            metrics=['mse', 'msssim']
        )
        
        assert 'mse' in metrics
        assert 'msssim' in metrics
        assert isinstance(metrics['mse'], float)
        assert isinstance(metrics['msssim'], float)
        assert metrics['mse'] >= 0

    def test_model_architecture_options(self):
        """Test different model architectures through clean interface."""
        architectures = ['unet', 'identity']  # Test available architectures
        
        for arch in architectures:
            denoiser = create_rgb_denoiser(
                architecture=arch,
                device='cpu'
            )
            
            test_image = torch.rand(3, 64, 64)
            result = denoiser.denoise(test_image)
            
            assert result.shape == test_image.shape
            assert torch.all(torch.isfinite(result))

    def test_device_handling(self):
        """Test proper device placement and handling."""
        # Test CPU device
        cpu_denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu'
        )
        assert cpu_denoiser.device.type == 'cpu'
        
        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_denoiser = create_rgb_denoiser(
                architecture='unet', 
                device='cuda'
            )
            assert cuda_denoiser.device.type == 'cuda'

    def test_inference_engine_clean_api(self):
        """Test the InferenceEngine with clean programmatic interface."""
        # Create a simple model for testing
        model = torch.nn.Conv2d(3, 3, 3, padding=1)
        
        # Should initialize without CLI dependencies
        engine = InferenceEngine(
            model=model,
            device='cpu'
        )
        
        test_image = torch.rand(3, 64, 64)
        result = engine.process(test_image)
        
        assert result.shape == test_image.shape
        assert torch.all(torch.isfinite(result))


class TestCleanInferenceE2EPipeline:
    """End-to-end pipeline tests using the clean API."""

    def test_rgb_end_to_end_pipeline(self):
        """Test complete RGB processing pipeline with clean API."""
        # Create and configure RGB denoiser
        denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu',
            filter_units=48
        )
        
        # Load test image
        test_image = torch.rand(3, 128, 128)
        ground_truth = test_image + torch.randn_like(test_image) * 0.05
        
        # Run inference
        denoised = denoiser.denoise(test_image)
        
        # Compute metrics
        metrics = compute_image_metrics(
            predicted_image=denoised,
            ground_truth_image=ground_truth,
            metrics=['mse', 'msssim']
        )
        
        # Validate results
        assert denoised.shape == test_image.shape
        assert 'mse' in metrics
        assert 'msssim' in metrics
        assert isinstance(metrics['mse'], float)

    def test_bayer_end_to_end_pipeline(self):
        """Test complete Bayer processing pipeline with clean API."""
        # Create and configure Bayer denoiser
        denoiser = create_bayer_denoiser(
            architecture='unet',
            device='cpu',
            filter_units=48,
            enable_preupsampling=False
        )
        
        # Load test Bayer image (4 channels)
        test_bayer = torch.rand(4, 128, 128)
        
        # Run inference (should output RGB)
        rgb_result = denoiser.denoise(test_bayer)
        
        # Validate results
        assert rgb_result.shape[0] == 3  # RGB output
        assert rgb_result.shape[1:] == (256, 256)  # 2x upsampling from demosaicing
        assert torch.all(torch.isfinite(rgb_result))

    def test_model_loading_from_experiment_directory(self):
        """Test loading models from experiment directories with clean API."""
        weights_dir = Path("src/rawnind/models/weights")
        experiment_dirs = list(weights_dir.glob("*ProfiledRGB*"))
        
        if not experiment_dirs:
            pytest.skip("No experiment directories available")
            
        experiment_dir = experiment_dirs[0]
        
        # Should load best model from experiment directory
        denoiser = load_model_from_checkpoint(
            checkpoint_path=str(experiment_dir),  # Pass directory, not file
            device='cpu',
            use_best_checkpoint=True
        )
        
        assert denoiser is not None
        assert hasattr(denoiser, 'denoise')

    def test_compression_end_to_end_pipeline(self):
        """Test compression/decompression pipeline with clean API."""
        # Create compressor
        compressor = create_compressor(
            architecture='standard',
            device='cpu'
        )
        
        # Test image
        test_image = torch.rand(3, 128, 128)
        
        # Compress and decompress
        compressed_data = compressor.compress(test_image)
        decompressed_image = compressor.decompress(compressed_data)
        
        # Validate results
        assert decompressed_image.shape == test_image.shape
        assert torch.all(torch.isfinite(decompressed_image))
        assert 'bits_per_pixel' in compressed_data  # Should include compression metrics

    def test_file_based_processing_pipeline(self):
        """Test processing images directly from file paths."""
        # Create temporary test image file
        with tempfile.NamedTemporaryFile(suffix='.exr', delete=False) as tmp_file:
            test_image = torch.rand(3, 128, 128)
            # Save test image (using existing image_denoiser save_image function for now)
            from rawnind.inference.image_denoiser import save_image
            save_image(test_image.unsqueeze(0), tmp_file.name)
            
            # Create denoiser
            denoiser = create_rgb_denoiser(
                architecture='unet',
                device='cpu'
            )
            
            # Process from file path
            result = denoiser.denoise_from_file(
                input_path=tmp_file.name,
                output_path=None  # Return tensor instead of saving
            )
            
            assert result.shape == test_image.shape
            assert torch.all(torch.isfinite(result))


class TestCleanInferenceConfiguration:
    """Test configuration and customization of the clean inference API."""

    def test_custom_model_parameters(self):
        """Test creating models with custom architecture parameters."""
        denoiser = create_rgb_denoiser(
            architecture='unet',
            filter_units=64,  # Custom filter units
            device='cpu'
        )
        
        # Should accept custom parameters
        assert denoiser.filter_units == 64
        
        test_image = torch.rand(3, 64, 64)
        result = denoiser.denoise(test_image)
        assert result.shape == test_image.shape

    def test_bayer_specific_options(self):
        """Test Bayer-specific configuration options."""
        denoiser = create_bayer_denoiser(
            architecture='unet',
            device='cpu',
            enable_preupsampling=True,  # Bayer-specific option
            color_correction_matrix=torch.eye(3)  # Custom color matrix
        )
        
        test_bayer = torch.rand(4, 128, 128)
        rgb_result = denoiser.denoise(test_bayer)
        
        assert rgb_result.shape[0] == 3  # RGB output

    def test_metrics_configuration(self):
        """Test configurable metrics computation."""
        original = torch.rand(3, 162, 162)
        processed = original + torch.randn_like(original) * 0.1
        
        # Test different metric combinations
        basic_metrics = compute_image_metrics(
            predicted_image=processed,
            ground_truth_image=original,
            metrics=['mse']
        )
        assert 'mse' in basic_metrics
        assert len(basic_metrics) == 1
        
        full_metrics = compute_image_metrics(
            predicted_image=processed,
            ground_truth_image=original,
            metrics=['mse', 'msssim', 'psnr']
        )
        assert len(full_metrics) == 3
        assert all(metric in full_metrics for metric in ['mse', 'msssim', 'psnr'])


class TestCleanInferenceIntegration:
    """Integration tests demonstrating real-world usage scenarios."""

    @pytest.mark.slow
    def test_production_workflow_rgb_denoising(self):
        """Test a production-like workflow for RGB denoising."""
        # Step 1: Load trained model from checkpoint
        weights_dir = Path("src/rawnind/models/weights")
        rgb_model_dirs = list(weights_dir.glob("*ProfiledRGBToProfiledRGB*"))
        
        if not rgb_model_dirs:
            pytest.skip("No RGB models available for integration test")
            
        checkpoint_dir = rgb_model_dirs[0]
        
        denoiser = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_dir),
            device='cpu',
            use_best_checkpoint=True
        )
        
        # Step 2: Process test image
        test_image = torch.rand(3, 128, 128)
        denoised = denoiser.denoise(test_image)
        
        # Step 3: Compute quality metrics
        metrics = compute_image_metrics(
            predicted_image=denoised,
            ground_truth_image=test_image,
            metrics=['mse', 'msssim']
        )
        
        # Validate complete pipeline
        assert denoised.shape == test_image.shape
        assert 'mse' in metrics
        assert isinstance(metrics['mse'], float)

    @pytest.mark.slow  
    def test_production_workflow_bayer_denoising(self):
        """Test a production-like workflow for Bayer denoising."""
        # Step 1: Load trained Bayer model
        weights_dir = Path("src/rawnind/models/weights")
        bayer_model_dirs = list(weights_dir.glob("*BayerToProfiledRGB*"))
        
        if not bayer_model_dirs:
            pytest.skip("No Bayer models available for integration test")
            
        checkpoint_dir = bayer_model_dirs[0]
        
        denoiser = load_model_from_checkpoint(
            checkpoint_path=str(checkpoint_dir),
            device='cpu',
            use_best_checkpoint=True
        )
        
        # Step 2: Process Bayer image
        bayer_image = torch.rand(4, 128, 128)
        rgb_result = denoiser.denoise(bayer_image)
        
        # Step 3: Compute metrics against ground truth RGB
        ground_truth_rgb = torch.rand(3, 256, 256)  # Expected output size
        metrics = compute_image_metrics(
            predicted_image=rgb_result,
            ground_truth_image=ground_truth_rgb,
            metrics=['mse']
        )
        
        # Validate complete pipeline
        assert rgb_result.shape[0] == 3  # RGB output
        assert 'mse' in metrics

    def test_model_comparison_workflow(self):
        """Test comparing different models using clean API."""
        architectures = ['unet', 'identity']
        test_image = torch.rand(3, 64, 64)
        results = {}
        
        for arch in architectures:
            denoiser = create_rgb_denoiser(
                architecture=arch,
                device='cpu'
            )
            
            denoised = denoiser.denoise(test_image)
            
            metrics = compute_image_metrics(
                predicted_image=denoised,
                ground_truth_image=test_image,
                metrics=['mse']
            )
            
            results[arch] = metrics['mse']
        
        # Both should produce valid results
        assert len(results) == 2
        assert all(isinstance(mse, float) for mse in results.values())

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing options."""
        denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu',
            memory_efficient=True  # Option for large images
        )
        
        # Test with larger image that might need memory optimization
        large_image = torch.rand(3, 512, 512)
        
        result = denoiser.denoise(
            large_image,
            tile_size=256,  # Process in tiles for memory efficiency
            overlap=32
        )
        
        assert result.shape == large_image.shape
        assert torch.all(torch.isfinite(result))


class TestCleanInferenceErrorHandling:
    """Test error handling in the clean inference API."""

    def test_invalid_architecture_error(self):
        """Test error handling for invalid architectures."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            create_rgb_denoiser(
                architecture='nonexistent_arch',
                device='cpu'
            )

    def test_invalid_checkpoint_path_error(self):
        """Test error handling for invalid checkpoint paths."""
        with pytest.raises(FileNotFoundError):
            load_model_from_checkpoint(
                checkpoint_path="/nonexistent/path/model.pt",
                device='cpu'
            )

    def test_invalid_image_shape_error(self):
        """Test error handling for invalid image shapes."""
        denoiser = create_rgb_denoiser(
            architecture='unet',
            device='cpu'
        )
        
        # Wrong number of channels
        with pytest.raises(ValueError, match="Expected 3 channels"):
            invalid_image = torch.rand(4, 128, 128)  # 4 channels for RGB model
            denoiser.denoise(invalid_image)

    def test_invalid_metrics_error(self):
        """Test error handling for invalid metrics."""
        test_image = torch.rand(3, 128, 128)
        
        with pytest.raises(ValueError, match="Unknown metric"):
            compute_image_metrics(
                predicted_image=test_image,
                ground_truth_image=test_image,
                metrics=['nonexistent_metric']
            )


if __name__ == "__main__":
    pytest.main([__file__])