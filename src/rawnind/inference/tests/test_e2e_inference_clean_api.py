import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from rawnind.inference.clean_api import (
    InferenceConfig,
    create_rgb_denoiser,
    create_bayer_denoiser,
    create_compressor,
    load_model_from_checkpoint,
    compute_image_metrics,
    convert_device_format,
)
from rawnind.dataset.clean_api import DatasetConfig, create_test_dataset

# Mock models for testing to avoid actual model loading and complex inference
@pytest.fixture
def mock_denoiser_model():
    """Mock a basic denoising model."""
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.eval.return_value = mock_model
    # Ensure forward returns a tensor of appropriate shape for RGB denoiser
    mock_model.return_value = torch.randn(1, 3, 128, 128)
    return mock_model

@pytest.fixture
def mock_bayer_denoiser_model():
    """Mock a Bayer denoising model."""
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.eval.return_value = mock_model
    # Bayer input results in RGB output (often 2x resolution)
    mock_model.return_value = torch.randn(1, 3, 256, 256) # Assume 2x resolution
    return mock_model

@pytest.fixture
def mock_compression_model():
    """Mock a denoise+compress model that returns a dict."""
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.eval.return_value = mock_model
    mock_model.return_value = {
        "reconstructed_image": torch.randn(1, 3, 128, 128),
        "bpp": torch.tensor(0.5)
    }
    return mock_model

@pytest.fixture
def mock_base_inference_load_model():
    """Mock ImageToImageNN.load_model to prevent actual file I/O."""
    with patch('rawnind.inference.base_inference.ImageToImageNN.load_model') as mock_load:
        yield mock_load

@pytest.fixture(autouse=True)
def setup_inference_model_mocks(mock_denoiser_model, mock_bayer_denoiser_model, mock_compression_model):
    """Patch model creation calls within model_factory.py."""
    with patch('rawnind.models.raw_denoiser.UtNet2') as mock_utnet2, \
         patch('rawnind.models.raw_denoiser.UtNet3') as mock_utnet3, \
         patch('rawnind.models.bm3d_denoiser.BM3D_Denoiser') as mock_bm3d, \
         patch('rawnind.models.standard_compressor.JPEGXL_ImageCompressor') as mock_jpegxl, \
         patch('rawnind.models.compression_autoencoders.AbstractRawImageCompressor') as mock_compressor_arch, \
         patch('rawnind.models.manynets_compression.ManyPriors_RawImageCompressor') as mock_manypriors:

        mock_utnet2.return_value = mock_denoiser_model
        mock_utnet3.return_value = mock_denoiser_model
        mock_bm3d.return_value = mock_denoiser_model
        mock_jpegxl.return_value = mock_compression_model # Use compression model for JPEGXL
        mock_compressor_arch.return_value = mock_compression_model 
        mock_manypriors.return_value = mock_compression_model

        yield

class TestE2EInferenceCleanAPI:
    """End-to-end tests for the inference package's clean API."""

    def test_rgb_denoiser_inference_e2e(self, mock_base_inference_load_model):
        """
        Test a full RGB denoiser inference workflow end-to-end.
        """
        config = InferenceConfig(
            architecture="unet",
            input_channels=3,
            device="cpu",
            filter_units=48,
        )
        
        denoiser = create_rgb_denoiser(
            architecture=config.architecture,
            device=config.device,
            filter_units=config.filter_units
        )

        assert denoiser is not None
        assert denoiser.architecture == config.architecture
        assert str(denoiser.device) == config.device

        dummy_rgb_image = torch.randn(3, 128, 128) # Single image
        denoised_output = denoiser.denoise(dummy_rgb_image)

        assert isinstance(denoised_output, torch.Tensor)
        assert denoised_output.shape == (3, 128, 128)
        assert denoised_output.device == torch.device(config.device)
        
        # Test batch input
        dummy_rgb_batch = torch.randn(2, 3, 128, 128)
        denoised_batch_output = denoiser.denoise_batch(dummy_rgb_batch)
        assert denoised_batch_output.shape == (2, 3, 128, 128)

        # Verify input channel validation
        with pytest.raises(ValueError, match="Expected 3 channels, got 4"):
            denoiser.denoise(torch.randn(4, 128, 128))

    def test_bayer_denoiser_inference_e2e(self, mock_base_inference_load_model):
        """
        Test a full Bayer denoiser inference workflow end-to-end (domain preservation).
        """
        config = InferenceConfig(
            architecture="utnet3",
            input_channels=4,  # Bayer
            device="cpu",
            filter_units=48,
            enable_preupsampling=True # Test pre-upsampling flag
        )

        denoiser = create_bayer_denoiser(
            architecture=config.architecture,
            device=config.device,
            filter_units=config.filter_units,
            enable_preupsampling=config.enable_preupsampling
        )

        assert denoiser is not None
        assert denoiser.architecture == config.architecture
        assert str(denoiser.device) == config.device
        assert denoiser.supports_bayer == True
        assert denoiser.demosaic_fn is not None

        dummy_bayer_image = torch.randn(4, 128, 128) # Single Bayer image
        dummy_rgb_xyz_matrix = torch.eye(3) # Identity matrix for simplicity

        # Mock the internal raw.demosaic and rawproc.camRGB_to_lin_rec2020_images for pure unit test for flow
        with patch('rawnind.dependencies.raw_processing.demosaic', return_value=torch.randn(4, 3, 256, 256)), \
             patch('rawnind.dependencies.raw_processing.camRGB_to_lin_rec2020_images', return_value=torch.randn(4, 3, 256, 256)) as mock_camrgb:
            
            denoised_output = denoiser.denoise_bayer(dummy_bayer_image, dummy_rgb_xyz_matrix)
            assert isinstance(denoised_output, torch.Tensor)
            assert denoised_output.shape == (3, 256, 256) # Expect 3 channels, 2x resolution
            assert denoised_output.device == torch.device(config.device)
            mock_camrgb.assert_called_once() # Ensure color transform is applied

        # Verify input channel validation for Bayer
        with pytest.raises(ValueError, match="Bayer image must have 4 channels, got 3"):
            denoiser.denoise_bayer(torch.randn(3, 128, 128), dummy_rgb_xyz_matrix)
        
        # Test InferenceConfig validation for enable_preupsampling
        with pytest.raises(ValueError, match="Preupsampling can only be used with 4-channel"):
            InferenceConfig(architecture="unet", input_channels=3, device="cpu", enable_preupsampling=True)

    def test_compressor_inference_e2e(self, mock_base_inference_load_model):
        """
        Test a full compressor inference workflow end-to-end.
        """
        config = InferenceConfig(
            architecture="ManyPriors",
            input_channels=3,
            device="cpu",
            encoder_arch="Balle",
            decoder_arch="Balle",
            hidden_out_channels=192,
            bitstream_out_channels=64,
        )

        compressor = create_compressor(
            architecture=config.architecture,
            input_channels=config.input_channels,
            device=config.device,
            encoder_arch=config.encoder_arch,
            decoder_arch=config.decoder_arch,
            hidden_out_channels=config.hidden_out_channels,
            bitstream_out_channels=config.bitstream_out_channels,
        )

        assert compressor is not None
        assert compressor.architecture == config.architecture
        assert str(compressor.device) == config.device

        dummy_image = torch.randn(3, 128, 128)
        compression_results = compressor.compress_and_denoise(dummy_image)

        assert isinstance(compression_results, dict)
        assert "denoised_image" in compression_results
        assert "bpp" in compression_results
        assert "compression_ratio" in compression_results
        
        assert isinstance(compression_results["denoised_image"], torch.Tensor)
        assert compression_results["denoised_image"].shape == (3, 128, 128)
        assert compression_results["denoised_image"].device == torch.device(config.device)
        assert compression_results["bpp"] > 0 # Should have a positive bpp
        assert compression_results["compression_ratio"] > 0

    def test_load_model_from_checkpoint_robustness(self, tmp_path, mock_base_inference_load_model):
        """
        Test robustness of load_model_from_checkpoint function.
        Includes mock directory structure for args.yaml, trainres.yaml, and saved_models.
        """
        # Create mock experiment directory structure
        exp_dir = tmp_path / "mock_experiment"
        (exp_dir / "saved_models").mkdir(parents=True)
        
        # Create args.yaml
        args_content = {
            "arch": "unet",
            "in_channels": 3,
            "funit": 48,
            "loss": "ms_ssim", # Example loss
        }
        with open(exp_dir / "args.yaml", "w") as f:
            yaml.safe_dump(args_content, f)

        # Create trainres.yaml
        trainres_content = {
            "best_step": {
                "val_ms_ssim": 100, # Best checkpoint at step 100
                "val_psnr": 90,
            },
            "100": {"val_ms_ssim": 0.95, "val_psnr": 30.0} # Metrics for step 100
        }
        with open(exp_dir / "trainres.yaml", "w") as f:
            yaml.safe_dump(trainres_content, f)
        
        # Create dummy checkpoint files
        (exp_dir / "saved_models" / "iter_50.pt").touch()
        (exp_dir / "saved_models" / "iter_100.pt").touch() # Best checkpoint
        (exp_dir / "saved_models" / "iter_150.pt").touch()

        # Test loading from directory (should pick best checkpoint)
        loaded_model_info_dir = load_model_from_checkpoint(str(exp_dir), device="cpu")
        assert loaded_model_info_dir is not None
        assert 'model' in loaded_model_info_dir
        assert 'config' in loaded_model_info_dir
        assert loaded_model_info_dir['config'].architecture == "unet"
        assert loaded_model_info_dir['config'].input_channels == 3
        mock_base_inference_load_model.assert_called_once_with(
            loaded_model_info_dir['model'], str(exp_dir / "saved_models" / "iter_100.pt"), device=torch.device("cpu")
        )
        mock_base_inference_load_model.reset_mock() # Reset for next test

        # Test loading from specific checkpoint file
        loaded_model_info_file = load_model_from_checkpoint(str(exp_dir / "saved_models" / "iter_50.pt"), device="cpu")
        assert loaded_model_info_file is not None
        mock_base_inference_load_model.assert_called_once_with(
            loaded_model_info_file['model'], str(exp_dir / "saved_models" / "iter_50.pt"), device=torch.device("cpu")
        )

        # Test auto-detection failure (e.g., no args.yaml)
        (exp_dir / "args.yaml").unlink() # Remove args.yaml
        with pytest.raises(ValueError, match="Architecture not provided and could not be auto-detected"):
            load_model_from_checkpoint(str(exp_dir), device="cpu")
        
        # Test fallback to latest if trainres.yaml is missing
        (exp_dir / "trainres.yaml").unlink() # Remove trainres.yaml
        loaded_model_info_latest = load_model_from_checkpoint(str(exp_dir), device="cpu")
        assert loaded_model_info_latest is not None
        # Should now load the latest (iter_150.pt) if no best_step is available
        mock_base_inference_load_model.assert_called_with(
            loaded_model_info_latest['model'], str(exp_dir / "saved_models" / "iter_150.pt"), device=torch.device("cpu")
        )

    def test_compute_image_metrics(self):
        """
        Test compute_image_metrics for various metrics and MS-SSIM size constraint.
        """
        # Dummy images (batch, channels, height, width)
        pred_rgb = torch.randn(1, 3, 200, 200)
        gt_rgb = torch.randn(1, 3, 200, 200)
        
        # Test MSE and PSNR (should always work with valid inputs)
        metrics_results = compute_image_metrics(pred_rgb, gt_rgb, metrics=["mse", "psnr"])
        assert "mse" in metrics_results
        assert "psnr" in metrics_results
        assert isinstance(metrics_results["mse"], float)
        assert isinstance(metrics_results["psnr"], float)
        
        # Test MS-SSIM with valid size
        metrics_results_msssim_valid = compute_image_metrics(pred_rgb, gt_rgb, metrics=["ms_ssim"])
        assert "ms_ssim" in metrics_results_msssim_valid
        assert isinstance(metrics_results_msssim_valid["ms_ssim"], float)
        
        # Test MS-SSIM with invalid size (should warn and return NaN)
        small_pred_rgb = torch.randn(1, 3, 100, 100) # Too small for MS-SSIM
        small_gt_rgb = torch.randn(1, 3, 100, 100)
        
        with patch('logging.warning') as mock_warning:
            metrics_results_msssim_invalid = compute_image_metrics(small_pred_rgb, small_gt_rgb, metrics=["ms_ssim"])
            mock_warning.assert_called_once_with("Skipping ms_ssim: image size 100 too small (need ≥162)")
            assert "ms_ssim" in metrics_results_msssim_invalid
            assert np.isnan(metrics_results_msssim_invalid["ms_ssim"])
            
        # Test masking application
        mask = torch.zeros_like(pred_rgb)
        mask[:, :, 50:150, 50:150] = 1.0 # Only middle is valid
        masked_metrics = compute_image_metrics(pred_rgb, gt_rgb, metrics=["mse"], mask=mask)
        assert "mse" in masked_metrics
        # (Further assertions could check if masked MSE is correctly computed)
        
        # Test unknown metric
        with patch('logging.warning') as mock_warning:
            unknown_metric_results = compute_image_metrics(pred_rgb, gt_rgb, metrics=["unknown_metric"])
            mock_warning.assert_called_once_with("Unknown metric: unknown_metric")
            assert "unknown_metric" in unknown_metric_results
            assert np.isnan(unknown_metric_results["unknown_metric"])

    def test_convert_device_format(self):
        """Test convert_device_format utility."""
        assert convert_device_format("cpu") == -1
        assert convert_device_format("cuda") == 0
        assert convert_device_format("cuda:1") == 1
        assert convert_device_format(torch.device("cpu")) == -1
        assert convert_device_format(torch.device("cuda:0")) == 0
        assert convert_device_format(torch.device("cuda:1")) == 1
        assert convert_device_format(-1) == -1
        assert convert_device_format(0) == 0
        assert convert_device_format(1) == 1
        assert convert_device_format("custom_device") == "custom_device" # Pass through unknown strings

        with pytest.raises(ValueError, match="Unsupported device specification"):
            convert_device_format(None) # Invalid type