import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from rawnind.inference.clean_api import compute_image_metrics
from rawnind.dependencies.pt_losses import metrics as metrics_dict_from_module

@pytest.fixture
def dummy_images():
    """Returns dummy predicted and ground truth images."""
    return torch.randn(1, 3, 200, 200), torch.randn(1, 3, 200, 200)

class TestComputeImageMetrics:
    """Unit tests for the compute_image_metrics function."""

    @patch.dict('rawnind.inference.clean_api.metrics_dict_from_module', {
        'mse': MagicMock(return_value=torch.tensor(0.01)),
        'psnr': MagicMock(return_value=torch.tensor(25.0)),
        'ms_ssim': MagicMock(return_value=torch.tensor(0.98))
    })
    def test_metrics_computation(self, dummy_images):
        """Test basic computation of various metrics."""
        pred, gt = dummy_images
        metrics_list = ["mse", "psnr", "ms_ssim"]
        results = compute_image_metrics(pred, gt, metrics_list)

        assert "mse" in results and results["mse"] == 0.01
        assert "psnr" in results and results["psnr"] == 25.0
        assert "ms_ssim" in results and results["ms_ssim"] == 0.98
        
        # Ensure underlying metric functions were called
        metrics_dict_from_module['mse'].assert_called_once()
        metrics_dict_from_module['psnr'].assert_called_once()
        metrics_dict_from_module['ms_ssim'].assert_called_once()

    def test_image_shapes_mismatch(self):
        """Test that mismatched image shapes raise ValueError."""
        pred = torch.randn(1, 3, 100, 100)
        gt = torch.randn(1, 3, 150, 150) # Mismatched height/width
        with pytest.raises(ValueError, match="Image shapes must match"):
            compute_image_metrics(pred, gt, ["mse"])

    @patch('logging.warning')
    def test_ms_ssim_size_constraint(self, mock_warning):
        """Test MS-SSIM size constraint (image smaller than 162)."""
        pred_small = torch.randn(1, 3, 100, 100)
        gt_small = torch.randn(1, 3, 100, 100)
        
        results = compute_image_metrics(pred_small, gt_small, ["ms_ssim"])
        
        # Assert warning was logged
        mock_warning.assert_called_once_with("Skipping ms_ssim: image size 100 too small (need ≥162)")
        
        # Assert NaN value for MS-SSIM
        assert "ms_ssim" in results
        assert np.isnan(results["ms_ssim"])

    @patch('logging.warning')
    def test_unknown_metric_handling(self, mock_warning, dummy_images):
        """Test handling of unknown metrics."""
        pred, gt = dummy_images
        results = compute_image_metrics(pred, gt, ["unknown_metric"])
        
        mock_warning.assert_called_once_with("Unknown metric: unknown_metric")
        assert "unknown_metric" in results
        assert np.isnan(results["unknown_metric"])

    def test_masking_application(self):
        """Test that masks are applied correctly before metric computation."""
        pred = torch.ones(1, 3, 10, 10) * 0.5
        gt = torch.ones(1, 3, 10, 10) * 1.0
        mask = torch.zeros(1, 1, 10, 10)
        mask[:, :, 0:5, :] = 1.0 # Mask first half
        
        # Mock MSE to return sum of masked squared differences
        with patch.dict('rawnind.inference.clean_api.metrics_dict_from_module', {
            'mse': MagicMock(return_value=torch.tensor(0.25))
        }):
            results = compute_image_metrics(pred, gt, ["mse"], mask=mask)
            assert results["mse"] == 0.25 # Only the unmasked part contributes
        
        # Ensure input tensors to the metric function are masked
        metrics_dict_from_module['mse'].assert_called_once()
        called_pred, called_gt = metrics_dict_from_module['mse'].call_args[0]
        # Check that the masked regions are zero or close to zero
        # Note: direct tensor equality checks with floating point numbers can be tricky.
        # We'll check if the original unmasked regions remain, and masked regions are zeroed.
        assert torch.all(called_pred[:, :, 5:, :] == 0) # Masked region should be zero
        assert torch.all(called_gt[:, :, 5:, :] == 0)   # Masked region should be zero
        assert torch.all(called_pred[:, :, :5, :] == pred[:, :, :5, :]) # Unmasked is original
        assert torch.all(called_gt[:, :, :5, :] == gt[:, :, :5, :])     # Unmasked is original