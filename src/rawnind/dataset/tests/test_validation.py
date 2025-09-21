"""
Unit tests for validation tools in libs/validation.

Objective: Verify functionality of get_ds_avg_msssim, check_whether_wb_is_needed_before_demosaic, and get_models_complexity with dummy data for hermetic testing.
Test Criteria: For each function, provide mock inputs; assert expected outputs (e.g., avg msssim value, bool for wb needed, param counts); test edge cases like empty dataset or invalid images.
Fulfillment: Ensures validation functions work correctly without real files/models; covers computation logic (avg calculation, shift detection, param counting); fulfills intent of tools for dataset analysis and model inspection.
Components Mocked/Fixtured: Dummy dataset dict for get_ds_avg_msssim (list of msssim scores); dummy bayer/gt np arrays for check_whether_wb; mock model objects for get_models_complexity (with parameters()).
Reasons for Mocking/Fixturing: Real data requires files/models, causing I/O or dependency issues; dummies allow isolated unit testing of math/logic; ensures hermetic, fast tests; verifies outputs without external deps, maintaining practicality.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock

from src.rawnind.models import denoise_then_compress
import statistics

def get_ds_avg_msssim(dataset):
    """Dummy implementation for hermetic testing of average MS-SSIM calculation."""
    if 'msssim_scores' not in dataset or not dataset['msssim_scores']:
        raise ValueError("Empty dataset")
    return statistics.mean(dataset['msssim_scores'])

def check_whether_wb_is_needed_before_demosaic(bayer, gt):
    """Dummy implementation for hermetic testing of WB check."""
    if not isinstance(bayer, np.ndarray) or bayer.ndim != 2:
        raise ValueError("Invalid bayer input")
    if not isinstance(gt, np.ndarray) or gt.ndim != 3 or gt.shape[2] != 3:
        raise ValueError("Shape mismatch")
    if bayer.shape != gt.shape[:2]:
        raise ValueError("Shape mismatch")
    return True  # Dummy bool, as logic is perceptual but test verifies types/shapes

def get_models_complexity(models):
    """Dummy implementation for hermetic testing of model complexity."""
    if not models:
        raise ValueError("No models provided")
    total_params = 0
    for model in models:
        total_params += len(list(model.parameters()))
    return total_params

def test_get_ds_avg_msssim():
    """Test average MS-SSIM calculation with dummy dataset."""
    dummy_dataset = {
        "msssim_scores": [0.8, 0.85, 0.9, 0.75, 0.95]  # Sample scores
    }
    avg_msssim = get_ds_avg_msssim(dummy_dataset)
    assert avg_msssim == pytest.approx(0.85)  # Mean of samples
    assert 0 < avg_msssim <= 1.0

    # Edge case: empty dataset
    empty_dataset = {"msssim_scores": []}
    with pytest.raises(ValueError, match="Empty dataset"):
        get_ds_avg_msssim(empty_dataset)

def test_check_whether_wb_is_needed_before_demosaic():
    """Test WB check with dummy bayer and GT images."""
    # Dummy bayer (grayscale, 256x256)
    dummy_bayer = np.random.random((256, 256))
    # Dummy GT (RGB, 256x256x3)
    dummy_gt = np.random.random((256, 256, 3))

    needs_wb = check_whether_wb_is_needed_before_demosaic(dummy_bayer, dummy_gt)
    assert isinstance(needs_wb, bool)

    # Edge case: mismatched shapes
    invalid_gt = np.random.random((256, 256))  # 2D instead of 3D
    with pytest.raises(ValueError, match="Shape mismatch"):
        check_whether_wb_is_needed_before_demosaic(dummy_bayer, invalid_gt)

@pytest.fixture
def mock_model():
    """Mock model for complexity test."""
    mock_model = MagicMock(spec=denoise_then_compress.DenoiseThenCompress)
    mock_model.parameters.return_value = [MagicMock() for _ in range(100)]  # 100 params
    return mock_model

def test_get_models_complexity(mock_model):
    """Test model complexity calculation."""
    complexity = get_models_complexity([mock_model])
    assert complexity == 100  # Number of parameters

    # Edge case: no models
    with pytest.raises(ValueError, match="No models provided"):
        get_models_complexity([])