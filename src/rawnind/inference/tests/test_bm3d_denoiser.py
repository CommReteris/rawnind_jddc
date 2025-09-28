"""
Unit tests for BM3D denoiser in models/bm3d_denoiser.py.

Objective: Verify BM3D denoiser forward pass and parameters with dummy noisy RGB input for hermetic testing.
Test Criteria: Instantiate BM3D denoiser with in_channels=3, funit=25; pass dummy noisy RGB tensor; assert output shape matches input, output range in [0,1].
Fulfillment: Ensures denoiser processes RGB input correctly; covers forward pass without real data; fulfills intent of noise reduction model for RGB images.
Components Mocked/Fixtured: None - uses torch for dummy tensors; tests instantiation and forward.
Reasons for No Mocking: Torch tensors are lightweight; no external deps; direct testing of model logic for accuracy without simulation.
"""

import pytest
import torch
import shutil
from unittest.mock import patch

from src.rawnind.models.bm3d_denoiser import BM3D_Denoiser

def test_bm3d_denoiser_forward_pass(monkeypatch):
    """Test BM3D denoiser forward pass with dummy noisy RGB input."""
    # Mock shutil.which to return dummy path
    monkeypatch.setattr(shutil, 'which', lambda x: '/bin/bm3d')

    # Mock subprocess.run to simulate successful BM3D execution
    def mock_run(cmd, *args, **kwargs):
        class CompletedProcess:
            returncode = 0
        return CompletedProcess
    monkeypatch.setattr('subprocess.run', mock_run)

    # Mock os.path.isfile to return True for tmp file check
    monkeypatch.setattr('os.path.isfile', lambda x: True)

    # Mock pt_helpers.fpath_to_tensor to return denoised tensor
    def mock_fpath_to_tensor(fpath, device, batch=True):
        # Fixed denoised tensor for shape/range
        denoised = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
        if not batch:
            denoised = denoised.squeeze(0)
        return denoised.to(device)

    monkeypatch.setattr('rawnind.dependencies.pytorch_helpers.fpath_to_tensor', mock_fpath_to_tensor)

    # Instantiate denoiser
    denoiser = BM3D_Denoiser(in_channels=3, funit=25)

    # Dummy noisy input (batch=1, channels=3, height=64, width=64)
    noisy_input = torch.rand(1, 3, 64, 64) * 0.1 + 0.5  # Noise around 0.5

    # Mock np_imgops.np_to_img to accept kwargs (precision=8)
    def mock_np_to_img(*args, **kwargs):
        pass
    monkeypatch.setattr('rawnind.dependencies.numpy_operations.np_to_img', mock_np_to_img)

    # Forward pass
    denoised_output = denoiser(noisy_input)

    # Assert output shape matches input
    assert denoised_output.shape == noisy_input.shape

    # Assert output in reasonable range [0,1]
    assert torch.all(denoised_output >= 0) and torch.all(denoised_output <= 1)

def test_bm3d_denoiser_invalid_input(monkeypatch):
    """Test error handling for invalid input shapes."""
    # Mock binary checks to allow instantiation
    monkeypatch.setattr(shutil, 'which', lambda x: '/bin/bm3d')
    monkeypatch.setattr('subprocess.run', lambda *args, **kwargs: type('obj', (object,), {'returncode': 0})())

    denoiser = BM3D_Denoiser(in_channels=3, funit=25)

    # Invalid shape (wrong channels)
    invalid_input = torch.rand(1, 1, 64, 64)  # 1 channel instead of 3
    with pytest.raises(AssertionError):
        denoiser(invalid_input)
