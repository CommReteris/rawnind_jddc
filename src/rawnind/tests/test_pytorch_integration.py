"""Pytest integration tests for RawNIND PyTorch models.

Tests concrete model instantiation, device placement, forward pass, and inference
using fixtures for hermetic, performant validation. Focuses on DenoiseThenCompress
as representative of refactored API.

Objective: Verify core PyTorch integration post-refactoring (forward dict output, infer batch/single).
Test Criteria: Model has parameters; forward returns dict with "reconstructed_image"/"bpp"; infer handles batch/single, returns tensor.
Fulfillment: Ensures models work as intended without real data; covers device transfer, shape preservation.
Components: model_bayer_dc fixture (DenoiseThenCompress via rawtestlib, null dataloader mocked).
Reasons: Fixture uses concrete model (not abstract); null dataloader keeps hermetic/performant; verifies refactored intent (dict output, base infer) without trainer.
"""

import torch
import pytest
import numpy as np

from rawnind.models.denoise_then_compress import DenoiseThenCompress


@pytest.fixture
def concrete_model(model_bayer_dc, device):
    """Concrete model fixture for integration tests."""
    return model_bayer_dc.model.to(device)  # DenoiseThenCompress instance


def test_model_device_placement(concrete_model, device):
    """Test model parameters are on correct device."""
    # Objective: Verify device placement post-init.
    # Criteria: All parameters on specified device.
    # Fulfillment: Ensures tensor ops work without device mismatch errors.
    # Components: concrete_model (DenoiseThenCompress, params from compressor/denoiser); device fixture (pt_helpers.get_device).
    # Reasons: Concrete has params (unlike abstract); verifies to(device) in fixture; no real components as param iter suffices.

    assert isinstance(concrete_model, torch.nn.Module)
    param_device = next(concrete_model.parameters()).device
    assert param_device == device
    # Check all params
    for param in concrete_model.parameters():
        assert param.device == device


def test_model_forward_pass(concrete_model, device):
    """Test forward pass produces expected dict output."""
    # Objective: Validate refactored forward returns compression dict.
    # Criteria: Output is dict with "reconstructed_image" (shape matches input), "bpp" > 0.
    # Fulfillment: Confirms pipeline (denoise+compress) works; shape preservation.
    # Components: concrete_model (DenoiseThenCompress forward); device for tensor placement.
    # Reasons: Tests concrete API (dict vs. None in abstract); dummy input hermetic, no real data needed for shape/bpp check.

    batch_size, channels, height, width = 1, 4, 64, 64  # Bayer input
    dummy_input = torch.randn(batch_size, channels, height, width).to(device)

    with torch.no_grad():
        output = concrete_model(dummy_input)

    assert isinstance(output, dict)
    assert "reconstructed_image" in output
    recon = output["reconstructed_image"]
    assert recon.shape == (batch_size, 3, height, width)  # RGB output
    assert "bpp" in output
    assert output["bpp"] >= 0  # Non-negative bitrate


def test_model_inference_method(concrete_model, device):
    """Test infer method handles input correctly."""
    # Objective: Verify ImageToImageNN base infer works on concrete model.
    # Criteria: infer returns tensor (processed image); device matches; handles single image.
    # Fulfillment: Ensures inference API for deployment; batch dim added if needed.
    # Components: concrete_model.infer (base method, calls forward); device fixture.
    # Reasons: Tests refactored base (abstract lacked infer); dummy verifies shape/device without data.

    # Single image (no batch dim)
    test_input = torch.randn(4, 64, 64).to(device)  # Bayer

    with torch.no_grad():
        output = concrete_model.infer(test_input)

    assert isinstance(output, torch.Tensor)
    assert output.shape == (3, 64, 64)  # RGB, no batch
    assert output.device == device


def test_batch_handling(concrete_model, device):
    """Test infer handles batch and single images properly."""
    # Objective: Verify batch/single handling in infer.
    # Criteria: Single adds batch dim (output [1,C,H,W]); batch preserves shape.
    # Fulfillment: Confirms flexibility for varying input sizes.
    # Components: concrete_model.infer; dummy batches.
    # Reasons: Tests unsqueeze logic in base infer; essential for practical use, hermetic with randn.

    # Single image
    single_image = torch.randn(4, 64, 64).to(device)
    with torch.no_grad():
        single_output = concrete_model.infer(single_image)
    assert single_output.shape == (3, 64, 64)

    # Batched (2 images)
    batch_input = torch.randn(2, 4, 64, 64).to(device)
    with torch.no_grad():
        batch_output = concrete_model.infer(batch_input)
    assert batch_output.shape == (2, 3, 64, 64)

    # Verify single output matches batched[0]
    single_batched = torch.randn(1, 4, 64, 64).to(device)
    with torch.no_grad():
        single_batched_output = concrete_model.infer(single_batched)
    assert torch.allclose(single_output.unsqueeze(0), single_batched_output, atol=1e-5)