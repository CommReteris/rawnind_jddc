import torch
import pytest
from rawnind.dependencies.pt_losses import MS_SSIM_loss, L1_loss, MSE_loss, PSNR_metric, ms_ssim_metric

def test_l1_loss():
    """
    Test L1 loss function for basic functionality.

    Objective: Verify that L1_loss computes correct L1 distance between tensors.
    Test criteria: Perfect match returns 0.0, constant difference returns expected value.
    How testing for this criteria fulfills purpose: Ensures L1 loss works for model training.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure mathematical computation without external dependencies.
    """
    loss_fn = L1_loss()
    input_tensor = torch.randn(1, 3, 256, 256)
    target_tensor = input_tensor.clone()
    assert loss_fn(input_tensor, target_tensor) == 0.0

    target_tensor = torch.zeros_like(input_tensor)
    input_tensor = torch.ones_like(input_tensor)
    assert loss_fn(input_tensor, target_tensor) == 1.0

def test_mse_loss():
    """
    Test MSE loss function for basic functionality.

    Objective: Verify that MSE_loss computes correct mean squared error between tensors.
    Test criteria: Perfect match returns 0.0, constant difference returns expected value.
    How testing for this criteria fulfills purpose: Ensures MSE loss works for model training.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure mathematical computation without external dependencies.
    """
    loss_fn = MSE_loss()
    input_tensor = torch.randn(1, 3, 256, 256)
    target_tensor = input_tensor.clone()
    assert loss_fn(input_tensor, target_tensor) == 0.0

    target_tensor = torch.zeros_like(input_tensor)
    input_tensor = torch.ones_like(input_tensor)
    assert loss_fn(input_tensor, target_tensor) == 1.0

def test_psnr_metric():
    """
    Test PSNR metric computation for image quality assessment.

    Objective: Verify that PSNR_metric computes correct peak signal-to-noise ratio.
    Test criteria: Perfect match returns infinity, known difference returns expected PSNR value.
    How testing for this criteria fulfills purpose: Ensures PSNR works for model evaluation.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure mathematical computation without external dependencies.
    """
    metric_fn = PSNR_metric()
    # Perfect match should be infinity
    input_tensor = torch.ones(1, 3, 256, 256)
    target_tensor = torch.ones(1, 3, 256, 256)
    assert torch.isinf(metric_fn(input_tensor, target_tensor))

    # Test with known values
    input_tensor = torch.ones(1, 3, 256, 256) * 0.5
    target_tensor = torch.ones(1, 3, 256, 256) * 0.25
    # MSE = (0.5 - 0.25)^2 = 0.0625
    # PSNR = 20 * log10(1.0 / sqrt(0.0625)) = 20 * log10(1.0 / 0.25) = 20 * log10(4) ~= 12.04
    assert torch.allclose(metric_fn(input_tensor, target_tensor), torch.tensor(12.0412), atol=1e-4)

def test_ms_ssim_loss_basic():
    """
    Test basic MS-SSIM loss functionality with valid image sizes.

    Objective: Verify that MS_SSIM_loss computes correct loss for valid image dimensions.
    Test criteria: Perfect match returns loss close to 0.0.
    How testing for this criteria fulfills purpose: Ensures MS-SSIM works for model training.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure mathematical computation without external dependencies.
    """
    # Test basic functionality, not the size constraint here
    loss_fn = MS_SSIM_loss(data_range=1.0, channel=3)
    # A perfect match should result in a loss of 0 (since SSIM would be 1)
    input_tensor = torch.rand(1, 3, 168, 168)
    target_tensor = input_tensor.clone()
    assert torch.allclose(loss_fn(input_tensor, target_tensor), torch.tensor(0.0), atol=1e-5)

@pytest.mark.parametrize("size", [160, 159, 64, 32])
def test_ms_ssim_size_constraint(size):
    """
    Test MS-SSIM size constraint enforcement for invalid image dimensions.

    Objective: Verify that MS_SSIM_loss enforces minimum size requirement of 161px.
    Test criteria: Images smaller than 161px raise AssertionError with specific message.
    How testing for this criteria fulfills purpose: Ensures domain constraints are properly enforced.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure constraint validation without external dependencies.
    """
    loss_fn = MS_SSIM_loss(data_range=1.0, channel=3)
    input_tensor = torch.rand(1, 3, size, size)
    target_tensor = torch.rand(1, 3, size, size)

    with pytest.raises(AssertionError, match="Image size should be larger than 160"):
        loss_fn(input_tensor, target_tensor)

def test_ms_ssim_valid_size():
    """
    Test MS-SSIM with valid minimum image size (161px).

    Objective: Verify that MS_SSIM_loss accepts valid image sizes without errors.
    Test criteria: Images of size 161px and larger do not raise exceptions.
    How testing for this criteria fulfills purpose: Ensures valid use cases work properly.
    What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
    The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure constraint validation without external dependencies.
    """
    loss_fn = MS_SSIM_loss(data_range=1.0, channel=3)
    # Size 161 should work without raising an error
    input_tensor = torch.rand(1, 3, 161, 161)
    target_tensor = torch.rand(1, 3, 161, 161)
    try:
        loss_fn(input_tensor, target_tensor)
    except ValueError:
        pytest.fail("MS_SSIM_loss raised ValueError unexpectedly for a valid size.")

def test_ms_ssim_metric():
    """
    Test ms_ssim_metric for evaluation purposes.

    Objective: Verify MS-SSIM metric computation returns similarity scores.
    Test criteria: Returns values between 0 and 1, higher for similar images.
    How testing fulfills purpose: Ensures metric works for model evaluation.
    Components mocked: None - direct tensor operations.
    Reason for hermeticity: Pure mathematical computation without external deps.
    """
    # Perfect match should be close to 1.0
    input_tensor = torch.rand(1, 3, 168, 168)
    target_tensor = input_tensor.clone()
    score = ms_ssim_metric(input_tensor, target_tensor)
    assert torch.allclose(score, torch.tensor(1.0), atol=1e-3)

    # Different images should have lower score
    different_tensor = torch.rand(1, 3, 168, 168)
    score_diff = ms_ssim_metric(input_tensor, different_tensor)
    assert score_diff < score  # Different images should score lower
    assert 0 <= score_diff <= 1  # Score should be in valid range