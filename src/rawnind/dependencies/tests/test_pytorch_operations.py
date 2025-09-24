import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.rawnind.dependencies.pytorch_operations import (
    pt_crop_batch, img_to_batch, batch_to_img, pixel_unshuffle, oneloss,
    fragile_checksum, RoundNoGradient, crop_to_multiple, gamma_pt
)

@pytest.fixture
def dummy_image_tensor():
    """Returns a dummy image tensor for testing."""
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

@pytest.fixture
def dummy_batch_tensor():
    """Returns a dummy batch tensor for testing."""
    # Example: 16 patches, 3 channels, 64x64 resolution
    return torch.rand(16, 3, 64, 64, dtype=torch.float32)

class TestPyTorchOperations:

    def test_pt_crop_batch(self):
        """
        Test batch cropping functionality for center cropping.

        Objective: Verify that pt_crop_batch correctly crops batches to specified size from center.
        Test criteria: Output shape matches expected cropped dimensions, center cropping is applied.
        How testing for this criteria fulfills purpose: Ensures batch processing works for training data preparation.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor manipulation without external dependencies.
        """
        batch = torch.rand(4, 3, 128, 128)
        cropped_size = 64
        cropped_batch = pt_crop_batch(batch, cropped_size)
        assert cropped_batch.shape == (4, 3, cropped_size, cropped_size)
        # Check if the center part is indeed cropped
        expected_crop = batch[:, :, 32:96, 32:96]
        assert torch.equal(cropped_batch, expected_crop)

    def test_img_to_batch_to_img_round_trip(self):
        """
        Test round-trip conversion between image and batch formats.

        Objective: Verify that img_to_batch and batch_to_img are inverse operations.
        Test criteria: Original image tensor equals reconstructed image tensor after conversion cycle.
        How testing for this criteria fulfills purpose: Ensures data integrity during patch-based processing.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor reshaping without external dependencies.
        """
        imgtensor = torch.rand(1, 3, 768, 512, dtype=torch.float32)
        patch_size = 64
        
        btensor = img_to_batch(imgtensor, patch_size)
        
        assert list(btensor.shape) == [96, 3, 64, 64] # (768/64) * (512/64) = 12 * 8 = 96
        
        imgtensor_back = batch_to_img(btensor, 768, 512)
        
        assert (imgtensor != imgtensor_back).sum() == 0

    def test_img_to_batch_truncation(self):
        """
        Test img_to_batch handling of non-divisible dimensions.

        Objective: Verify that img_to_batch handles dimensions not divisible by patch_size correctly.
        Test criteria: Output shape accounts for truncation to nearest divisible size.
        How testing for this criteria fulfills purpose: Ensures robust patch extraction for variable input sizes.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor slicing without external dependencies.
        """
        imgtensor = torch.rand(1, 3, 770, 514, dtype=torch.float32) # Non-divisible dimensions
        patch_size = 64
        btensor = img_to_batch(imgtensor, patch_size)
        # Expected: (768/64) * (512/64) = 96 patches
        assert list(btensor.shape) == [96, 3, 64, 64]

    def test_pixel_unshuffle_functional(self):
        """
        Test pixel_unshuffle functionality for different downscale factors.

        Objective: Verify pixel_unshuffle correctly rearranges pixels for downscaling.
        Test criteria: Output shape matches expected dimensions for given downscale factor.
        How testing for this criteria fulfills purpose: Ensures proper pixel rearrangement for model architectures.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor reshaping without external dependencies.
        """
        input_tensor = torch.randn(1, 3, 8, 8, dtype=torch.float32)
        downscale_factor = 2
        output_tensor = pixel_unshuffle(input_tensor, downscale_factor)
        assert output_tensor.shape == (1, 12, 4, 4)

        downscale_factor = 4
        input_tensor_4x = torch.randn(1, 3, 16, 16, dtype=torch.float32)
        output_tensor_4x = pixel_unshuffle(input_tensor_4x, downscale_factor)
        assert output_tensor_4x.shape == (1, 48, 4, 4) # 3 * 4^2 = 3 * 16 = 48

    def test_pixel_unshuffle_inverse_pixel_shuffle(self):
        """
        Test pixel_unshuffle inverse relationship with PixelShuffle.

        Objective: Verify pixel_unshuffle is the inverse of PyTorch PixelShuffle.
        Test criteria: pixel_shuffle(pixel_unshuffle(x)) ≈ x for round-trip transformation.
        How testing for this criteria fulfills purpose: Ensures pixel rearrangement operations are reversible.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations with PyTorch PixelShuffle.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor reshaping using standard PyTorch operations.
        """
        input_tensor = torch.randn(1, 12, 4, 4, dtype=torch.float32) # Example output of pixel_unshuffle
        downscale_factor = 2
        # To test inverse, we'd need pixel_shuffle, which is nn.PixelShuffle
        pixel_shuffle_layer = torch.nn.PixelShuffle(downscale_factor)
        
        # Applying pixel_shuffle directly because pixel_unshuffle is the inverse
        # So (pixel_shuffle(pixel_unshuffle(x,2),2)) should get back to x
        # Create an input that can be unshuffled and then shuffled back
        original_img = torch.randn(1, 3, 8, 8, dtype=torch.float32)
        unshuffled_img = pixel_unshuffle(original_img, downscale_factor)
        re_shuffled_img = pixel_shuffle_layer(unshuffled_img)
        assert original_img.shape == re_shuffled_img.shape
        # Note: floating point differences might occur, so compare approximately
        assert torch.allclose(original_img, re_shuffled_img, atol=1e-6)


    def test_oneloss(self, dummy_image_tensor):
        """
        Test oneloss function returns constant loss value.

        Objective: Verify oneloss always returns 1.0 regardless of input.
        Test criteria: Loss value equals 1.0, device matches input tensor device.
        How testing for this criteria fulfills purpose: Ensures baseline loss function behavior.
        What components are mocked, monkeypatched, or are fixtures: dummy_image_tensor fixture.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses fixture for consistent test input, no external dependencies.
        """
        loss = oneloss(dummy_image_tensor, None)
        assert loss.item() == 1.0
        assert loss.device == dummy_image_tensor.device

    def test_fragile_checksum(self, dummy_image_tensor):
        """
        Test fragile_checksum generates consistent and changing values.

        Objective: Verify fragile_checksum produces tuple of floats and changes with input.
        Test criteria: Returns 3-element tuple of floats, different checksums for different tensors.
        How testing for this criteria fulfills purpose: Validates tensor fingerprinting for debugging.
        What components are mocked, monkeypatched, or are fixtures: dummy_image_tensor fixture.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Uses fixture for consistent test input, no external dependencies.
        """
        checksum = fragile_checksum(dummy_image_tensor)
        assert isinstance(checksum, tuple)
        assert len(checksum) == 3
        assert all(isinstance(val, float) for val in checksum)
        # Ensure it changes if tensor changes
        modified_tensor = dummy_image_tensor.clone()
        modified_tensor[0, 0, 0, 0] += 0.1
        modified_checksum = fragile_checksum(modified_tensor)
        assert checksum != modified_checksum

    def test_round_no_gradient_forward(self):
        """
        Test RoundNoGradient forward pass rounds values correctly.

        Objective: Verify RoundNoGradient.apply rounds tensor values to nearest integers.
        Test criteria: Output matches expected rounded values [0., 0., 1., 1., 2.].
        How testing for this criteria fulfills purpose: Ensures custom autograd function works for quantization.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor operations with custom autograd function.
        """
        input_tensor = torch.tensor([0.1, 0.5, 0.9, 1.4, 1.6], requires_grad=True)
        rounded_tensor = RoundNoGradient.apply(input_tensor)
        assert torch.equal(rounded_tensor, torch.tensor([0., 0., 1., 1., 2.]))

    def test_round_no_gradient_backward(self):
        """
        Test RoundNoGradient backward pass preserves gradients.

        Objective: Verify RoundNoGradient backward pass passes gradients unchanged.
        Test criteria: Input gradients equal [1., 1., 1.] after backward pass.
        How testing for this criteria fulfills purpose: Ensures custom autograd function doesn't break gradient flow.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations with autograd.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure autograd operations with custom function.
        """
        input_tensor = torch.tensor([0.1, 0.5, 0.9], requires_grad=True)
        rounded_tensor = RoundNoGradient.apply(input_tensor)
        # Simulate a downstream loss
        loss = rounded_tensor.sum()
        loss.backward()
        # Gradients should pass through unchanged
        assert torch.equal(input_tensor.grad, torch.tensor([1., 1., 1.]))

    def test_crop_to_multiple(self):
        """
        Test crop_to_multiple crops tensors to multiples of given size.

        Objective: Verify crop_to_multiple crops dimensions to nearest multiple of specified value.
        Test criteria: Output shape is (1, 3, 512, 256) from input (1, 3, 513, 257) with multiple=64.
        How testing for this criteria fulfills purpose: Ensures tensors fit model requirements.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor slicing without external dependencies.
        """
        tensor = torch.rand(1, 3, 513, 257)
        multiple = 64
        cropped_tensor = crop_to_multiple(tensor, multiple)
        assert cropped_tensor.shape == (1, 3, 512, 256)
        assert cropped_tensor.size(-2) % multiple == 0
        assert cropped_tensor.size(-1) % multiple == 0

    def test_gamma_pt_positive_values(self):
        """
        Test gamma_pt applies gamma correction to positive values.

        Objective: Verify gamma_pt applies 1/gamma transformation to positive values.
        Test criteria: Output matches img**(1/gamma_val) for positive inputs.
        How testing for this criteria fulfills purpose: Ensures proper gamma correction for image processing.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor operations without external dependencies.
        """
        img = torch.tensor([0.1, 0.5, 0.8, 1.0], dtype=torch.float32)
        gamma_val = 2.2
        expected_output = img**(1/gamma_val)
        output = gamma_pt(img, gamma_val)
        assert torch.allclose(output, expected_output)

    def test_gamma_pt_non_positive_values(self):
        """
        Test gamma_pt preserves non-positive values unchanged.

        Objective: Verify gamma_pt leaves non-positive values unchanged while correcting positive ones.
        Test criteria: Non-positive values remain unchanged, positive values are gamma-corrected.
        How testing for this criteria fulfills purpose: Ensures gamma correction doesn't affect invalid values.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor operations without external dependencies.
        """
        img = torch.tensor([-0.1, 0.0, 0.5, 1.0], dtype=torch.float32)
        gamma_val = 2.2
        output = gamma_pt(img, gamma_val)
        # Non-positive values should remain unchanged, positive values are gamma-corrected
        assert torch.allclose(output[0:2], torch.tensor([-0.1, 0.0]))
        assert torch.allclose(output[2:], torch.tensor([0.5, 1.0])**(1/gamma_val))

    def test_gamma_pt_in_place(self):
        """
        Test gamma_pt in-place modification of tensors.

        Objective: Verify gamma_pt modifies tensors in-place when in_place=True.
        Test criteria: Input tensor is modified to match expected gamma-corrected values.
        How testing for this criteria fulfills purpose: Ensures in-place operations work for memory efficiency.
        What components are mocked, monkeypatched, or are fixtures: None - direct tensor operations.
        The reasons for which the test will be able to fulfill its objective without the real components being mocked/patched/fixtured: Pure tensor operations without external dependencies.
        """
        img = torch.tensor([0.1, 0.5, 0.8], dtype=torch.float32)
        original_img_clone = img.clone()
        gamma_pt(img, in_place=True)
        expected_output = original_img_clone**(1/2.2)
        assert torch.allclose(img, expected_output)
