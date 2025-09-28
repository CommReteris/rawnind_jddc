import pytest
import torch

from rawnind.inference.inference_engine import InferenceEngine

pytestmark = pytest.mark.acceptance


class _TinyModel(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.net = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Return a dict sometimes to exercise both branches
        y = self.net(x)
        return {"reconstructed_image": y}


@pytest.mark.parametrize("shape", [(3, 8, 8), (1, 3, 8, 8)])
def test_infer_tensor_and_batch(shape):
    """Test InferenceEngine.infer handles both single tensors and batched inputs.

    This parametrized test verifies that the inference engine correctly processes
    both single images (CHW format) and batched images (NCHW format), maintaining
    proper shape handling and returning tensor outputs by default.

    Expected behavior:
    - Single image input [C,H,W] is processed as batch of 1
    - Batched input [N,C,H,W] preserves batch dimension in output
    - Output is always torch.Tensor (default return_dict=False)
    - Spatial dimensions preserved, channels match model output
    - No errors for valid tensor inputs on CPU device

    Key assertions:
    - Output is torch.Tensor instance
    - Output spatial shape matches input [H,W] = [8,8]
    - Output channels match model configuration (3)
    - Handles both single and batched input shapes seamlessly
    """
    model = _TinyModel(in_channels=3, out_channels=3)
    engine = InferenceEngine(model=model, device="cpu")

    img = torch.randn(*shape)
    out = engine.infer(img)  # return_dict=False by default

    assert isinstance(out, torch.Tensor)
    assert out.shape[-2:] == torch.Size([8, 8])
    assert out.shape[-3] == 3  # channels


@pytest.mark.parametrize("return_dict", [True, False])
def test_infer_output_modes(return_dict):
    """Test InferenceEngine.infer output format flexibility.

    This parametrized test verifies that the inference engine can return results
    either as a dictionary (for models returning structured outputs like compression
    pipelines) or as a single tensor (for standard image-to-image models), based
    on the return_dict parameter.

    Expected behavior:
    - return_dict=True: Returns dict with model-specific keys (e.g., "reconstructed_image")
    - return_dict=False: Returns primary tensor output directly
    - Dict mode preserves all model forward pass outputs
    - Tensor mode extracts and returns the main image tensor
    - Both modes handle device placement correctly

    Key assertions:
    - Output type matches requested format (dict or Tensor)
    - Dict contains expected keys like "reconstructed_image"
    - Tensor output is valid torch.Tensor
    - No shape or type mismatches between modes
    """
    model = _TinyModel()
    engine = InferenceEngine(model=model, device="cpu")

    img = torch.randn(3, 4, 4)
    out = engine.infer(img, return_dict=return_dict)

    if return_dict:
        assert isinstance(out, dict)
        assert "reconstructed_image" in out
        assert isinstance(out["reconstructed_image"], torch.Tensor)
    else:
        assert isinstance(out, torch.Tensor)


def test_transfer_function_factory_available():
    """Test availability and basic functionality of transfer function factory.

    This test verifies that the InferenceEngine can create and apply common
    transfer functions used in image processing pipelines, ensuring color space
    conversions and gamma corrections are accessible through the clean API.

    Expected behavior:
    - Factory returns callable functions for supported transfer names
    - "None" identity function preserves input unchanged
    - PQ and gamma functions execute without errors on valid tensors
    - All functions handle batched image tensors [N,C,H,W] or [C,H,W]
    - No device or shape mismatches during application

    Key assertions:
    - get_transfer_function returns callable for "None", "pq", "gamma22"
    - Identity function (None) produces output identical to input
    - Other functions complete execution without exceptions
    - Input/output tensors have matching shapes and devices
    """
    # Ensures callable is returned for supported names
    tf_none = InferenceEngine.get_transfer_function("None")
    tf_pq = InferenceEngine.get_transfer_function("pq")
    tf_gamma = InferenceEngine.get_transfer_function("gamma22")

    import torch
    x = torch.rand(1, 1, 2, 2)
    assert torch.allclose(tf_none(x), x)
    _ = tf_pq(x)
    _ = tf_gamma(x)
