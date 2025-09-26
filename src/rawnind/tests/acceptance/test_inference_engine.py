<<<<<<< HEAD
import torch
import pytest
=======
import pytest
import torch
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

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
    model = _TinyModel(in_channels=3, out_channels=3)
    engine = InferenceEngine(model=model, device="cpu")

    img = torch.randn(*shape)
    out = engine.infer(img)  # return_dict=False by default

    assert isinstance(out, torch.Tensor)
    assert out.shape[-2:] == torch.Size([8, 8])
    assert out.shape[-3] == 3  # channels


@pytest.mark.parametrize("return_dict", [True, False])
def test_infer_output_modes(return_dict):
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


<<<<<<< HEAD
@pytest.mark.xfail(reason="dependencies.raw_processing not yet migrated; enable once available")
=======
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
def test_transfer_function_factory_available():
    # Ensures callable is returned for supported names
    tf_none = InferenceEngine.get_transfer_function("None")
    tf_pq = InferenceEngine.get_transfer_function("pq")
    tf_gamma = InferenceEngine.get_transfer_function("gamma22")

    import torch
    x = torch.rand(1, 1, 2, 2)
    assert torch.allclose(tf_none(x), x)
    _ = tf_pq(x)
    _ = tf_gamma(x)
