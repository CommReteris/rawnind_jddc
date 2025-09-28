import pytest
import torch

from rawnind.dataset.clean_api import ConfigurableDataset as RawImageDataset

pytestmark = pytest.mark.acceptance


class _TinyRawDataset(RawImageDataset):
    def __init__(self, *args, **kwargs):
        # Initialize base dataset with required args
        super().__init__(num_crops=2, crop_size=16)
        # Create tiny tensors to represent clean/noisy patches
        self._pairs = [
            (torch.randn(3, 16, 16), torch.randn(3, 16, 16))
            for _ in range(4)
        ]

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, idx):
        clean, noisy = self._pairs[idx]
        # Respect the dataset output contract keys if the base class expects them
        return {"clean": clean, "noisy": noisy}


def test_random_crops_and_center_crop_shapes():
    """Test that random crops and center crop produce tensors of expected shapes.

    This test verifies the core cropping functionality in the dataset base contracts.
    It ensures that both random multi-crop extraction and single center cropping
    maintain proper tensor shapes for downstream training operations.

    Expected behavior:
    - random_crops returns multiple crops with batch dimension [num_crops, C, H, W]
    - center_crop returns single crop [C, H, W]
    - Both preserve channel and spatial dimensions
    - Masks are generated with matching shapes

    Key assertions:
    - Crop tensors are torch.Tensors
    - Random crops shape: [num_crops, 3, crop_size, crop_size]
    - Center crop shape: [3, crop_size, crop_size]
    - Mask shapes match image crop shapes
    """
    ds = _TinyRawDataset()

    ximg = torch.randn(3, 32, 32)
    mask = torch.ones_like(ximg, dtype=torch.bool)

    # Random crops (API: ximg, yimg=None, whole_img_mask)
    x_crops, mask_crops = ds.random_crops(ximg, yimg=None, whole_img_mask=mask)
    assert isinstance(x_crops, torch.Tensor)
    assert isinstance(mask_crops, torch.Tensor)
    assert x_crops.shape == torch.Size([ds.num_crops, 3, ds.crop_size, ds.crop_size])
    assert mask_crops.shape == torch.Size([ds.num_crops, 3, ds.crop_size, ds.crop_size])

    # Center crop (API: ximg, yimg=None, mask)
    x_center, mask_center = ds.center_crop(ximg, yimg=None, mask=mask)
    assert x_center.shape == torch.Size([3, ds.crop_size, ds.crop_size])
    assert mask_center.shape == torch.Size([3, ds.crop_size, ds.crop_size])


def test_clean_clean_dataset_available():
    """Test that clean-clean dataset implementation is available in the package.

    This acceptance test verifies that the clean dataset module can be imported
    without errors, ensuring that the core dataset functionality is properly
    exposed in the refactored package structure. It serves as a smoke test
    for the dataset API availability post-refactoring.

    Expected behavior:
    - Import succeeds without ModuleNotFoundError or ImportError
    - CleanCleanImageDataset class is accessible
    - No legacy CLI dependencies are triggered during import

    Key assertions:
    - Successful import of CleanCleanImageDataset
    - No exceptions raised during module access
    """
    from rawnind.dataset.clean_datasets import CleanCleanImageDataset  # noqa: F401
