import pytest
import torch

from rawnind.dataset.base_dataset import RawImageDataset

pytestmark = pytest.mark.acceptance


class _TinyRawDataset(RawImageDataset):
    def __init__(self, *args, **kwargs):
        # Allow parent to set defaults; override members needed for __getitem__
        super().__init__(*args, **kwargs)
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
    ds = _TinyRawDataset()
    # Random crops (simulate API)
    rc = ds.random_crops(torch.randn(3, 32, 32), crop_size=16, count=2)
    assert isinstance(rc, list) and len(rc) == 2
    for patch in rc:
        assert isinstance(patch, torch.Tensor)
        assert patch.shape == torch.Size([3, 16, 16])

    # Center crop
    cc = ds.center_crop(torch.randn(3, 32, 32), crop_size=16)
    assert cc.shape == torch.Size([3, 16, 16])


@pytest.mark.xfail(reason="Specific dataset classes not yet extracted from rawds.py")
def test_clean_clean_dataset_available():
    from rawnind.dataset.clean_datasets import CleanCleanImageDataset  # noqa: F401
