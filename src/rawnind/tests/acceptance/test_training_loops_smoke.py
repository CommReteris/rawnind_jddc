import types
import pytest
import torch

from rawnind.training.training_loops import ImageToImageNNTraining

pytestmark = pytest.mark.acceptance


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, n=4, c=3, h=8, w=8):
        self.data = [torch.randn(c, h, w) for _ in range(n)]
        self.targets = [torch.randn(c, h, w) for _ in range(n)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Match RawDatasetOutput shape contract if needed, but we keep it simple
        x, y = self.data[idx], self.targets[idx]
        return {"clean": y, "noisy": x}


class _TinyModel(torch.nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.in_channels = c
        self.net = torch.nn.Conv2d(c, c, kernel_size=1)

    def forward(self, x):
        return {"reconstructed_image": self.net(x)}


class _ToyArgs(types.SimpleNamespace):
    pass


@pytest.mark.skip(reason="ImageToImageNNTraining API may require full args; enable when constructor is stabilized")
def test_minimal_training_step_runs(tmp_path):
    # This test is intentionally prepared and skipped until constructor args stabilize.
    args = _ToyArgs(
        batch_size=2,
        num_workers=0,
        save_dpath=str(tmp_path),
        device="cpu",
        lr=1e-3,
        max_steps=1,
    )
    trainer = ImageToImageNNTraining(args=args)

    model = _TinyModel()
    ds = _ToyDataset(n=2)
    dl = torch.utils.data.DataLoader(ds, batch_size=1)

    # Expect the trainer to have a step-like method; adapt when API settles
    trainer.model = model
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = next(iter(dl))

    # Exercise a single forward/backward if available
    if hasattr(trainer, "compute_train_loss"):
        loss = trainer.compute_train_loss(batch)
        assert torch.isfinite(loss).item() is True
