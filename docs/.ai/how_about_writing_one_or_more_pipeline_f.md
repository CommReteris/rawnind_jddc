### Why a pipeline factory + registries
A pipeline factory with registries gives you three wins over direct calls into `raw_processing`:
- Extensibility: you can add/replace steps (e.g., a new demosaic or WB) without touching `raw_processing.py`.
- Composability: build different flows for Bayer→RGB, HDR export, self‑supervised pairs, etc., from the same primitives.
- Observability and policy: uniform validation, timing, logging, and caching around every step.

Below is a concrete design that fits your current layout (`dependencies/raw_processing.py`, `pytorch_helpers.py`) and aligns with `DatasetConfig.preprocessing_steps`.

### Mental model
- A `PipelineStep` is a small, pure-ish function that transforms `(context, data)` to `(context, data)`.
- A `Pipeline` is an ordered list of steps, with pre/post hooks for validation and metrics.
- A `Registry` maps step names to step constructors (and can be extended by users/plugins).
- A `PipelineFactory` composes steps from a declarative spec (list of names + params), resolves dependencies, and returns a callable.

### Core interfaces (minimal viable)
```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple

import numpy as np
import torch

# --- Context shared across steps ---
@dataclass
class PipelineContext:
    device: torch.device
    cfg: Any  # DatasetConfig or ProcessingConfig
    meta: Dict[str, Any] = field(default_factory=dict)  # camera EXIF, bayer pattern, etc.
    log: Callable[[str], None] = print  # replace with structured logger

# --- Step protocol ---
class PipelineStep(Protocol):
    def __call__(self, ctx: PipelineContext, x: Any) -> Tuple[PipelineContext, Any]:
        ...

# --- Registry ---
class StepRegistry:
    def __init__(self):
        self._reg: Dict[str, Callable[..., PipelineStep]] = {}
    def register(self, name: str):
        def deco(factory: Callable[..., PipelineStep]):
            self._reg[name] = factory
            return factory
        return deco
    def create(self, name: str, **kwargs) -> PipelineStep:
        if name not in self._reg:
            raise KeyError(f"Unknown step '{name}'")
        return self._reg[name](**kwargs)

STEPS = StepRegistry()

# --- Pipeline ---
@dataclass
class Pipeline:
    steps: List[PipelineStep]
    def __call__(self, ctx: PipelineContext, x: Any) -> Tuple[PipelineContext, Any]:
        for step in self.steps:
            ctx, x = step(ctx, x)
        return ctx, x

# --- Factory ---
@dataclass
class PipelineFactory:
    registry: StepRegistry = STEPS
    def build(self, spec: Iterable[Dict[str, Any]]) -> Pipeline:
        # spec: [{"name": "raw_load", "params": {...}}, ...]
        steps = [self.registry.create(it["name"], **it.get("params", {})) for it in spec]
        return Pipeline(steps)
```

### Adapters over your current `raw_processing`
You already have good primitives in `dependencies/raw_processing.py` (`RawLoader`, `BayerProcessor`, `ColorTransformer`, `hdr_nparray_to_file`). We wrap them into steps and register them.

```python
from rawnind.dependencies import raw_processing as rawproc
from rawnind.dependencies.pytorch_helpers import fpath_to_tensor

@STEPS.register("raw_load")
def mk_raw_load(force_rggb: bool = True, return_float: bool = True) -> PipelineStep:
    loader = rawproc.RawLoader(rawproc.ProcessingConfig())
    def step(ctx: PipelineContext, fpath: str):
        mono, meta = loader.raw_fpath_to_rggb_img_and_metadata(
            fpath, force_rggb=force_rggb, return_float=return_float
        )
        ctx.meta.update(meta)
        return ctx, mono  # mono shape [1, H, W] float
    return step

@STEPS.register("bayer_normalize")
def mk_bayer_normalize(target: str = "RGGB") -> PipelineStep:
    bp = rawproc.BayerProcessor()
    def step(ctx: PipelineContext, mono: np.ndarray):
        mono2, ctx.meta = bp.normalize_to_pattern(mono, ctx.meta, target)
        return ctx, mono2
    return step

@STEPS.register("white_balance")
def mk_white_balance(source: str = "as_shot") -> PipelineStep:
    def step(ctx: PipelineContext, mono: np.ndarray):
        wb = rawproc.estimate_white_balance(ctx.meta, mode=source)
        out = rawproc.apply_wb_bayer(mono, wb, ctx.meta)
        return ctx, out
    return step

@STEPS.register("demosaic")
def mk_demosaic(method: str = "bilinear") -> PipelineStep:
    def step(ctx: PipelineContext, mono: np.ndarray):
        rgb = rawproc.demosaic(mono, ctx.meta, method=method)  # [3,H,W]
        return ctx, rgb
    return step

@STEPS.register("color_transform")
def mk_color_transform(inp: str = "lin_rec2020", out: str = "lin_rec2020") -> PipelineStep:
    ct = rawproc.ColorTransformer(inp, out)
    def step(ctx: PipelineContext, rgb: np.ndarray):
        rgb2 = ct.apply(rgb, ctx.meta)
        return ctx, rgb2
    return step

@STEPS.register("to_tensor")
def mk_to_tensor(batch: bool = False, crop_to_multiple: Optional[int] = None) -> PipelineStep:
    def step(ctx: PipelineContext, x: np.ndarray):
        t = torch.tensor(x, device=ctx.device)
        if crop_to_multiple:
            from rawnind.dependencies.pytorch_operations import crop_to_multiple as crop
            t = crop(t, crop_to_multiple)
        if batch:
            t = t.unsqueeze(0)
        return ctx, t
    return step

@STEPS.register("hdr_write")
def mk_hdr_write(fmt: str = "exr") -> PipelineStep:
    def step(ctx: PipelineContext, rgb: np.ndarray):
        out_fpath = ctx.meta.get("out_fpath")
        if out_fpath:
            rawproc.hdr_nparray_to_file(rgb, out_fpath, fmt=fmt)
        return ctx, rgb
    return step
```

This keeps `raw_processing` as the source of truth while giving you a composable interface.

### Building pipelines from `DatasetConfig`
You already have `DatasetConfig.preprocessing_steps: List[str]`. We can interpret those with parameters.

```python
# Example spec derived from DatasetConfig
spec = [
    {"name": "raw_load", "params": {"force_rggb": True}},
    {"name": "bayer_normalize", "params": {"target": "RGGB"}},
    {"name": "white_balance", "params": {"source": "as_shot"}},
    {"name": "demosaic", "params": {"method": "bilinear"}},
    {"name": "color_transform", "params": {
        "inp": cfg.input_color_profile,
        "out": cfg.output_color_profile,
    }},
    {"name": "to_tensor", "params": {"batch": True, "crop_to_multiple": 2}},
]

pf = PipelineFactory()
pipeline = pf.build(spec)

ctx = PipelineContext(device=torch.device(cfg.device), cfg=cfg, meta={"out_fpath": None})
ctx, rgb_tensor = pipeline(ctx, "C:/path/to/file.ARW")
```

You can predefine named presets:

```python
PIPELINE_PRESETS = {
    "bayer_to_tensor_rgb": [
        {"name": "raw_load"},
        {"name": "bayer_normalize"},
        {"name": "white_balance"},
        {"name": "demosaic", "params": {"method": "bilinear"}},
        {"name": "color_transform"},
        {"name": "to_tensor", "params": {"batch": True, "crop_to_multiple": 2}},
    ],
    "raw_to_exr": [
        {"name": "raw_load"},
        {"name": "bayer_normalize"},
        {"name": "white_balance"},
        {"name": "demosaic"},
        {"name": "color_transform"},
        {"name": "hdr_write", "params": {"fmt": "exr"}},
    ],
}
```

### Advanced options to consider
- Validation hooks: before/after each step, assert shape/dtype contracts (e.g., `mono: [1,H,W] float`, `rgb: [3,H,W] float`, Bayer alignment preserved when requested).
- Timing/metrics: wrap execution with timers and emit per‑step durations. This makes IO/demosaic hotspots visible.
- Caching layer: add a `cache` step using a fingerprint of `(fpath, mtime, ProcessingConfig, step-spec)` to store/retrieve NPZ/EXR from a disk cache.
- Lazy provider loading: steps that touch OpenEXR/TIFF load providers only on demand (fail fast with actionable error messages).
- Plugin system: allow third‑party `STEPS.register` via Python entry points (`pyproject.toml`) under `rawnind.steps`.
- Type guards: use `typing.Protocol` and/or `pydantic` models for step params to surface misconfigurations earlier.

### How this integrates with your current layers
- Dataset layer: `clean_api.py` can pick a preset by `dataset_type` and `data_format`, then call the pipeline inside the dataset’s `__getitem__` before cropping.
- Dependencies layer: remains the single source of truth for image math and IO. The pipeline only wraps those calls.
- Legacy parity: build a “legacy-parity” preset that reproduces the original path and use it in golden tests to compare outputs against `legacy_raw.py` on a few fixtures.

### Example: wiring into `RawImageDataset`
```python
class PipelineBackedDataset(RawImageDataset):
    def __init__(self, cfg: DatasetConfig, pipeline: Pipeline, *args, **kwargs):
        super().__init__(num_crops=cfg.num_crops_per_image, crop_size=cfg.crop_size)
        self.cfg = cfg
        self.pipeline = pipeline
        self.ctx = PipelineContext(device=torch.device(cfg.device), cfg=cfg)
        self.files = [Path(p) for p in cfg.content_fpaths]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fpath = str(self.files[idx])
        ctx, x = self.pipeline(self.ctx, fpath)  # x is [3,H,W] tensor (per preset)
        # Build mask and optional target, then call random_crops from base class
        mask = torch.ones((self.crop_size, self.crop_size), dtype=torch.bool, device=x.device)  # example
        x_crops, mask_crops = self.random_crops(x, None, mask)
        return {"x_crops": x_crops, "mask_crops": mask_crops, "meta": ctx.meta}
```

### Error handling and contracts
- Each step should document input/output contract with simple asserts and human‑readable errors including `dataset_type`, `bayer_pattern`, and `maintain_bayer_alignment` from `DatasetConfig`.
- Include a `ctx.meta.setdefault("history", []).append({"step": name, "shape": tuple(x.shape), "dtype": str(x.dtype)})` to aid debugging.

### Testing strategy
- Unit tests per step using small arrays; verify shape/dtype/metadata changes.
- Pipeline composition tests: given a preset and a tiny sample raw, the result should meet invariants and match reference outputs within a tolerance.
- Property‑based tests: for cropability/shape invariants across randomized inputs and different `bayer_pattern` metadata.

### Migration roadmap
1) Introduce `PipelineContext`, `StepRegistry`, `Pipeline`, `PipelineFactory` in `dependencies` (new file `pipeline.py`).
2) Add step adapters for 4–6 core operations (load, normalize, WB, demosaic, color, to_tensor).
3) Create two presets: `bayer_to_tensor_rgb` and `raw_to_exr`.
4) Wire a `ConfigurableDataset` in `dataset.clean_api` to pick a preset based on `DatasetConfig`.
5) Add tests and a small disk cache step as a follow‑up.

### Alternative: torchdata DataPipes as the pipeline
If you prefer more PyTorch‑native streaming and sharding, the exact same steps can be implemented as `DataPipes` and composed with `torchdata`. The registry/factory ideas still apply; only the step type changes to a DataPipe transform.

### Bottom line
Yes—add a small, typed pipeline factory with registries on top of `raw_processing`. Start with adapters for existing primitives, define a couple of presets, and route `DatasetConfig.preprocessing_steps` through the factory. You’ll gain modularity, testability, and observability with minimal disruption to the current codebase.