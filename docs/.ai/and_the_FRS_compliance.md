### What “FRS compliance” means here
In the context of your refactor, the Functional Requirements Specification (FRS) you’ve been driving toward boils down to:
- Clear separation of concerns: dataset orchestration vs. domain logic (raw/Bayer/color/HDR/IO)
- Clean public dataset API with standard configs, splits, and batch contracts
- Preservation of legacy behavior where intended, with constants/thresholds parity
- Robust IO and color/raw processing through the consolidated dependencies layer
- Guardrails that prevent reintroducing legacy CLI and ad‑hoc code paths

Below is a concise compliance matrix with status, concrete pointers, and remaining gaps.

### FRS compliance matrix (status snapshot)
- Module boundaries (dataset vs dependencies)
  - Status: Compliant
  - Evidence:
    - `src\rawnind\dataset\clean_api.py` imports `..dependencies.pytorch_helpers as pt_helpers_dep`, `..dependencies.raw_processing as rawproc`, `..dependencies.arbitrary_processing as arbitrary_proc`, and `..dependencies.json_saver.load_yaml` (lines 24–27 in your snapshot).
    - `src\rawnind\dataset\base_dataset.py` imports `..dependencies.pytorch_helpers` and `..dependencies.raw_processing` (lines 21–22).
    - `src\rawnind\dependencies\raw_processing.py` encapsulates raw/Bayer/color/HDR IO (file shows `RawLoader`, `BayerProcessor`, `ColorTransformer`, `hdr_nparray_to_file`, etc.).

- Clean dataset API: standardized entry points, configs, splits, batch format
  - Status: Largely compliant
  - Evidence:
    - Public re-exports in `src\rawnind\dataset\__init__.py` list `create_training_dataset`, `create_validation_dataset`, `create_test_dataset`, `DatasetConfig`, `prepare_dataset_splits`, etc.
    - `DatasetConfig` defined in `src\rawnind\dataset\dataset_config.py` with validation in `__post_init__` (even crop size, channel constraints, defaults for quality thresholds).
  - Notes:
    - Ensure `validate_dataset_format`, `prepare_dataset_splits`, and `DatasetMetadata` are implemented and covered by tests (they are re-exported; verify their concrete definitions live in `clean_api.py`).

- Legacy behavior parity (constants and policies)
  - Status: Compliant
  - Evidence:
    - `src\rawnind\dataset\__init__.py` re-exports constants mirroring legacy: `MAX_MASKED=0.5`, `MAX_RANDOM_CROP_ATTEMPS=10`, `MASK_MEAN_MIN=0.8`, `ALIGNMENT_MAX_LOSS=0.035`, `OVEREXPOSURE_LB=0.99`, `TOY_DATASET_LEN=25`.
    - `legacy_rawds.py` shows the same values; `base_dataset.py` reimplements `RawImageDataset` behaviors (crop attempts, mask thresholds) while importing `raw_processing` for shape checks/alignments.

- Raw/Bayer pipeline moved to dependencies
  - Status: Compliant
  - Evidence:
    - `dependencies/raw_processing.py` contains provider detection (OpenEXR/TIFF), `ProcessingConfig`/`RawLoader`, Bayer normalization and pattern handling, color transforms via `from . import color_management as icc`, and HDR writers such as `hdr_nparray_to_file`.
    - Dataset layer files import and use `rawproc` rather than rolling their own raw/Bayer/color logic.

- IO and tensorization consolidated
  - Status: Compliant
  - Evidence:
    - `dependencies/pytorch_helpers.py` exposes `fpath_to_tensor` (with retry, optional `crop_to_multiple`) backed by `dependencies/numpy_operations.img_fpath_to_np_flt`.
    - Dataset code imports `pytorch_helpers` from dependencies.

- HDR/EXR export handling
  - Status: Compliant
  - Evidence:
    - `dependencies/raw_processing.py` imports `Imath` and provides HDR/EXR writer (`hdr_nparray_to_file`, `raw_fpath_to_hdr_img_file`).
  - Note:
    - Keep acceptance/integration tests around bit‑depth/provider behavior (you previously had such tests; ensure they’re still green).

- Public API surfacing and legacy guardrails
  - Status: Likely compliant
  - Evidence:
    - Clean API re-exports are present in `dataset/__init__.py`.
    - You previously referenced acceptance tests like `tests/acceptance/test_legacy_cli_removed.py`. Confirm they still pass with the current package layout.

- Configuration and validation
  - Status: Compliant
  - Evidence:
    - `DatasetConfig.__post_init__` enforces even `crop_size`, positive counts, and channel constraints based on `dataset_type` (Bayer=4 channels, RGB=3 channels) and merges default quality thresholds.
    - YAML load path exists (`dependencies.json_saver.load_yaml`) and is imported by `clean_api.py`.

### Focused checklist (actionable)
Use this as a sign‑off sheet for FRS readiness:
- [x] Dataset layer only orchestrates; no raw/Bayer/color/HDR logic implemented there
- [x] Dependencies contain raw/Bayer/color/HDR/IO logic (`raw_processing`, `pytorch_helpers`, `numpy_operations`)
- [x] Public API is exposed via `rawnind.dataset.__init__` with the intended surface
- [x] Legacy constants and cropping/mask behavior preserved in dataset layer
- [x] `DatasetConfig` validates critical parameters and defaults; supports both Bayer and RGB
- [x] File‑to‑tensor IO uses `pytorch_helpers.fpath_to_tensor`
- [x] HDR/EXR writing paths live in `raw_processing`
- [x] Split utilities and dataset builders present in `clean_api.py`
- [ ] Verify acceptance tests preventing legacy CLI exposure are present and passing
- [ ] Verify end‑to‑end integration tests for dataset clean API are present and passing

### Gaps and caveats spotted
- Test coverage pointers
  - Your recent VCS diff indicates edits in `dataset/clean_api.py`, `dataset/dataset_config.py`, and `dataset/tests/test_integration.py`. Ensure the updated tests still cover:
    - Batch output contract (shapes, presence of `mask_crops`, `rgb_xyz_matrix` when expected)
    - Split behavior with `test_reserve_images`/`test_reserve_config_path`
    - Bayer alignment when `maintain_bayer_alignment=True`
- API existence vs. export
  - `dataset/__init__.py` re‑exports several names (e.g., `DatasetMetadata`, `validate_dataset_format`). Confirm all of these are defined in `clean_api.py` and have tests.

### Quick verification guide (copy‑paste for reviewers)
- Confirm imports and boundaries
  - `open src\rawnind\dataset\clean_api.py` and look for the `..dependencies.*` imports (present lines 24–27).
  - `open src\rawnind\dataset\base_dataset.py` and confirm imports of `pytorch_helpers` and `raw_processing` (present lines 21–22).
- Confirm constants parity
  - `open src\rawnind\dataset\__init__.py` and verify constants match legacy values (present lines 68–75 in your snapshot).
- Confirm `DatasetConfig` validations and defaults
  - `open src\rawnind\dataset\dataset_config.py` and verify `__post_init__` constraints and `quality_thresholds` merging (lines 106–134 in your snapshot).
- Confirm dependencies host domain logic
  - `open src\rawnind\dependencies\raw_processing.py` and verify presence of `ColorTransformer`, `RawLoader`, `BayerProcessor`, `hdr_nparray_to_file` (visible in file header and body).
- Confirm IO path
  - `open src\rawnind\dependencies\pytorch_helpers.py` and check `fpath_to_tensor` uses `numpy_operations.img_fpath_to_np_flt` and optional `crop_to_multiple`.

### Minimal compliance examples
- Using the clean dataset API
```python
from rawnind.dataset import create_training_dataset, DatasetConfig

cfg = DatasetConfig(
    dataset_type="bayer_pairs",      # or "rgb_pairs"
    data_format="bayer_rgb_pairs",   # or "clean_noisy" / "clean_clean"
    input_channels=4,
    output_channels=3,
    crop_size=128,
    num_crops_per_image=8,
    batch_size=4,
    content_fpaths=[".../sample1.ARW", ".../sample2.ARW"],
    device="cuda:0",
    bayer_pattern="RGGB",
)
train_ds = create_training_dataset(cfg)
```
- Path‑to‑tensor via consolidated IO
```python
from rawnind.dependencies.pytorch_helpers import fpath_to_tensor
img = fpath_to_tensor(".../image.exr", device="cpu", batch=True)
```

### Conclusion
- Overall FRS compliance: Strong. The refactor meets the architectural boundaries, preserves legacy‑critical behavior where intended (constants, crop/mask policies), and consolidates domain logic in `dependencies`. The remaining due diligence is to confirm the presence and passing status of the acceptance/integration tests that enforce the clean API surface and legacy‑free packaging.

If you’d like, I can turn this into a formal sign‑off document (with test IDs and pass/fail) or produce a per‑requirement traceability list mapping each FRS clause to file:symbols and specific test cases.