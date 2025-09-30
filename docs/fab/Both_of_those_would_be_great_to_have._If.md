### Visual map: Legacy → Dependencies → Clean Dataset API

```mermaid
flowchart TD
  subgraph Legacy [Legacy (reference only)]
    L1[legacy_raw.py]\n- raw reading\n- Bayer handling\n- color transforms\n- HDR/EXR export
    L2[legacy_rawds.py]\n- dataset classes\n- cropping/masks\n- alignment & QC\n- splits & loaders
  end

  subgraph Dependencies [Consolidated Domain Logic]
    D1[src\rawnind\dependencies\raw_processing.py]\n- RawLoader / ProcessingConfig\n- BayerProcessor\n- ColorTransformer\n- hdr_nparray_to_file\n- raw_fpath_to_* helpers
    D2[src\rawnind\dependencies\pytorch_helpers.py]\n- get_device\n- fpath_to_tensor\n- sdr_pttensor_to_file
    D3[src\rawnind\dependencies\numpy_operations.py]\n- img_fpath_to_np_flt\n- numpy IO/conversion
    D4[src\rawnind\dependencies\json_saver.py]\n- load_yaml / save_yaml
  end

  subgraph Dataset [Clean, orchestration layer]
    C1[src\rawnind\dataset\base_dataset.py]\n- RawImageDataset (cropping/masks)\n- constants import
    C2[src\rawnind\dataset\clean_api.py]\n- create_*_dataset\n- splits & validation\n- batch format
    C3[src\rawnind\dataset\__init__.py]\n- re-export clean API\n- constants
  end

  L1 --> D1
  L1 --> D2
  L2 --> C1
  D1 --> C1
  D1 --> C2
  D2 --> C1
  D2 --> C2
  D3 --> D2
  D4 --> C2
```

### Point-by-point mapping (legacy → dependencies → dataset)

- Raw file loading and metadata
  - Legacy: `legacy_raw.py` functions like `raw_fpath_to_mono_img_and_metadata`, provider detection, metadata normalization
  - Dependencies: `dependencies/raw_processing.py` (`RawLoader`, `ProcessingConfig`, `raw_fpath_to_rggb_img_and_metadata`, provider logic for OpenEXR/TIFF)
  - Dataset: Imported and used in `dataset/base_dataset.py` and `dataset/clean_api.py` via `from ..dependencies import raw_processing as rawproc`

- Bayer pattern handling and normalization
  - Legacy: `legacy_raw.py` inline conversions to RGGB (e.g., `mono_any_to_mono_rggb`, `set_bayer_pattern_name`), assumes cropping borders, pattern-specific shifts
  - Dependencies: `raw_processing.BayerProcessor` (pattern normalization, RGGB conversions, demosaicing hooks)
  - Dataset: Bayer datasets in `dataset/bayer_datasets.py` and general loaders in `clean_api.py` call into `rawproc` for mosaic logic

- Color transforms and white balance
  - Legacy: `legacy_raw.py` used `common.libs.icc` for profile transforms; applied channel-wise WB for mosaics
  - Dependencies: `raw_processing.ColorTransformer` and `from . import color_management as icc` (alias mirrors legacy naming), WB application in `raw_processing`
  - Dataset: passes `color_profile`/`output_color_profile` and flags via `DatasetConfig`, with `clean_api` delegating transforms to `raw_processing`

- HDR/EXR export
  - Legacy: `legacy_raw.py` `hdr_*` functions and OpenEXR/oiio branches
  - Dependencies: `raw_processing.hdr_nparray_to_file`, `raw_fpath_to_hdr_img_file`, OpenEXR provider detection at top of module
  - Dataset: When exporting or validating HDR, `clean_api` uses `rawproc` helpers; tests in acceptance target OpenEXR bit depth behavior under `dependencies/tests`

- Robust image IO to tensor
  - Legacy: scattered file → numpy → tensor conversions in `legacy_rawds.py`
  - Dependencies: `pytorch_helpers.fpath_to_tensor` (retry logic, `crop_to_multiple`), backed by `numpy_operations.img_fpath_to_np_flt`
  - Dataset: `base_dataset.py` and `clean_api.py` import `pytorch_helpers as pt_helpers_dep` for device/tensorization

- Cropping and mask policy
  - Legacy: `legacy_rawds.py` `RawImageDataset.random_crops`/`make_a_random_crop` and constants `MAX_MASKED`, `MAX_RANDOM_CROP_ATTEMPS`, `MASK_MEAN_MIN`, `ALIGNMENT_MAX_LOSS`, `OVEREXPOSURE_LB`
  - Dependencies: N/A (policy is dataset-level)
  - Dataset: `dataset/base_dataset.py` reimplements these methods and uses/imports the constants; `dataset/__init__.py` re-exports constants for API consumers

- Dataset construction, splits, and batch format
  - Legacy: multiple dataset classes and ad-hoc split code in `legacy_rawds.py`
  - Dependencies: N/A
  - Dataset: `dataset/clean_api.py` provides `create_training_dataset`, `create_validation_dataset`, `create_test_dataset`, `prepare_dataset_splits`, standardized batch outputs, optional stats collection

- Configuration and validation
  - Legacy: implicit args and global constants
  - Dependencies: `json_saver.load_yaml` for config IO
  - Dataset: `dataset/dataset_config.py` `DatasetConfig` dataclass with `__post_init__` validation and defaults; `clean_api.validate_training_type_and_dataset_config`

### File inventory with quick pointers

- src\rawnind\dataset
  - `clean_api.py`: imports `pt_helpers_dep`, `rawproc`, `arbitrary_proc`, `load_yaml`; defines clean dataset builders, split utilities, and formatting of outputs
  - `base_dataset.py`: defines `RawImageDataset` cropping/masking logic; imports `pytorch_helpers` and `raw_processing`
  - `bayer_datasets.py`: specialized paired datasets using `raw_processing` and `pytorch_helpers`
  - `dataset_config.py`: `DatasetConfig` with validation and sensible defaults; Bayer/RGB sub-configs
  - `__init__.py`: re-exports clean API and constants matching legacy semantics

- src\rawnind\dependencies
  - `raw_processing.py` (~799 lines): raw providers, Bayer handling, color transforms, EXR/HDR IO, metadata normalization; comments align with legacy behavior
  - `pytorch_helpers.py` (~170 lines): device management, `fpath_to_tensor`, SDR save, schedulers
  - `numpy_operations.py`: numpy image IO and conversions used by `pytorch_helpers`
  - `json_saver.py`: YAML/JSON helpers used by dataset layer

- Legacy (reference only, not packaged into `rawnind.*`)
  - `legacy_raw.py`: pre-refactor raw+Bayer+color+HDR logic
  - `legacy_rawds.py`: pre-refactor dataset classes, cropping, masks, alignment policies

### Compliance checklist for your FRS (copy-paste ready)

- Module boundaries
  - [ ] Dataset layer must not implement raw/Bayer/color/HDR logic; it must import from `rawnind.dependencies.raw_processing` and `rawnind.dependencies.pytorch_helpers`
  - [ ] Cropping/masking/thresholds live in `rawnind.dataset.base_dataset.RawImageDataset` and constants re-exported by `rawnind.dataset`
  - [ ] All config loading uses `rawnind.dependencies.json_saver.load_yaml`

- Dataset construction and validation
  - [ ] Use `rawnind.dataset.create_training_dataset`, `create_validation_dataset`, `create_test_dataset` for public entry points
  - [ ] Validate inputs via `rawnind.dataset.DatasetConfig.__post_init__` (enforces channel counts, crop size parity, etc.)
  - [ ] Enforce split policies via `prepare_dataset_splits` and honor `test_reserve_images`/`test_reserve_config_path`

- Raw/Bayer pipeline
  - [ ] Read raw images via `raw_processing.RawLoader` and helpers like `raw_fpath_to_rggb_img_and_metadata`
  - [ ] Normalize Bayer patterns to RGGB using `raw_processing.BayerProcessor` where required; maintain `maintain_bayer_alignment` when specified in `DatasetConfig`
  - [ ] Apply color management via `raw_processing.ColorTransformer` using `input_color_profile`/`output_color_profile`

- IO and tensorization
  - [ ] Convert file paths to tensors via `pytorch_helpers.fpath_to_tensor` (use `device` from `DatasetConfig` and optional `crop_to_multiple`)
  - [ ] Save SDR tensors via `pytorch_helpers.sdr_pttensor_to_file` where needed in tests/utilities

- HDR/EXR
  - [ ] Export HDR using `raw_processing.hdr_nparray_to_file` and `file_format == "exr"`/`dynamic_range == "hdr"` config combinations
  - [ ] Respect provider detection branches in `raw_processing` for OpenEXR/TIFF handling

- Cropping/mask policies
  - [ ] Use `RawImageDataset.random_crops` and `make_a_random_crop`; enforce `MAX_MASKED`, `MAX_RANDOM_CROP_ATTEMPS`, `MASK_MEAN_MIN`
  - [ ] Keep alignment and overexposure checks consistent with `ALIGNMENT_MAX_LOSS` and `OVEREXPOSURE_LB`

- Public API surfacing
  - [ ] Re-export clean API and constants via `rawnind.dataset.__init__` to keep user imports stable
  - [ ] Avoid importing `legacy_*` modules anywhere in packaged code (acceptance tests enforce this)

### Example: tracing a legacy routine to its refactored counterpart

- Legacy `legacy_raw.py` RGGB normalization (e.g., `mono_any_to_mono_rggb` around raw border handling)
  - Now: `raw_processing.BayerProcessor` handles pattern-to-RGGB with metadata-aware cropping; `raw_processing.RawLoader` normalizes sizes and `RGBG_pattern`
  - Dataset usage: `bayer_datasets.py` constructs paired inputs by calling `rawproc` functions to get aligned Bayer and RGB, then hands off to `RawImageDataset` for cropping

- Legacy HDR writer in `legacy_raw.py` (OpenEXR branch using `Imath`)
  - Now: `raw_processing.hdr_nparray_to_file` with provider detection at module import; acceptance tests in `dependencies/tests/test_openEXR_bit_depth.py` cover bit depth

- Legacy dataset cropping in `legacy_rawds.py` (`RawImageDataset.random_crops`/`make_a_random_crop`)
  - Now: `dataset/base_dataset.py` with the same method names and constants; `dataset/__init__.py` re-exports those constants

### Minimal usage examples (sanity-check the paths)

```python
from rawnind.dataset import create_training_dataset, DatasetConfig

cfg = DatasetConfig(
    dataset_type="bayer_pairs",
    data_format="bayer_rgb_pairs",
    input_channels=4,
    output_channels=3,
    crop_size=128,
    num_crops_per_image=8,
    batch_size=4,
    content_fpaths=["path/to/sample1.ARW", "path/to/sample2.ARW"],
    device="cuda:0",
    bayer_pattern="RGGB",
)
train_ds = create_training_dataset(cfg)
```

```python
# File path → tensor using consolidated IO
from rawnind.dependencies.pytorch_helpers import fpath_to_tensor
img = fpath_to_tensor("path/to/image.exr", device="cpu", batch=True)
```

### How to verify quickly in this repo

- Imports in dataset layer
  - Open `src\rawnind\dataset\clean_api.py` and confirm imports: `from ..dependencies import pytorch_helpers as pt_helpers_dep`, `from ..dependencies import raw_processing as rawproc`, `from ..dependencies.json_saver import load_yaml`
  - Open `src\rawnind\dataset\base_dataset.py` and confirm: `from ..dependencies import pytorch_helpers as pt_helpers` and `from ..dependencies.raw_processing import *`

- Constants parity
  - Open `src\rawnind\dataset\__init__.py` and confirm constants mirror legacy values: `MAX_MASKED = 0.5`, `MAX_RANDOM_CROP_ATTEMPS = 10`, `MASK_MEAN_MIN = 0.8`, `ALIGNMENT_MAX_LOSS = 0.035`, `OVEREXPOSURE_LB = 0.99`

- Domain logic location
  - Open `src\rawnind\dependencies\raw_processing.py` and look for: `ColorTransformer`, `hdr_nparray_to_file`, `RawLoader`, and comments referencing legacy behavior

### Summary

- The domain logic was not dropped; it was consolidated under `src\rawnind\dependencies` and is now consumed by a slimmer, cleaner dataset API in `src\rawnind\dataset`. The visual map and the checklist above should make the architecture and the responsibilities clear and directly actionable for your FRS/gap analysis.

If you want, I can expand the mapping into a line-referenced concordance (legacy function → new symbol with file:line ranges) or generate separate Mermaid views per concern (IO, color, Bayer, HDR, dataset orchestration).