### Test Suite overview - September 23, 2025

- All tests in `src\rawnind\tests` (top-level test suite)
- All tests under `src\rawnind\*\tests` (subpackage-specific tests): `dataset`, `dependencies`, `inference`, `training`

Below is a high‑level overview, then a file‑by‑file breakdown per location.

---

### High‑level overview

- Top-level test suite: End-to-end “clean API” tests for datasets, training, inference, and an E2E image-processing
  pipeline; parity tests vs rawpy; acceptance/guardrail tests around refactor (imports/layout, tiny inference engine
  checks, minimal training smoke, dataset base contracts, and experiment utilities). Also a test ensuring legacy
  CLI/modules are not exposed.
- Subpackage suites:
    - `dataset/tests`: Unit/integration tests for legacy dataset loaders, loading-time benchmark mock, and a simplified
      “manproc” pipeline test (mocked) across variants.
    - `dependencies/tests`: Focused unit tests for OpenEXR bit‑depth handling and raw processing conversions.
    - `inference/tests`: Unit tests for denoising/compression components (e.g., BM3D, standard compressor) and
      integration hooks (PyTorch, external RAW denoise, playraw flow).
    - `training/tests`: Unit tests for training pieces like alignment, progressive rawnind flow, and validate/test
      paths.

Markers and stability:

- Acceptance tests are marked `pytest.mark.acceptance`, CPU‑only, lightweight; 0‑XFAIL/0‑SKIP policy.
- Some tests are marked `integration` / `slow` (e.g., raw processing vs rawpy). Others heavily mock I/O to remain
  hermetic.

---

### Top-level: src\rawnind\tests

- `acceptance/README.md`
    - Explains acceptance suite purpose: guardrails for refactor/partition (inference, training, dataset, dependencies,
      imports/layout). Provides how-to-run commands and the 0‑XFAIL/0‑SKIP acceptance policy.

- `acceptance\test_imports_and_layout.py`
    - Verifies public symbols exist in the new module layout. Checks attributes exist on:
        - `rawnind.training.training_loops`: `TrainingLoops`, `ImageToImageNNTraining`, `PRGBImageToImageNNTraining`,
          `BayerImageToImageNNTraining`, `DenoiseCompressTraining`, `DenoiserTraining`
        - `rawnind.training.experiment_manager`: `ExperimentManager`
        - `rawnind.dataset.base_dataset`: `RawImageDataset`, `RawDatasetOutput`
        - `rawnind.inference.inference_engine`: `InferenceEngine`
        - `rawnind.dependencies.utilities`: `load_yaml`, `dict_to_yaml`
    - Marked `acceptance`.

- `acceptance\test_inference_engine.py`
    - Tiny CPU `InferenceEngine` smoke tests with a 1x1 conv model.
    - Tests tensor vs batch input, return_dict mode, and transfer-function factory availability/behavior.
    - Marked `acceptance`.

- `acceptance\test_dataset_base_contracts.py`
    - Defines a tiny in-memory subclass of `RawImageDataset` and asserts `random_crops` and `center_crop` shape
      contracts.
    - Includes import availability check for a newly extracted dataset class (`CleanCleanImageDataset`).
    - Marked `acceptance`.

- `acceptance\test_training_loops_smoke.py`
    - Lightweight contract check for `ImageToImageNNTraining` (import/type); acceptance forbids skips.
    - Marked `acceptance`.

- `acceptance\test_dependencies_and_experiments.py`
    - Tests YAML roundtrip (`dict_to_yaml`/`load_yaml`).
    - Tests `ExperimentManager` helpers: cleaning saved model iterations, removing empty models, reading best steps from
      a results YAML.
    - Marked `acceptance`.

- `conftest.py`
    - Project-wide test configuration/fixtures for this suite (not expanded here; present at
      `src\rawnind\tests\conftest.py`).

- `download_sample_data.py`
    - Helper to fetch/calc sample RAW file paths for tests (used by rawpy parity test).

- `test_e2e_dataset_clean_api.py`
    - Large E2E “clean dataset API” suite with many classes covering:
        - Factory functions (`create_*_dataset`) and config validation (`DatasetConfig`, `validate_dataset_format`).
        - Preprocessing/augmentation (crops/masking, Bayer alignment, augmentation flow).
        - Splits and reserved-test enforcement; metadata/integrity checks.
        - Loading performance behaviors (lazy loading, multiprocessing) via mocks.
        - Specialized dataset types: RawNIND Academic dataset, HDR dataset, color space conversions and Bayer
          demosaicing options.
        - Integration with training (mock trainer), error-handling for missing/insufficient data.
        - Statistics and noise analysis; validation/test loaders; caching.
    - Uses many mock dataloaders to be hermetic.

- `test_e2e_image_processing_pipeline_clean.py`
    - E2E clean API image-processing pipeline scenarios:
        - Realistic RGB/Bayer denoising flows, model loading robustness, memory efficiency, batch processing.
        - Architecture switching and error-handling.
        - A production-like workflow exercising the full clean inference pipeline.
        - Ensures “no legacy CLI imports/side effects” in clean API usage.

- `test_e2e_inference_clean_api.py`
    - E2E clean inference API suite including:
        - Factory creation: `create_rgb_denoiser`, `create_bayer_denoiser`, `create_compressor`.
        - Load model from checkpoint (RGB/Bayer), single/batch inference, metrics computation.
        - Device handling and `InferenceEngine` API.
        - Production model-comparison and memory-efficient workflows; configuration coverage and error cases (invalid
          arch/checkpoint/image shape/metrics).

- `test_e2e_training_clean_api.py`
    - E2E clean training API suite including:
        - Factory functions (`create_denoiser_trainer`, `create_denoise_compress_trainer`, `create_experiment_manager`)
          and config objects (`TrainingConfig`, `ExperimentConfig`).
        - End-to-end workflows for RGB and Bayer denoisers using mocked datasets/dataloaders.
        - Checkpoint save/load and resumption; experiment manager cleanup flows.
        - Standalone validation/testing hooks; scheduling/early stopping; metrics during training.
        - Joint denoise+compress training; Bayer-specific handling (demosaicing, color matrices).
        - Validation of architectures/losses/channel mismatches; multi-device (CPU/GPU) tests; visualization and output
          saving.

- `test_legacy_cli_removed.py`
    - Acceptance test to ensure legacy modules/namespaces are not importable (`rawnind.libs.*`, legacy root modules).
    - Verifies no `__main__`/`main()` blocks in core modules and that `pyproject.toml` declares no legacy console
      scripts.

- `test_raw_processing_vs_rawpy.py`
    - Integration/slow parity tests between package raw processing and `rawpy`:
        - Pixel‑level normalization parity of mono mosaic given RAW metadata.
        - White-balance normalization equivalence.
        - End-to-end demosaic on a real RAW sample producing reasonable RGB outputs.
    - Uses `download_sample_data.get_sample_raw_path` with fallbacks and skips if not available.

- `__init__.py`
    - Package marker for the tests package.

---

### Subpackage: src\rawnind\dataset\tests

- `test_datasets.py`
    - Legacy dataset class unit tests (unittest style) for multiple dataset flavors: clean/noisy PRGB and Bayer
      variants, validation datasets, and test dataloaders. Also references old `...libs.rawproc` and
      `arbitrary_proc_fun`.

- `test_datasets_load_time.py`
    - Converts a benchmark into a unit test using mocks. Parametrized over four dataset classes; verifies a mock timing
      dict with `min/avg/max` behaves sanely (avg > min, etc.).

- `test_manproc.py`
    - Simplified, parametrized tests simulating a “manproc” pipeline over combinations of `model_type` (dc/denoise),
      `input_type` (bayer/prgb/proc), and `variant` (basic/ext_raw/progressive).
    - Uses `MagicMock` model and mocked `offline_custom_test` that writes to a `json_saver`. Some branches call
      `pytest.skip` based on a mocked “high loss” sentinel.

- `test_validation.py`
    - Present in directory; not expanded above, but by name likely covers dataset validation checks (e.g.,
      shapes/consistency). [File exists in listing.]

- `__init__.py`

---

### Subpackage: src\rawnind\dependencies\tests

- `test_openEXR_bit_depth.py`
    - Tests correctness around OpenEXR bit-depth conversions/handling.

- `test_raw_processing.py`
    - Tests selected raw processing utilities, including scene‑linear to PQ transform (`scenelin_to_pq`, etc.).

- `__init__.py`

---

### Subpackage: src\rawnind\inference\tests

- `test_bm3d_denoiser.py`
    - Unit tests for a BM3D denoiser integration (shape/basic behavior assertions).

- `test_ext_raw_denoise.py`
    - Tests for an “external raw denoise” path (likely integration mock around raw pipeline and denoiser).

- `test_playraw.py`
    - Tests a “playraw” flow (probably demo/sample processing path) ensuring the pipeline runs and outputs sensible
      tensors.

- `test_pytorch_integration.py`
    - Checks PyTorch integration behaviors used by inference components.

- `test_standard_compressor.py`
    - Unit tests for `models.standard_compressor.JPEG_ImageCompressor` forward pass and error handling.
    - Mocks external binaries/I/O; asserts nonzero bpp, reasonable PSNR approximation, and correct error on invalid
      channels.

- `__init__.py`

---

### Subpackage: src\rawnind\training\tests

- `test_alignment.py`
    - Tests alignment utilities used during training (e.g., align noisy/clean pairs).

- `test_progressive_rawnind.py`
    - Tests a “progressive rawnind” training flow (progressive schedule/logic) for correctness.

- `test_validate_and_test.py`
    - Tests the validation and testing routines invoked by trainers.

- `__init__.py`

---

### Notable markers and patterns

- Acceptance tests: marked `pytest.mark.acceptance`; 0‑XFAIL/0‑SKIP policy across acceptance.
- Integration/slow: `test_raw_processing_vs_rawpy.py` is marked `integration` and `slow`, fetches/uses a small RAW
  sample and may skip if unavailable.
- Mocking/I/O isolation: Many tests rely on `tmp_path`, heavy use of `unittest.mock` to avoid real filesystem or GPU
  requirements.

---

### Quick index by path

- `src\rawnind\tests\acceptance\`: package guardrails and tiny‑CPU smoke tests
- `src\rawnind\tests\`: E2E clean API tests for dataset, training, inference, pipeline; legacy CLI removal; raw vs rawpy
  parity
- `src\rawnind\dataset\tests\`: dataset loaders/performance/manproc pipeline
- `src\rawnind\dependencies\tests\`: raw/OpenEXR utility tests
- `src\rawnind\inference\tests\`: model/inference component tests
- `src\rawnind\training\tests\`: training utility/flow tests

If you want, I can extract specific test names within any file/class (e.g., list all `test_*` methods) or map them to
features/components in a matrix.

---

E2E mocking audit and proposals (2025-09-23)

Where these suggestions live
- This section documents the audit and concrete enhancement proposals that I referenced in discussions. It provides exact locations in code where mocks/patches circumvent public interfaces in the four top-level E2E suites, plus actionable remediation suggestions.

Scope: Four E2E files under src\rawnind\tests
- test_e2e_dataset_clean_api.py
- test_e2e_inference_clean_api.py
- test_e2e_image_processing_pipeline_clean.py
- test_e2e_training_clean_api.py

Instances that circumvent public interfaces
1) test_e2e_dataset_clean_api.py
   - Backdoor dataloader injection via data_loader_override (multiple locations):
     • Around lines ~814–818: create_training_dataset(..., data_loader_override=mock_bayer_dataloader())
     • Around lines ~908–912: create_training_dataset(..., data_loader_override=mock_dataloader())
     • Around lines ~953–957: create_training_dataset(..., data_loader_override=mock_dataloader_with_missing())
     Why this circumvents: Injects arbitrary iterables into dataset factory, bypassing config validation, file discovery, and preprocessing hooks that the public API is meant to exercise.

   - Cross-package patching of training factory:
     • Around lines ~830–841: @patch('rawnind.training.create_denoiser_trainer') with a Mock trainer
     Why this circumvents: Replaces the public training-creation path rather than going through the supported integration point exposed by training APIs.

   Remediation proposals:
   - Define and document a supported synthetic dataset hook in the public API, e.g., create_synthetic_dataset(config) or a DatasetConfig.testing_mode flag. Deprecate data_loader_override from the public surface and confine it to internal tests.
   - Provide a tiny on-disk fixture generator (public utility) that creates 2–3 synthetic images per split to exercise file discovery and preprocessing end-to-end without network or large assets.
   - For trainer integration, introduce a public TrainerInterface protocol and a minimal create_null_trainer() for tests, removing the need to patch.

2) test_e2e_training_clean_api.py
   - Cross-package patching of dataset factory:
     • Around lines ~527–544: @patch('rawnind.dataset.create_training_dataset') with a generator-returning mock
     Why this circumvents: Avoids the public dataset creation path and format validation the trainer expects.

   Remediation proposals:
   - Publicly support prepare_datasets with DatasetConfig objects (no patch) and allow a minimal in-memory dataset via a first-class SyntheticDataset provider in rawnind.dataset.
   - Offer a tiny, supported TestDataLoader.from_tensors(...) helper under rawnind.dataset.testing to create loaders from tensors without touching private parameters.

3) test_e2e_inference_clean_api.py
   - No monkeypatch/patch detected for core flows. Uses public creators and tmp paths. Keep as-is.

4) test_e2e_image_processing_pipeline_clean.py
   - No monkeypatch/patch detected for core flows. Uses public creators and tmp paths. Keep as-is.

Cross-cutting proposals to raise the bar (acceptance and E2E alignment)
- Add a supported synthetic data path in both dataset and training packages so E2E tests don’t need backdoors.
- Provide small on-disk image fixture generation utilities to test file-based pipelines hermetically.
- Expose light-weight testing hooks (documented) rather than permitting arbitrary override parameters.
- Prefer protocol-based interfaces (typing.Protocol) for trainers/datasets to allow dependency inversion in tests without patching internals.

Rationale
- These changes preserve hermetic, fast tests while ensuring we exercise public APIs exactly as users would, improving confidence and reducing coupling to private internals.