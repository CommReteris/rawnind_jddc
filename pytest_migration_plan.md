# Pytest Migration Plan for RawNIND Test Suite

## Current State
The test suite is partially migrated to pytest:
- `pyproject.toml` includes pytest and pytest-cov as test dependencies.
- Some files like `test_pytorch_integration.py` use pytest (imports, test methods, `pytest.main()`).
- Most tests (`test_manproc_*.py`, `validate_and_test_*.py`) are standalone scripts using custom `rawtestlib` subclasses for null dataloaders and `offline_custom_test()`.
- `rawtestlib.py` provides lightweight test classes inheriting from training classes (e.g., `DCTestCustomDataloaderBayerToProfiledRGB`) to skip dataloading for fast validation.
- Execution relies on custom tools like `test_all_known.py` and shell scripts; not full pytest discovery.
- Presence of `.pytest_cache/` indicates occasional pytest runs, but inconsistency persists.
- AGENTS.md documents custom testing patterns, not pytest.

## Identified Issues
1. **Inconsistent Testing Styles**: Mix of pytest, custom classes with manual `if __name__ == "__main__"`, and executable scripts. Hard to discover/run uniformly.
2. **Lack of Pytest Features**:
   - No fixtures for reusable setup (e.g., device from `pt_helpers.get_device()`, model instantiation via rawtestlib).
   - No parametrization for testing multiple model types (e.g., bayer2prgb vs. prgb2prgb, dc vs. denoise).
   - Limited assertions; some use `print()` for output, risking uncaught failures.
   - No skipping for known issues (e.g., MSSSIM skips in scripts) or expected failures.
3. **Custom Framework Dependencies**: Tests tightly coupled to training classes; `get_dataloaders()` overridden to None, but no mocking for full pipeline tests.
4. **Execution and Discovery**: Relies on manual invocation; no `pytest src/rawnind/tests` support. Custom `test_all_known.py` needs integration.
5. **Device and Environment Handling**: Assumes CUDA/CPU availability; no fixture to control/switch devices for CI/portability.
6. **Coverage and Reporting**: pytest-cov available but not configured; no HTML reports or thresholds.
7. **Documentation**: AGENTS.md and README.md need updates for pytest commands.
8. **Other**: Some tests output images/files without cleanup; potential for leftover artifacts in `tests_output/`.

## Migration Plan
### Phase 1: Setup and Configuration (Architect/Code)
- Update `pyproject.toml`:
  - Add `[tool.pytest.ini_options]` section for discovery (`testpaths = src/rawnind/tests`, `python_files = test_*.py`, `python_classes = Test*`, `python_functions = test_*`).
  - Configure coverage: `addopts = --cov=src/rawnind --cov-report=html --cov-report=term-missing`.
  - Markers: `@pytest.mark.model_type("dc")`, etc., for filtering.
- Create `conftest.py` in `src/rawnind/tests/` for shared fixtures:
  - `device`: Yield `pt_helpers.get_device()`.
  - `model_bayer_dc`: Fixture yielding `rawtestlib.DCTestCustomDataloaderBayerToProfiledRGB(launch=False)`.
  - `model_prgb_dc`, `model_bayer_denoise`, `model_prgb_denoise`: Similar for other types.
  - `sample_dataloader`: Mock or skip for full tests; use real for integration.
  - `tmp_output_dir`: Fixture for `tests_output/` cleanup.

### Phase 2: Convert Test Files (Code Mode)
- **Standalone Scripts** (e.g., `test_manproc_dc_bayer2prgb.py`):
  - Convert to pytest functions: `def test_manproc_dc_bayer2prgb(model, dataloader):` using fixtures.
  - Replace `if __name__ == "__main__":` with pytest structure.
  - Parametrize: `@pytest.mark.parametrize("model_type", ["bayer", "prgb"])` to consolidate similar tests.
  - Update skips: Use `@pytest.mark.skipif(condition, reason="Known MSSSIM issue")`.
  - Assertions: Replace prints with `assert "manproc_msssim_loss" in results`.
- **Custom Classes** (e.g., `test_pytorch_example.py`):
  - Move methods to pytest: `def test_model_device_placement(self, device): assert ...`.
  - Use `self` only if class needed; prefer functions for simplicity.
- **Validation Scripts** (e.g., `validate_and_test_dc_bayer2prgb.py`):
  - Similar conversion; parametrize for dc/denoise.
- **Batch Conversion**:
  - Group by pattern: manproc_* (30+ files) → parametrized modules (e.g., `test_manproc.py` with params).
  - progressive_* and playraw_* → Similar grouping.
  - Non-model tests (e.g., `test_openEXR_bit_depth.py`, `test_alignment.py`) → Direct pytest conversion.
- **rawtestlib Integration**:
  - Keep as-is for fixtures; expose via conftest.py.
  - Add `offline_custom_test` wrapper as fixture or method.

### Phase 3: Fix Issues and Enhance (Python Engineer Mode)
- **Assertions**: Ensure all use `assert`; add tolerance for float comparisons (e.g., `np.testing.assert_allclose`).
- **Device Consistency**: Use fixture; test on CPU, skip GPU-specific if needed.
- **Mocking**: For dataloader-dependent tests, use `pytest.skip` or `monkeypatch` to mock `get_dataloaders`.
- **Cleanup**: Use `tmpdir` fixture for outputs; add `--debug_options` as param.
- **Coverage**: Run `pytest --cov` post-migration; aim >80% on models/libs.
- **Error Handling**: Integrate custom `error_handler` from AGENTS.md.

### Phase 4: Update Execution and Docs (Python Engineer/Architect)
- Modify `src/rawnind/tools/test_all_known.py`:
  - Add `pytest.main(["src/rawnind/tests", "-v", "--cov"])` for full suite.
  - Preserve `--tests test_manproc --model_types dc` via pytest markers/params.
- Update `test_all_needed.sh` to `pytest src/rawnind/tests`.
- Docs:
  - Update AGENTS.md: Add pytest commands, e.g., "Run all tests: `pytest src/rawnind/tests -v`".
  - Update README.md: Section on "Testing with Pytest".
  - Add to this plan file: Usage examples.

### Phase 5: Verification (Python Engineer Mode)
- Run `pytest -v` on converted files; fix failures.
- Full suite: `pytest src/rawnind/tests --cov`.
- Compare outputs with old custom runs.
- CI: Suggest adding to pyproject.toml or .github/workflows.

## Current Discrepancies from Pytest Run
From baseline pytest execution (19% coverage, 4 failures, 1 error in 5 collected tests):

1. **ArgParse Error in test_manproc_dc_bayer2prgb**:
   - Error: SystemExit from missing required args (--arch, --match_gain, etc.) in DCTestCustomDataloaderBayerToProfiledRGB init via abstract_trainer.get_args().
   - Cause: conftest.py fixture attempts monkeypatch for get_args but incomplete (configargparse not imported, mock_get_args defined inside fixture without proper Namespace).
   - Impact: Prevents manproc test collection/execution; custom scripts not discovered.
   - Adaptation: Complete monkeypatch in conftest.py with full preset_args dict (e.g., arch="DenoiseThenCompress", in_channels=4); use preset_args kwarg in rawtestlib subclasses for hermetic init without CLI.

2. **Failures in test_pytorch_integration.py (4 tests)**:
   - test_model_device_placement: StopIteration (next(model.parameters()) fails as model has no parameters).
   - test_model_forward_pass: assert isinstance(output, dict) fails (forward returns None).
   - test_model_inference_method & test_batch_handling: AttributeError (no 'infer' method).
   - Cause: Uses abstract AbstractRawImageCompressor (forward=pass, no params, no infer); outdated API assuming concrete implementation.
   - Refactoring: AbstractRawImageCompressor is base; concrete like DenoiseThenCompress/ManyPriors_RawImageCompressor implement forward returning dict with "reconstructed_image", "bpp". 'infer' in ImageToImageNN base, called on self.model.
   - Adaptation: Update tests to use concrete fixture (e.g., model_bayer_dc.model as DenoiseThenCompress); verify forward returns dict, infer handles batch/single; add param for model type.

3. **Low Test Collection (5/50+ files)**:
   - Cause: Most tests (test_manproc_*.py, validate_and_test_*.py) are standalone scripts with if __name__ == "__main__": offline_custom_test(); not pytest-discoverable.
   - Adaptation: Convert to pytest functions using fixtures (e.g., def test_manproc_dc_bayer2prgb(model_bayer_dc): model_bayer_dc.offline_custom_test(...)); parametrize across types.

No obsolete tests identified yet; all target existing functionality (manproc via rawproc, models via libs). Coverage low in core (abstract_trainer 21%, raw.py 12%) due to uncollected scripts.

## Test Mapping and Removals/Adaptations
Mapping old custom tests to new pytest equivalents (no removals; all adaptable as refactored code preserves intent):

- **test_manproc_*.py (30+ files, e.g., test_manproc_dc_bayer2prgb.py)**:
  - Old: Standalone script instantiating rawtestlib subclass, calling offline_custom_test(manproc_dataloader).
  - New: pytest function def test_manproc_dc_bayer2prgb(model_bayer_dc, manproc_dataloader): with model_bayer_dc.offline_custom_test(manproc_dataloader); parametrize("model_type", ["dc_bayer", "dc_prgb", "denoise_bayer", "denoise_prgb"]).
  - Reason: Preserves manproc validation; fixtures handle init. Group into test_manproc.py for consolidation.

- **test_playraw_*.py (4 files)**:
  - Old: Similar to manproc but uses playraw dataloader.
  - New: def test_playraw_dc_bayer2prgb(model_bayer_dc, playraw_dataloader): model_bayer_dc.offline_custom_test(...); parametrize for variants.
  - Reason: playraw tests specific loader; adapt to fixture.

- **test_progressive_*.py (8 files)**:
  - Old: Progressive manproc/playraw with ge/le thresholds.
  - New: Parametrize threshold in test_progressive_manproc(model, threshold="ge/le"); use manproc_dataloader fixture.
  - Reason: Progressive is param variant; consolidate.

- **validate_and_test_*.py (4 files)**:
  - Old: Validation wrappers calling offline_std_test/offline_custom_test.
  - New: Separate functions test_validate_dc_bayer(model_bayer_dc), test_custom_denoise_prgb(model_prgb_denoise, custom_dataloader); mark as integration.
  - Reason: Validation intent preserved; fixtures enable.

- **test_pytorch_integration.py**:
  - Old: Unittest-style class testing abstract compressor.
  - New: Convert to pytest functions using concrete model fixture; update assertions for forward dict, infer method.
  - Adaptation: Use DenoiseThenCompress fixture; verify bpp, reconstructed_image.

- **Other (test_alignment.py, test_openEXR_bit_depth.py, etc.)**:
  - Old/New: Direct conversion to pytest (no major changes needed); add fixtures for device/tmp_path.

No removals: All map to refactored components (rawproc for manproc, concrete models for integration). For replaced (e.g., old abstract forward), add new tests for DenoiseThenCompress forward/bpp.

## Coverage Improvement Strategies (Target 5-10% Gain: 19% → 24-29%)
Baseline: 19% total; low in libs/models (abstract_trainer 21%, raw.py 12%, compression_autoencoders 33%). Many 0% in tools/scripts due to non-collection.

1. **Conversion to Pytest (3-5% gain)**: Collect 50+ tests; each script conversion covers init/args (e.g., rawtestlib 67% → 100% with param tests).

2. **Add Unit Tests for Core Components (4-6% gain)**:
   - Models: Parametrized tests for forward/infer with dummy tensors (e.g., @pytest.mark.parametrize("in_ch", [3,4]) def test_denoise_then_compress_forward(model, dummy_input): assert "reconstructed_image" in output).
   - Libs: Test rawproc.match_gain, pt_helpers.get_device edge cases (CPU/GPU switch, invalid input); mock torch for hermetic.
   - Trainers: Test get_args with mock CLI, autocomplete_args validation.

3. **Edge Cases & Broader Applicability (2-3% gain)**:
   - Refactored: Test DenoiseThenCompress bpp with varying num_distributions; infer single vs. batch.
   - Error Conditions: Test invalid in_channels (raises ValueError), empty batch (skips gracefully).
   - Devices: Parametrize device fixture (CPU/CUDA); verify tensor placement.

4. **Integration Enhancements**: Use manproc_dataloader for end-to-end; add coverage for untested branches (e.g., preupsample=True in Bayer models).

Use pytest-cov thresholds in pyproject.toml; focus on models/libs (target 40-50%).

## Handling Failing Tests
- **ArgParse (manproc error)**: Fix conftest.py monkeypatch: Import configargparse, define mock_get_args outside fixture returning Namespace(preset_args); pass preset_args to rawtestlib init. Remove SystemExit by setting test_only=True early.
- **Integration Failures**: Replace AbstractRawImageCompressor with concrete DenoiseThenCompress fixture; implement missing forward (returns dict); add infer via ImageToImageNN base. Update assertions: Check output["reconstructed_image"].shape, output["bpp"] > 0; verify infer adds batch dim.
- **General**: For coverage loss from adaptations, add equivalents (e.g., test_abstract_init skipped, but test_concrete_denoise_then_compress_forward covers intent). No removals; all fixable without prod changes.

Preserve intent: Verify model device/placement, forward dict output, infer batch handling, param iteration.

## Risks and Mitigations
- Breaking existing fast tests: Keep rawtestlib; add `--no-dataloader` marker.
- Performance: Fixtures cache models; null dataloaders remain fast.
- Compatibility: Test on CPU/GPU; use `torch.device("cpu")` default.
- Scope: Prioritize manproc/playraw (core); defer niche like grapher.py.

## Timeline
- Phase 1-2: 1-2 iterations (convert 5-10 files first).
- Phase 3-4: After verification.
- Total: 4-6 tool uses in code mode.

This plan ensures full pytest adoption while preserving custom utilities.