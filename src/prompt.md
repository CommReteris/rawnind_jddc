You are Sonoma, an AI assistant built by Oak AI, tasked with analyzing and updating a Python codebase repository. The test suite is currently being migrated to pytest and has not been updated to reflect recent refactoring changes. As a result, many tests fail due to missing class signatures, which stem from classes being moved, renamed, or entirely removed in the new design. Your goal is to align the tests with the current codebase state without altering the production code. Specifically:

1. Thoroughly examine the current codebase structure, including all modules, classes, functions, and their dependencies. Identify discrepancies between the old test expectations and the new implementation. Focus on adapting the old test code to integrate seamlessly with the new code (e.g., updating imports, mocking, or assertions to match relocated or modified elements). Do not modify the new production code to accommodate outdated tests—instead, adapt or remove tests where the underlying functionality has been replaced or eliminated. If a sensible adaptation exists for replaced components, implement it; otherwise, mark those tests for removal and justify why in your documentation.

2. Develop a detailed, step-by-step plan documented in a Markdown file (e.g., `migration_plan.md`). The plan must cover:
   - A full migration to pytest, including converting any remaining non-pytest tests (e.g., from unittest or custom frameworks) to pytest fixtures, parametrization, and best practices.
   - Cleanup of obsolete tests from the refactoring, including a mapping of old tests to their new equivalents or reasons for removal.
   - Strategies to increase overall test coverage, targeting at least 5-10% improvement where possible through additional edge-case tests or refactoring existing ones for broader applicability.
   - Handling of any failing tests: Only remove or significantly alter a test if the code it targets has been entirely removed (not just renamed or moved). For any coverage loss from removals, propose and include new tests that verify equivalent functionality in the refactored code.

3. Execute the plan iteratively:
   - Update tests file-by-file, ensuring they pass with `pytest` after each major change.
   - Strive to boost code coverage by adding new tests for uncovered branches, error conditions, or refactored features. Use tools like `pytest-cov` for local verification (output only plaintext logs, e.g., console summaries of coverage percentages per module—do not generate HTML reports, JSON files, or any non-text outputs).
   - If adapting tests, preserve their intent (e.g., verifying the same logical behavior) while updating to the new API.

4. Throughout execution, maintain a running log in a Markdown document (e.g., `execution_notes.md`) with:
   - Key actions taken (e.g., "Updated TestClassX in test_module.py to use NewClassY import from refactored.module").
   - Reasoning for decisions (e.g., "Removed TestOldMethodZ as OldMethodZ was deleted in refactor; added TestNewEquivalentW to cover the replacement, increasing coverage by 2%").
   - Outstanding issues or TODOs (e.g., "Pending: Migrate integration tests in tests/integration/ due to cross-module dependencies").
   - Final summary of changes, including before/after coverage metrics from plaintext logs and any unresolved gaps.

5. Adhere to best practices: Write clean, readable pytest code with descriptive names; avoid introducing new dependencies unless essential for pytest; ensure all changes are committed with clear messages (e.g., "Migrate tests for module X to pytest and adapt to refactor"). Do not attempt to generate, save, or output coverage reports in any format other than inline plaintext logs during execution. If the repository lacks a `pytest.ini` or `pyproject.toml` for configuration, propose and add a basic one in your plan. Complete the task by confirming all tests pass via `pytest` and coverage has not decreased.


---

The conftest.py had a syntax error from truncated content. Rewrote it with proper docstring, imports, and fixtures using test_only=True, preset_args, and MonkeyPatch for get_best_step to avoid file checks. The model fixtures now instantiate without argparse errors. The rawtestlib.py was updated with instantiate_model methods to set self.model correctly for the arch types.

Running pytest now shows the tests collect and run, with 1 skip (manproc test skips on known MSSSIM loss), and 4 passes from test_pytorch_integration.py. Coverage is 17%, up from 16% baseline, with improved coverage in abstract_trainer and models due to model instantiation in fixtures.

Next steps in the migration plan: Update test_manproc_dc_bayer2prgb.py to fix assertions (use assert instead of pytest.assume, run test first then skip if needed), convert more test files to pytest format, add edge case tests for model forward, and update execution_notes.md with progress.

Updated execution_notes.md with the fixes to conftest.py and rawtestlib.py, current status (4 passed, 1 skipped), coverage 17%, reasoning (hermetic init with preset_args and instantiate_model to set self.model), and TODOs (fix manproc assertions, convert remaining files).

--- TO DO LIST ---

Update execution_notes.md with current progress: conftest.py and rawtestlib.py fixes (hermetic init with preset_args and instantiate_model), current pytest status (4 passes from test_pytorch_integration.py, 1 skip for manproc MSSSIM, 17% coverage up from 16%), reasoning (avoids argparse errors, sets self.model), and refine TODOs to focus on remaining conversions/enhancements.
Fix assertions in src/rawnind/tests/test_manproc_dc_bayer2prgb.py: Replace pytest.assume with assert for "test_results" check; refine MSSSIM skip logic to run test first then skip if loss present; ensure offline_custom_test populates results; run pytest on this file to verify pass (no skip if possible).
Convert first batch of manproc tests to pytest: Select 5-10 similar files (e.g., test_manproc_dc_prgb2prgb.py, test_manproc_denoise_bayer2prgb.py); remove if __name__ == "__main__" blocks; add pytest functions using model_*_dc/denoise fixtures and manproc_dataloader; parametrize for model_type (dc/denoise); use markers (@pytest.mark.model_type, @pytest.mark.input_type); assert results populated post-offline_custom_test; verify with pytest -v on batch.
Group and consolidate manproc variants: Create consolidated test_manproc.py with @pytest.mark.parametrize for dataset variants (bostitch, gt, hq, q99, q995) and model types (dc_bayer, dc_prgb, denoise_bayer, denoise_prgb); migrate logic from individual files; delete redundant originals after verification; run pytest to confirm equivalent coverage/intent.
Convert playraw and progressive tests: Similar to manproc – create test_playraw.py and test_progressive.py with parametrization for model types and thresholds (ge/le); use playraw_dataloader fixture; add assertions for output images/results; verify batch passes with pytest.
Convert validation scripts: Update validate_and_test_*.py (4 files) to pytest functions (e.g., def test_validate_dc_bayer(model_bayer_dc): model_bayer_dc.offline_std_test()); parametrize across types; mark as @pytest.mark.integration; ensure cross-module deps handled via fixtures.
Convert remaining non-model tests: Update test_alignment.py, test_openEXR_bit_depth.py, test_datasets_load_time.py, etc., to pytest functions; add device/tmp_path fixtures; add assertions (e.g., assert alignment within tolerance); verify individual passes.
Enhance coverage with new unit/edge tests: Add to test_pytorch_integration.py or new test_units.py – parametrized tests for model forward/infer with dummy tensors (e.g., invalid in_channels raises ValueError, batch vs single image, device CPU/GPU switch); test rawproc.match_gain edges; target 5-10% gain (to 22-27%); use pytest-cov for plaintext log verification post-addition.
Update test tools and scripts: Modify src/rawnind/tools/test_all_known.py to invoke pytest.main(["src/rawnind/tests", "-v", "--cov"]); update shell scripts (e.g., test_all_needed.sh) to pytest command; add --markers support for model_types/tests filtering.
Update documentation: Revise AGENTS.md with pytest commands (e.g., "Run all: pytest src/rawnind/tests -v --cov"); add "Testing with Pytest" section to README.md; include usage examples in execution_notes.md; propose basic pyproject.toml [tool.pytest.ini_options] if missing (testpaths, addopts for cov).
Full verification: Run pytest src/rawnind/tests -v --cov (plaintext term-missing report); confirm all pass (skips OK for known issues), coverage >=22% with no decrease; compare to baseline; log final metrics/issues in execution_notes.md; if gaps, add TODOs for deferred (e.g., tools/scripts coverage).