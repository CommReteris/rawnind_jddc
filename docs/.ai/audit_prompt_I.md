### Prompt for AI Agent: Verify Codebase Compliance with Refactoring Rules

You are an AI software engineer tasked with verifying the compliance of the RawNIND PyTorch codebase after a major refactoring. The codebase was refactored from a monolithic structure (large files like abstract_trainer.py at 2497 lines, rawds.py at 1704 lines; tight coupling, mixed responsibilities, CLI dependencies via argparse/configargparse) into four modular packages: inference/, training/, dataset/, and dependencies/. The goal, per docs/.ai/partition_plan.md (focus lines 177-320), was to extract/rewrite legacy logic into clean programmatic APIs (dataclasses like TrainingConfig/InferenceConfig/DatasetConfig for configs; factories like create_rgb_denoiser, create_training_datasets for instantiation), minimize inter-package dependencies (e.g., training imports dataset loaders via clean interfaces), eliminate CLI entirely (sacrifice wrappers, integrate core logic), preserve domain intent (e.g., Bayer 4-channel RGGB demosaicing with resolution doubling to 3-channel RGB, MS-SSIM constraints >160px image size, rate-distortion optimization with multi-parameter AdamW groups, PyTorch 2.6+ checkpoint security via weights_only=False and add_safe_globals), and remediate anti-patterns (no duplicate impls, consistent names like ms_ssim, single source of truth via configs).

The refactoring followed strict TDD (pytest with hermetic mocks only for I/O/externals like file paths via tmp_path or libs like colour via MagicMock; no mocking real pipeline parts like dataloaders/models/losses in e2e tests), iterative remediations (I-VII docs detail CLI removal, placeholder replacements with extracted logic like validate_or_test ~280 lines with locks/caching, duplicate consolidation e.g., TestDataLoader from 3 files to 1), and legacy code as a "handbook" for domain (rewritten, not verbatim copied). Tools/ directory logic integrated into APIs (e.g., prep_image_dataset → dataset_preparation.py factory), CLI wrappers deleted. Tests treat failures as architectural diagnostics (e.g., interface mismatches from incomplete extraction).

**Project Rules (from .roo/rules/rules.md—must strictly follow):**
- **General**: Use available tools as described; never simplify/mock real pipeline parts in integration testing.
- **Anti-Patterns**: Never >1 impl of similar things; never >1 name unless necessary (e.g., canonical ms_ssim, no msssim_loss); never >1 sole source of truth (e.g., all params via dataclasses, no scattered literals).
- **Testing**: Run via `python -m pytest` + normal args (e.g., -v, -k "pattern", --markers); DO NOT generate HTML coverage reports (use --cov-report=term-missing for console summary if needed).
- **MCP Tool Rules**: Prioritize MCP (e.g., pycharm for searches/refactors, filesystem for lists/reads, sequentialthinking for analysis); chain as needed (e.g., search then summarize); fallback gracefully; respect limits (e.g., allowed dirs).

**Your Task**: Execute the following steps methodically to verify compliance. Use MCP tools first (e.g., pycharm search_in_files_by_text for duplicates, get_project_problems for issues, execute_command for pytest via <execute_command><command>python -m pytest -v --cov=src/rawnind --cov-report=term-missing</command></execute_command>). Report results clearly: Pass/fail per step, evidence (e.g., search outputs), and any violations/fixes needed (propose TDD: add failing test first, then code). If issues found, suggest targeted remediations (e.g., consolidate duplicates via rename_refactoring). Workspace: c:/Users/Rengo/PycharmProjects/rawnind_jddc. Do not assume—read files if needed (filesystem read_text_file). Output in structured format: Step #, Command/Tool Used, Results, Compliance Verdict.

**Step-by-Step Instructions:**

1. **Run Full Test Suite (Verify Functionality & No Real-Pipeline Mocks)**:
   - Execute `python -m pytest -v --cov=src/rawnind --cov-report=term-missing` (console coverage only, no HTML).
   - Focus: 100% pass rate? E2E tests (e.g., test_e2e_training_clean_api.py, test_e2e_image_processing_pipeline_clean.py) use real dataloaders/models/losses (no mocks of core like Bayer demosaicing or MS-SSIM)? Coverage >90% on src/rawnind (check term-missing for gaps in training_loops.py or raw_processing.py)?
   - If fails: Use --tb=short for diagnostics; trace to anti-patterns (e.g., interface mismatch)? Propose fix: Add parametrized test for variant (e.g., Bayer crop_size=256 >160), then implement.
   - Verdict: Pass if all pass, coverage high, no core mocks evident in failures.

2. **Search for Anti-Pattern: Duplicate Implementations**:
   - Use pycharm search_in_files_by_text query="class TestDataLoader|ProfiledRGBBayerImageDataset|RawImageDataset" scope=src/ (expect exactly 1 match each; >1 indicates duplicate).
   - Follow-up: codebase_search query="multiple implementations of Bayer demosaicing or MS-SSIM loss" (semantic check for similar code blocks).
   - If duplicates: Use rename_refactoring or edit_file to consolidate (e.g., keep in bayer_datasets.py, delete elsewhere); add test test_anti_patterns.py asserting uniqueness via ast parsing.
   - Verdict: Pass if single impls only.

3. **Search for Anti-Pattern: Multiple Names**:
   - pycharm search_in_files_by_text query="ms_ssim|msssim|MS-SSIM" scope=src/ (expect only "ms_ssim"; variants indicate inconsistency).
   - Extend: search_in_files_by_regex pattern="demosaic|denoiser|bpp" (check canonical like "demosaic" only, no "bayer_demosaic_fn").
   - If issues: replace_text_in_file to standardize (e.g., replace "msssim" with "ms_ssim"); update tests (e.g., test_pt_losses.py assert name).
   - Verdict: Pass if consistent naming.

4. **Search for Anti-Pattern: Multiple Sources of Truth**:
   - find_files_by_name_keyword "learning_rate|crop_size|batch_size" (expect refs only to config attrs like self.config.learning_rate; no hardcodes).
   - get_project_problems on src/rawnind/training/clean_api.py (check for scattered params or unused imports indicating remnants).
   - If violations: Refactor to centralize in dataclasses (e.g., TrainingConfig); add validation test in test_training_config.py.
   - Verdict: Pass if all via single config sources.

5. **Verify CLI Elimination**:
   - pycharm search_in_files_by_regex pattern="import argparse|configargparse|add_argument" scope=src/ (expect 0 in production; only in tests/docs like test_legacy_cli_removed.py).
   - If found: Use replace_text_in_file to remove; confirm via test_legacy_cli_removed.py (ast scan for no argparse).
   - Verdict: Pass if zero CLI in core.

6. **Validate Modular Structure**:
   - list_directory_tree path=src/rawnind/ (confirm packages: inference/ with clean_api.py, no tools/ remnants; configs only in dependencies/configs/; no mixed files).
   - find_files_by_glob "**/*.py" file_pattern="train_*|infer_*" (expect in training/inference/, not root).
   - Verdict: Pass if matches partition plan (e.g., training_loops.py in training/, raw_processing.py in dependencies/).

7. **Check VCS & Overall Health**:
   - get_project_vcs_status (review uncommitted: delete unversioned legacy backups like legacy_rawds.py if unused; no anti-patterns in diffs).
   - find_commit_by_message "anti-pattern duplicate|CLI removal|extraction" (audit history for fixes; expect commits addressing remediations I-VII).
   - get_project_problems (global: no errors/warnings in src/; focus extraction completeness like missing imports in training_loops.py).
   - Verdict: Pass if clean, no regressions.

**Final Output**: Summarize verdicts (e.g., "All pass: Modular, CLI-free, no anti-patterns"). If violations, prioritize TDD fixes (e.g., "Add failing test for duplicate, then consolidate"). Use MCP chaining (e.g., search → sequentialthinking for analysis). Report any domain gaps (e.g., Bayer tests fail? Trace to extraction).