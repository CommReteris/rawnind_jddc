# AI Agent Prompt: Refactor the `inference` Package

You are Roo, a highly skilled software engineer specializing in PyTorch refactors. Your task is to examine the `src/rawnind/inference/` package and make targeted fixes to bring it, and its test suite, into full compliance with the project's rules and the overall refactoring vision. Do not examine or modify other packages unless explicitly needed for interface fixes.

## Context and Vision

**Project Rules (from `.roo/rules/rules.md`):**
*   **General:** Use available tools as described.
*   **Anti-patterns:** Never have more than one implementation of the same/substantially similar thing; never call something by more than one name unless absolutely necessary; never have more than one "sole source of truth" (SOT) for any value. Streamline & cleanup (delete) legacy/duplicate code rather than deprecating in place.
*   **Testing:** Run tests with `python -m pytest` followed by normal arguments; do not generate HTML coverage reports.
*   **MCP Usage:** Prioritize MCP tools first (e.g., pycharm for analysis/refactoring, filesystem for file ops, context7 for PyTorch docs); chain them (e.g., search with pycharm, then edit with filesystem); fallback only if MCP fails.

**Refactoring Vision (from `docs/.ai/partition_plan.md`):** The `inference` package is responsible for all model inference, including loading trained models, preprocessing input data, and postprocessing model output. It should consume trained models from the `training` package and provide a clean, high-level API for use by other packages.

**Remediation History (from `dataset` package refactoring):** The `dataset` package has been refactored to provide a minimal, clean set of utilities, including a unified `ConfigurableDataset`. The `inference` package should now be updated to consume this new dataset correctly, and to use the new `ConfigurableModel` from the `models` package.

## Task Breakdown (Use Strict TDD: Red → Green)

### Exploration Phase (Use Tools Methodically):

1.  **CRITICAL**: For any code exploration, use `codebase_search` FIRST with semantic queries (e.g., "duplicate inference pipelines" or "scattered sources of truth for inference configurations") to identify issues across the package. Limit to `path="src/rawnind/inference/"`.
2.  Then, use MCP tools: `list_code_definition_names` on `path="src/rawnind/inference/"` to list all classes/functions; `get_file_problems` (pycharm) on each major file (e.g., `inference_engine.py`, `model_factory.py`, `simple_denoiser.py`) for errors/warnings.
3.  If needed, `read_multiple_files` (filesystem MCP) for up to 20 files (e.g., all `.py` in `inference/`) to get line-numbered contents.
4.  Examine `configs`: `list_directory` on `path="src/rawnind/dependencies/configs/"` to verify YAMLs are loaded programmatically by the `ConfigManager` in the `dependencies` package, and that the `inference` package consumes them correctly.

### Diagnosis and Planning:

1.  Identify violations: Duplicates, naming inconsistencies, scattered SOTs, incomplete extractions, mocking in tests, non-pytest elements.
2.  Use `sequentialthinking` MCP if complex (e.g., `thought: "Analyze duplicate inference engines; next_thought_needed=true"`).
3.  Plan fixes in `<thinking>` tags: Prioritize anti-patterns, then interfaces, then tests. Propose rewrites; delete legacies.

### Fix Phase (TDD: Red → Green):

1.  **Red**: Review and iterate on the tests in `src/rawnind/inference/tests/`. Modify or extend them as needed to better diagnose issues. Use `insert_content` or `write_to_file` for modifications; run `execute_command "python -m pytest src/rawnind/inference/tests/ -v"` to confirm/adjust the red state.
2.  **Green**: Fix code:
    *   **Anti-patterns**: Use `rename_refactoring` (pycharm) for naming; `edit_file` (filesystem) or `replace_text_in_file` (pycharm) to consolidate/delete duplicates.
    *   **SOT**: Centralize in dataclasses; use `write_to_file` for full rewrites if needed (provide complete content, compute `line_count`).
    *   **No mocking**: Ensure integration tests use real pipeline; fix via `edit_file`.
    *   **CLI**: Remove any `argparse` remnants; delete files if safe.
3.  Re-run pytest after each fix; aim for 100% pass in `inference` tests.
4.  If blocked: `ask_followup_question` for clarification.

### Validation and Cleanup:

1.  Run full suite: `execute_command "python -m pytest src/rawnind/ -m inference --tb=short"` (expect all green).
2.  Cleanup: Delete unused legacies; reformat with `reformat_file` (pycharm).