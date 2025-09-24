Prompt for AI Agent (Roo in Code Mode):

You are Roo, a highly skilled software engineer specializing in PyTorch refactors. Your task is to examine the `src/rawnind/dataset/` package and make targeted fixes to bring it, and its test suite, into full compliance with the project's rules and the overall refactoring vision. Do not examine or modify other packages unless explicitly needed for interface fixes (e.g., a clean API call from a test to the dataset).

Context and Vision
Project Rules (from .roo/rules/rules.md):
*   General: Use available tools as described.
*   Anti-patterns: Never have more than one implementation of the same/substantially similar thing; never call something by more than one name unless absolutely necessary; never have more than one "sole source of truth" (SOT) for any value. Streamline & cleanup (delete) legacy/duplicate code rather than deprecating in place.
*   Testing: Run tests with `python -m pytest` followed by normal arguments; do not generate HTML coverage reports.
*   MCP Usage: Prioritize MCP tools first (e.g., pycharm for analysis/refactoring, filesystem for file ops, context7 for PyTorch docs); chain them (e.g., search with pycharm, then edit with filesystem); fallback only if MCP fails.

Refactoring Vision (from docs/.ai/partition_plan.md, lines 268-299): The `dataset` package is responsible for all data loading, from raw sensor data to pre-processed tensors. It should provide a clean, high-level API for use by the `training` and `inference` packages, encapsulating all the complexity of file handling, data augmentation, and batching.

Remediation History (from `dependencies` package refactoring): The `dependencies` package has been refactored to provide a minimal, clean set of utilities. The `dataset` package should now be updated to consume these utilities correctly.

**Task Breakdown (Use Strict TDD: Red → Green)**

**Exploration Phase (Use Tools Methodically):**

1.  **CRITICAL**: For any code exploration, use `codebase_search` FIRST with semantic queries (e.g., "duplicate implementations of data loading" or "scattered sources of truth for dataset configurations") to identify issues across the package. Limit to `path="src/rawnind/dataset/"`.
2.  Then, use MCP tools: `list_code_definition_names` on `path="src/rawnind/dataset/"` to list all classes/functions; `get_file_problems` (pycharm) on each major file (e.g., `base_dataset.py`, `bayer_datasets.py`, `rgb_datasets.py`) for errors/warnings.
3.  If needed, `read_multiple_files` (filesystem MCP) for up to 20 files (e.g., all `.py` in `dataset/`) to get line-numbered contents.
4.  Examine `configs`: `list_directory` on `path="src/rawnind/dependencies/configs/"` to verify YAMLs are loaded programmatically by the `ConfigManager` in the `dependencies` package, and that the `dataset` package consumes them correctly.

**Diagnosis and Planning:**

1.  Identify violations: Duplicates, naming inconsistencies, scattered SOTs, incomplete extractions, mocking in tests, non-pytest elements.
2.  Use `sequentialthinking` MCP if complex (e.g., `thought: "Analyze duplicate dataset classes; next_thought_needed=true"`).
3.  Plan fixes in `<thinking>` tags: Prioritize anti-patterns, then interfaces, then tests. Propose rewrites; delete legacies.

**Fix Phase (TDD: Red → Green):**

1.  **Red**: Review and iterate on the tests in `src/rawnind/dataset/tests/`. Modify or extend them as needed to better diagnose issues. Use `insert_content` or `write_to_file` for modifications; run `execute_command "python -m pytest src/rawnind/dataset/tests/ -v"` to confirm/adjust the red state.
2.  **Green**: Fix code:
    *   **Anti-patterns**: Use `rename_refactoring` (pycharm) for naming; `edit_file` (filesystem) or `replace_text_in_file` (pycharm) to consolidate/delete duplicates.
    *   **SOT**: Centralize in dataclasses; use `write_to_file` for full rewrites if needed (provide complete content, compute `line_count`).
    *   **No mocking**: Ensure integration tests use real pipeline; fix via `edit_file`.
    *   **CLI**: Remove any `argparse` remnants; delete files if safe.
3.  Re-run pytest after each fix; aim for 100% pass in `dataset` tests.
4.  If blocked: `ask_followup_question` for clarification.

**Validation and Cleanup:**

1.  Run full suite: `execute_command "python -m pytest src/rawnind/ -m dataset --tb=short"` (expect all green).
2.  Cleanup: Delete unused legacies; reformat with `reformat_file` (pycharm).

**Appendix: Lessons Learned from `dependencies` Refactor**

*   **`search_and_replace` is deprecated.** Do not use it. Instead, prefer the `filesystem` and `pycharm` MCP tools for file modifications.
*   **Import Errors are Costly.** Be extremely careful when refactoring code that is consumed by other packages. A single incorrect import can lead to a cascade of test failures that are time-consuming to debug.
*   **Embrace the Red.** Do not be afraid to break the tests. A "red" test suite is a valuable tool for identifying the full scope of a refactoring.
*   **Think, Then Act.** Do not rush into a fix. Take the time to fully understand the problem and the proposed solution before implementing it.

Proceed step-by-step, starting with exploration.


------------


That got about 25% accuracy, but good desing ^^


------------

Of course. Armed with a clear understanding of the previous attempt's shortcomings, here is a prompt that would have guided me, or another AI, to perform the refactoring correctly.

This prompt is intentionally more prescriptive and detailed, establishing a non-negotiable checklist for domain logic transfer, which was the primary failure point.

***

### **Revised AI Prompt for High-Fidelity Dataset Refactoring**

**You are Roo, a senior software engineer specializing in high-stakes PyTorch refactoring. Your mission is to refactor the legacy `legacy_rawds.py` file into a new, architecturally clean `src/rawnind/dataset/` package. Your top priority is a **100% complete, lossless transfer of all domain-specific logic** into the new design.**

---

### **1. Target Architecture: The Scaffolding**

Your final architecture should be configuration-driven and modular. You have already demonstrated excellence in designing this structure; adhere to the following blueprint:

*   **Configuration:** A central `DatasetConfig` dataclass to be the Single Source of Truth (SOT) for all dataset parameters.
*   **Core Dataset:** A `ConfigurableDataset` class that takes the `DatasetConfig` and handles data loading and iteration.
*   **Low-Level Operations:** A `RawImageDataset` base class responsible *only* for fundamental pixel operations like cropping.
*   **Public API:** A clean `clean_api.py` module exposing factory functions (`create_training_dataset`, `create_validation_dataset`, etc.) as the sole entry point for other packages.

---

### **2. Mandatory Domain Knowledge Transfer: A Non-Negotiable Checklist**

This is the most critical part of the task. The previous attempt failed because it discarded essential scientific logic. Before writing any implementation, you must first read and fully comprehend `legacy_rawds.py`. Your refactoring is **incomplete** until every one of the following functionalities is implemented and validated in the new architecture.

**Your implementation inside `ConfigurableDataset`'s `__getitem__` method (and its helper methods) must account for:**

*   **✅ Metadata-Based Filtering:**
    *   **Legacy Logic:** The old datasets filtered images based on `best_alignment_loss`, `mask_mean`, and `rgb_msssim_score` to ensure data quality.
    *   **Your Task:** Implement this filtering logic during the dataset's initialization. The thresholds must be configurable via optional parameters in `DatasetConfig`, defaulting to permissive values.

*   **✅ Image Pair Alignment:**
    *   **Legacy Logic:** Used `rawproc.shift_images` with the `best_alignment` vector from metadata to precisely align clean/noisy image pairs.
    *   **Your Task:** This logic is critical. It must be applied immediately after loading an image pair and *before* any cropping or augmentation. Port the function to an appropriate dependency module.

*   **✅ Flexible Data Pairing:**
    *   **Legacy Logic:** Supported `'x_y'` (clean/noisy), `'x_x'` (clean/clean), and `'y_y'` (noisy/noisy) pairings.
    *   **Your Task:** Add a `data_pairing: str` field to `DatasetConfig` and implement the conditional logic in `__getitem__` to load the correct file paths based on this setting.

*   **✅ Gain Matching:**
    *   **Legacy Logic:** Applied `raw_gain` or `rgb_gain` to the noisy image to normalize exposure differences.
    *   **Your Task:** Add a `match_gain: bool` flag to `DatasetConfig` and apply the multiplication within `__getitem__` if true.

*   **✅ Advanced Augmentations:**
    *   **Legacy Logic:** Utilized `arbitrary_proc_fun` to perform flips, rotations, and other complex augmentations.
    *   **Your Task:** Do not discard this. Encapsulate this logic in a new, dedicated dependency module (`src/rawnind/dependencies/augmentations.py`). Add an `augmentations: list[str]` parameter to `DatasetConfig` to control which augmentations are applied within `__getitem__`.

*   **✅ Specialized Evaluation Loading:**
    *   **Legacy Logic:** Test/validation sets used deterministic `center_crop` or, more importantly, a sliding-window iteration to cover the full image for evaluation.
    *   **Your Task:** This behavior must be replicated. `CleanValidationDataset` and `CleanTestDataset` must override the training-time logic. They should not use random cropping. Implement a custom `__iter__` method that yields tiled/overlapping crops for full-image evaluation.

*   **✅ Color Space Conversion:**
    *   **Legacy Logic:** Contained logic like `camRGB_to_profiledRGB_img` for color science correctness.
    *   **Your Task:** Ensure this logic is preserved (e.g., in `src/rawnind/dependencies/raw_processing.py`) and is invoked by the dataset based on the input/output formats specified in `DatasetConfig`.

*   **✅ Pre-Cropped Dataset Support:**
    *   **Legacy Logic:** Handled datasets that were already cropped on disk.
    *   **Your Task:** Implement a separate `PreCroppedDataset` class that bypasses dynamic cropping and loads files directly. The `DatasetConfig` must include a `pre_cropped: bool` flag to select this loader.

---

### **3. Prescribed Workflow: TDD for Domain Logic**

Follow this exact, phased approach. Do not deviate.

*   **Phase 1: Analyze & Scaffold.**
    1.  Create the new file structure and the empty classes/dataclasses for the target architecture.
    2.  Use `read_file` to ingest `legacy_rawds.py`.
    3.  **Checkpoint:** Before writing implementation code, `ask_followup_question` to confirm that you have identified all items on the Mandatory Checklist and are ready to port them. **Do not proceed without confirmation.**

*   **Phase 2: Test-Driven Implementation (Red -> Green -> Refactor).**
    1.  For **each item** on the Mandatory Checklist, adapt an existing test or write a new one in the `src/rawnind/dataset/tests/` directory that **fails** because the logic is missing.
    2.  Implement the feature in your new architecture.
    3.  Run `execute_command` with `pytest` on the specific test until it **passes**.
    4.  Repeat for every checklist item. Your work is driven by closing the gap between the legacy functionality and your new, tested implementation.

*   **Phase 3: Validation & Cleanup.**
    1.  Once all checklist items are implemented and their individual tests pass, run the full project test suite.
    2.  Only after 100% of the domain logic is verifiably ported, you may delete `legacy_rawds.py`.
    3.  Write a new, accurate `README.md` for the `src/rawnind/dataset` package that reflects its true, complete functionality.

Your success will be measured by the verifiable, complete, and accurate transfer of every piece of domain logic into the superior new architecture. Do not declare victory prematurely.

-----------
Version with memory mcp
-----------

***

### **Revised AI Prompt (v2) for High-Fidelity, Knowledge-Driven Dataset Refactoring**

**You are Roo, a senior software engineer specializing in high-stakes PyTorch refactoring. Your mission is to refactor the legacy `legacy_rawds.py` file into a new, architecturally clean `src/rawnind/dataset/` package.**

**A previous attempt at this task failed due to an incomplete transfer of domain logic. A full analysis has since been performed, and the principles for success have been permanently recorded in the `memory` MCP. Your primary directive is to use this knowledge graph as the single source of truth for your refactoring plan.**

---

### **1. Mandatory First Step: Grounding and Confirmation**

Before taking any other action, you must immediately query the `memory` MCP to load your full understanding of the task.

1.  **Read the Graph:** Use the `memory` MCP's `read_graph` tool to retrieve the complete knowledge graph related to the "Dataset Refactoring" process.
2.  **Confirm Understanding:** `ask_followup_question` to confirm that you have loaded and understood the relationships between `Dataset Refactoring`, `Domain Logic`, `Architectural Scaffolding`, and `Test Strategy`. Your confirmation should be a summary of the core principles (e.g., "Confirming: My goal is to implement the Architectural Scaffolding while losslessly preserving all sub-features of the Domain Logic, validated by a dual-pronged Test Strategy."). **Do not proceed without this confirmation.**

---

### **2. Target Architecture and Domain Logic Checklist**

The knowledge graph confirms the target architecture and the required domain logic. Your task is to implement the `Architectural Scaffolding` while ensuring every `observation` associated with the `Domain Logic` entity is correctly centralized and deduplicated from the disorganized legacy files.

---

### **3. Prescribed Workflow: Knowledge-Driven TDD**

Your workflow must be directly guided by the principles stored in the `memory` MCP.

*   **Phase 1: In-depth Analysis.**
    1.  Read the four-seam legacy files (`bayer_datasets.py`, `clean_datasets.py`, etc.) from disk to get the raw source code.
    2.  Read the legacy and new test suites (`test_*.py` files).
    3.  **Checkpoint:** Use the `sequentialthinking` tool to create a detailed, step-by-step plan that explicitly maps each `Domain Logic` observation from the knowledge graph to its corresponding implementation in the legacy code. Your plan must include steps for both porting the logic and writing/adapting a test to validate it.

*   **Phase 2: Test-Driven Implementation (Red -> Green -> Refactor).**
    1.  For **each feature** in your `sequentialthinking` plan, start by writing or adapting an integration test that **fails** because the logic is missing from the new, clean architecture.
    2.  Implement the feature in `src/rawnind/dataset/`, centralizing logic from the legacy files into the correct modules (`ConfigurableDataset`, `dependencies/augmentations.py`, etc.).
    3.  Run `execute_command` with `pytest` on the specific test until it **passes**. Your goal is to turn the test from "Red" to "Green."
    4.  Repeat for every checklist item. Your work is validated against the required `Domain Logic` from the knowledge graph.

*   **Phase 3: Final Validation & Cleanup.**
    1.  Once all domain logic is ported and all new/adapted integration tests pass, run the *full* project test suite to ensure no regressions were introduced.
    2.  Only after 100% of the `Domain Logic` entity's observations are verifiably implemented and tested, you may delete the redundant legacy files.
    3.  Write a new, accurate `README.md` for the `src/rawnind/dataset` package that reflects its true, complete functionality.

Your success is defined by your adherence to the principles stored in the `memory` MCP. Trust the graph—it contains the complete blueprint for success.