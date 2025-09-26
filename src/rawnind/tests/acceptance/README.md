# Acceptance Test Suite (Partition Refactor Guardrails)

This directory contains acceptance tests that act as guardrails for the ongoing refactor/partition of the rawnind
codebase.
They verify high‑level behavior and public interfaces across the new packages (training, inference, dataset,
dependencies) without
requiring large datasets or GPUs. The suite is intentionally lightweight and focuses on contracts rather than deep
internals.

Raison d’être (why this suite exists)

- Provide early warning if refactoring breaks public interfaces or cross‑package wiring.
- Lock in minimal contracts for the newly extracted packages so work can proceed in small, safe slices.
<<<<<<< HEAD
- Enable continuous progress tracking via xfail/skip markers for pieces not yet migrated.
=======
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
- Validate critical runtime paths with tiny, CPU‑only stubs/fakes to avoid heavy dependencies.
- Offer executable documentation for the partition plan (what exists now vs. what is still in flight).

Scope covered now

- Inference: InferenceEngine runs on small tensors and returns expected shapes/modes.
- Training: Presence of TrainingLoops and derived classes; a prepared smoke test (skipped) for when constructor APIs
  stabilize.
- Dataset: Base RawImageDataset utilities (random/center crops) conform to shape contracts.
- Dependencies: YAML round‑trip and experiment utilities operate over a temp directory.
- Imports/layout: Public symbols exist under the new package namespaces.

<<<<<<< HEAD
Intentional xfail/skip markers

- tests that reference functionality not yet extracted (e.g., specific dataset classes) are xfail.
- the training smoke test is marked skip until the external construction API is stabilized.
  These markers should be flipped to passing assertions as each refactoring milestone lands.
=======
Minimum standards and policy

- No XFAILs or SKIPs allowed in the acceptance suite. Acceptance is a release gate; tests must pass.
- Prefer tiny, deterministic inputs; avoid filesystem/network, except where explicitly tested.
- Keep public symbol names stable; tests assert their presence by import path and attribute name.

Rationale: Acceptance tests are guardrails for refactors and releases. Allowing XFAILs or SKIPs lowers the bar and hides
regressions; instead, track gaps with issues. The acceptance suite must stay green.
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

Directory contents

- test_imports_and_layout.py — asserts public symbols exist in new modules.
<<<<<<< HEAD
- test_inference_engine.py — smoke tests for InferenceEngine (CPU‑only); xfail for transfer‑function dependency until
  available.
- test_training_loops_smoke.py — prepared minimal training step test (currently skipped pending API stabilization).
- test_dataset_base_contracts.py — verifies cropping utilities and establishes dataset output shape contracts; xfail
=======
- test_inference_engine.py — smoke tests for InferenceEngine (CPU‑only), including transfer‑function availability and
  behavior.
- test_training_loops_smoke.py — lightweight training loops contract check (kept minimal and CPU‑only).
- test_dataset_base_contracts.py — verifies cropping utilities and establishes dataset output shape contracts
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
  placeholder for future datasets.
- test_dependencies_and_experiments.py — YAML helpers and ExperimentManager filesystem behaviors.

Prerequisites

- Python 3.9+ recommended
- Install dependencies from repository root (choose one):
    - pip install -r requirements.txt
    - Or with uv: uv pip install -r requirements.txt

How to run (from repository root)
Windows PowerShell

- Run entire acceptance suite:
  pytest -q src\rawnind\tests\acceptance
<<<<<<< HEAD
- Include reasons for skips/xfails (useful during migration):
  pytest -q -rxXs src\rawnind\tests\acceptance
=======
- With detailed summary:
  pytest -q -rA src\rawnind\tests\acceptance
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
- Run a single file:
  pytest -q src\rawnind\tests\acceptance\test_inference_engine.py
- Run a single test by node id:
  pytest -q src\rawnind\tests\acceptance\test_inference_engine.py::test_infer_output_modes

Unix/macOS shell

- Run entire suite:
  pytest -q src/rawnind/tests/acceptance
<<<<<<< HEAD
- With reasons for skips/xfails:
  pytest -q -rxXs src/rawnind/tests/acceptance
=======
- With detailed summary:
  pytest -q -rA src/rawnind/tests/acceptance
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
- Single file / single test:
  pytest -q src/rawnind/tests/acceptance/test_inference_engine.py
  pytest -q src/rawnind/tests/acceptance/test_inference_engine.py::test_infer_output_modes

Useful pytest flags

- -q: quiet output
<<<<<<< HEAD
- -rxXs: show reasons for xfail/xfail/skips
=======
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
- -k <expr>: run tests matching expression (e.g., -k inference)
- -m acceptance: select tests with the acceptance marker (these tests set pytestmark = pytest.mark.acceptance)

Troubleshooting

- ImportError or ModuleNotFoundError: ensure you run pytest from the repository root so package imports resolve.
- CUDA/GPU not required: tests are CPU‑only and use tiny tensors.
<<<<<<< HEAD
- Expected failures: some tests are xfail by design until the corresponding refactor parts are implemented. They should
  appear as XFAIL, not FAIL.
- Windows paths: use backslashes in PowerShell examples above; forward slashes on Unix/macOS.

Maintaining and evolving this suite

- When you extract a new dataset class (e.g., CleanCleanImageDataset), convert its xfail placeholder into a real test
  and remove the xfail.
- When InferenceEngine gains transfer‑function dependencies under dependencies.raw_processing, un‑xfail the related test
  and add a couple of simple numeric assertions.
- As the ImageToImageNNTraining constructor stabilizes, remove the skip from the training smoke test and assert basic
  forward/backward works with a tiny model/dataset.
- Keep public symbol names stable; tests assert their presence by import path and attribute name.
- Prefer tiny, deterministic inputs; avoid filesystem/network, except where explicitly tested (e.g., ExperimentManager
  helpers).

CI integration tips

- Add a job that runs: pytest -q -rxXs src/rawnind/tests/acceptance
- Treat unexpected FAIL as a blocker; XFAILs are acceptable until their features are implemented, but avoid adding new
  XFAILs without issue tracking.

Last updated: 2025-09-20
=======

- Windows paths: use backslashes in PowerShell examples above; forward slashes on Unix/macOS.
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c
