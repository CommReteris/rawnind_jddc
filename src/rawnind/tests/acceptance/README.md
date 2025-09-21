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
- Enable continuous progress tracking via xfail/skip markers for pieces not yet migrated.
- Validate critical runtime paths with tiny, CPU‑only stubs/fakes to avoid heavy dependencies.
- Offer executable documentation for the partition plan (what exists now vs. what is still in flight).

Scope covered now

- Inference: InferenceEngine runs on small tensors and returns expected shapes/modes.
- Training: Presence of TrainingLoops and derived classes; a prepared smoke test (skipped) for when constructor APIs
  stabilize.
- Dataset: Base RawImageDataset utilities (random/center crops) conform to shape contracts.
- Dependencies: YAML round‑trip and experiment utilities operate over a temp directory.
- Imports/layout: Public symbols exist under the new package namespaces.

Intentional xfail/skip markers

- tests that reference functionality not yet extracted (e.g., specific dataset classes) are xfail.
- the training smoke test is marked skip until the external construction API is stabilized.
  These markers should be flipped to passing assertions as each refactoring milestone lands.

Directory contents

- test_imports_and_layout.py — asserts public symbols exist in new modules.
- test_inference_engine.py — smoke tests for InferenceEngine (CPU‑only); xfail for transfer‑function dependency until
  available.
- test_training_loops_smoke.py — prepared minimal training step test (currently skipped pending API stabilization).
- test_dataset_base_contracts.py — verifies cropping utilities and establishes dataset output shape contracts; xfail
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
- Include reasons for skips/xfails (useful during migration):
  pytest -q -rxXs src\rawnind\tests\acceptance
- Run a single file:
  pytest -q src\rawnind\tests\acceptance\test_inference_engine.py
- Run a single test by node id:
  pytest -q src\rawnind\tests\acceptance\test_inference_engine.py::test_infer_output_modes

Unix/macOS shell

- Run entire suite:
  pytest -q src/rawnind/tests/acceptance
- With reasons for skips/xfails:
  pytest -q -rxXs src/rawnind/tests/acceptance
- Single file / single test:
  pytest -q src/rawnind/tests/acceptance/test_inference_engine.py
  pytest -q src/rawnind/tests/acceptance/test_inference_engine.py::test_infer_output_modes

Useful pytest flags

- -q: quiet output
- -rxXs: show reasons for xfail/xfail/skips
- -k <expr>: run tests matching expression (e.g., -k inference)
- -m acceptance: select tests with the acceptance marker (these tests set pytestmark = pytest.mark.acceptance)

Troubleshooting

- ImportError or ModuleNotFoundError: ensure you run pytest from the repository root so package imports resolve.
- CUDA/GPU not required: tests are CPU‑only and use tiny tensors.
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
