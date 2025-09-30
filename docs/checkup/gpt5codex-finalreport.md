Consolidated audit and action plan (legacy_*.py excluded)

1. Pipeline operations package (`src/rawnind/operations/*.py`)
   • Emulator: Files define `Encoder`, `RawLoader`, etc., but no pipeline spec references them and production code never imports them.
   • Registry drift: `OperationRegistry` registers implementations in the non-existent module `src.rawnind.operations.model_operations.EncoderOperation`.
   • Action: Either delete these stubs or replace them with real wrappers whose paths match the registry when the pipeline is actually wired in.

2. Dataset clean API (`src/rawnind/dataset/clean_api.py` and `__init__.py`)
   • Current behaviour: `ConfigurableDataset` fabricates tensors with `torch.randn`, returning synthetic crops and metadata, then exports these through `CleanDataset`.
   • Duplication: Original loaders live in `base_dataset.py`, `bayer_datasets.py`, etc., and still back the training code. The clean layer duplicates logic but doesn’t access real image data.
   • Action: Rework `CleanDataset` to wrap the existing dataset modules (or otherwise load real crops). Until that happens, treat the clean dataset factories as placeholders and avoid exposing them as a production API.

3. Dependencies package (`src/rawnind/dependencies/*`)
   • Usage: These modules (raw_processing, numpy_operations, json_saver, etc.) remain the real implementation backing training/inference. No dead code detected.
   • Recommendation: When refactoring the clean pipeline, call into these modules instead of duplicating functionality in the operations stubs.

4. Registry/config duplication
   • Architecture maps (e.g., encoder/decoder choices) are hard-coded separately in inference and training modules and also partially in OperationRegistry specs.
   • Action: One source of truth—either a shared registry or a refactored clean pipeline—should own these mappings to prevent config drift.

Next steps for a coherent refactor:
A. Decide whether the function-based pipeline (PipelineOperation/OperationRegistry) will be adopted in production.
   – If yes, implement real wrappers using the dependencies and dataset modules, fix the registry paths, and integrate the assembler into inference/training entry points.
   – If no, remove the unused operations package and registry specs to reduce confusion.

B. Rework the clean dataset API so it loads real data via existing dataset modules. Once that is done, re-enable tests that rely on the clean API.

C. Consolidate architecture mappings and config adapters in one place.

This plan merges the findings across previous passes into a single actionable roadmap focusing on real data flow, removal of stubs, and deduplicated configuration.