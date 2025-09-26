### Corrected Architectural Understanding

You're absolutely right about the fundamental organizational issue. I was mixing up conceptual relationships with
logical packaging boundaries.

#### Key Insight

**Architecture should reflect what gets imported and used together, not abstract conceptual relationships.**

#### Correct Organization

**Core Modules (Actual Packages):**

- **Single Operation Registry** - All operations grouped by functional intent (denoising, enhancement, geometric)
  regardless of implementation backend
- **Pipeline Infrastructure** - `PipelineOperation`, `OperationPipeline`, `SmartPipelineAssembler`
- **Registry Extensions** - `ModelRegistry`, `QualityChecksRegistry`, `PreprocessingRegistry`

**Implementation Details (Not Architectural Components):**

- **Kornia/OpenCV/NumPy** - Backend libraries used inside operation wrappers
- **PyTorch Lightning** - Framework pattern affecting training code structure
- **Hydra** - Configuration loading patterns

#### The Universal Interface Principle

A `SharpenWrapper` belongs in `enhancement_operations` whether it uses Kornia, OpenCV, or custom code internally -
that's just an implementation detail hidden behind the universal `PipelineOperation` interface.

#### Why This Matters

- **Import clarity** - You import functional categories, not implementation backends
- **Backend flexibility** - Can switch from OpenCV to Kornia without changing architecture
- **Framework agnostic** - Lightning/Hydra influence coding style but don't create new modules

The architecture should serve actual usage patterns: developers want to find "enhancement operations" not "Kornia
operations."