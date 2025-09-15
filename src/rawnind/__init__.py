"""RawNIND package.

Lightweight package initializer to avoid importing heavyweight subpackages
(e.g., training stacks that depend on optional system libraries) at import time.

Subpackages:
- rawnind.libs: Core libraries for raw I/O, datasets, processing, and trainers.
- rawnind.tools: Command-line utilities and dataset preparation scripts.
- rawnind.models: Model definitions used in experiments.

Import submodules explicitly where needed, e.g.:
    from rawnind.libs import rawproc
    from rawnind.tools import crop_datasets
"""

# Intentionally avoid eager imports like `from . import libs, tools, models`
# to keep `import rawnind` cheap and not require optional runtime deps.

__all__ = ["libs", "tools", "models"]
