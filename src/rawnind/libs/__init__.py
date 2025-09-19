"""Internal library modules for RawNIND experiments.

This package collects utilities for raw I/O, dataset handling, processing helpers,
training abstractions, and small test shims.

Note: We intentionally avoid importing submodules at package import time to keep
`import .` lightweight and free of optional system dependencies.
Import submodules explicitly where needed, e.g.:
    from . import rawproc, raw
"""

__all__ = [
    "abstract_trainer",
    "arbitrary_proc_fun",
    "raw",
    "rawproc",
    "rawds",
    "rawds_manproc",
    "rawtestlib",
    "validation"
]
