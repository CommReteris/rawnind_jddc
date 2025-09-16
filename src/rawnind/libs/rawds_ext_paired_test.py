"""Lightweight shims for external paired test dataloaders used by integration scripts.

These classes are imported by several test scripts that are intended to be run
manually (guarded by `if __name__ == "__main__":`). To allow pytest collection
without requiring external datasets, we provide minimal, dependency‑free
implementations exposing the expected API surface.

If you actually want to run these integration scripts, replace these shims with
real implementations that read the external YAML configuration and construct
iterators over the dataset.
"""
from __future__ import annotations
from typing import Iterable, Iterator, List, Optional, Any


class _BaseExtTestDataloader:
    def __init__(self, content_fpaths: Optional[List[str]] = None, **_: Any) -> None:
        # Keep arguments for compatibility; not used in the stub.
        self.content_fpaths = content_fpaths or []

    def batched_iterator(self, batch_size: int = 1) -> Iterator[Any]:
        """Return an empty iterator to satisfy callers in test scripts.

        The integration tests that import these classes only run under __main__,
        so during pytest collection we just need a valid method. Returning an
        empty iterator ensures safety if called inadvertently.
        """
        if False:
            yield None  # pragma: no cover
        return iter(())


class CleanProfiledRGBNoisyBayerImageCropsExtTestDataloader(_BaseExtTestDataloader):
    pass


class CleanProfiledRGBNoisyProfiledRGBImageCropsExtTestDataloader(_BaseExtTestDataloader):
    pass
