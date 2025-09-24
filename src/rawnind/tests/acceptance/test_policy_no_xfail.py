import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.acceptance


def test_no_xfail_markers_in_acceptance_suite():
    """
    Enforce 0-XFAIL policy for acceptance tests: no test files in this directory
    may contain pytest xfail markers or calls. This prevents hiding regressions
    behind expected-fail annotations.
    """
    acceptance_dir = Path(__file__).parent
    patterns = [
        r"pytest\.mark\.xfail",
        r"@pytest\.mark\.xfail",
        r"pytest\.xfail\("
    ]
    regexes = [re.compile(p) for p in patterns]

    offenders = []
    for f in acceptance_dir.glob("test_*.py"):
        if f.name == Path(__file__).name:
            continue
        text = f.read_text(encoding="utf-8")
        for rx in regexes:
            if rx.search(text):
                offenders.append((str(f), rx.pattern))
                break

    assert not offenders, (
        "Found forbidden xfail markers/calls in acceptance tests: "
        + ", ".join(f"{path} ({pat})" for path, pat in offenders)
    )
