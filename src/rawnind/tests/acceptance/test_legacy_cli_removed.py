import importlib
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
class TestLegacyCliRemoved:
    def test_no_legacy_libs_namespace(self):
        """
        The refactored package must not expose the old rawnind.libs namespace
        (e.g., rawnind.libs.raw, rawnind.libs.rawds, etc.). Attempting to
        import it should fail with ModuleNotFoundError.
        """
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("rawnind.libs")

        for sub in ["raw", "rawds", "rawproc", "arbitrary_proc_fun"]:
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(f"rawnind.libs.{sub}")

    def test_legacy_root_modules_not_under_package(self):
        """
        Legacy root-level modules must not be packaged under rawnind.*. They may
        exist in the repository root for historical reference, but cannot be
        importable via the rawnind package namespace.
        """
        for mod in [
            "legacy_raw",
            "legacy_rawds",
            "legacy_abstract_trainer",
        ]:
            with pytest.raises(ModuleNotFoundError):
                importlib.import_module(f"rawnind.{mod}")

    def test_core_modules_have_no_main_blocks(self):
        """
        Ensure core modules do not contain legacy __main__ blocks or main()
        entry points. CLI should be driven by higher-level tooling or explicit
        APIs, not implicit script execution within package modules.
        """
        core_modules = [
            "rawnind.training.training_loops",
            "rawnind.inference.base_inference",
        ]
        for modname in core_modules:
            mod = importlib.import_module(modname)
            src = Path(mod.__file__).read_text(encoding="utf-8")
            assert "__main__" not in src, f"Legacy __main__ block found in {modname}"
            assert "def main(" not in src, f"Legacy main() entry-point found in {modname}"

    def test_importing_core_modules_does_not_parse_cli(self, monkeypatch):
        """
        Importing core modules must be safe and must not trigger any argument
        parsing or process exit on import. We simulate unknown CLI args and
        expect the import to succeed without raising SystemExit.
        """
        monkeypatch.setattr(sys, "argv", ["prog", "--unknown-arg-for-test"])  # noisy argv
        for mod in [
            "rawnind.training.training_loops",
            "rawnind.inference.base_inference",
        ]:
            # Reload fresh to ensure import-time behavior is exercised
            sys.modules.pop(mod, None)
            importlib.invalidate_caches()
            importlib.import_module(mod)

    def test_pyproject_has_no_legacy_console_scripts(self):
        """
        The package metadata must not declare legacy console scripts pointing to
        removed namespaces. This guards against accidentally re-exposing legacy
        CLIs via packaging.
        """
        # Locate pyproject.toml at repo root
        repo_root = Path(__file__).resolve().parents[3]
        pyproject = repo_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found at project root"

        text = pyproject.read_text(encoding="utf-8")
        # No legacy scripts, and nothing referencing rawnind.libs or legacy_* modules anywhere in the file
        banned_substrings = ["rawnind.libs", "legacy_raw", "legacy_rawds", "legacy_abstract_trainer"]
        for banned in banned_substrings:
            assert banned not in text, f"pyproject.toml contains legacy reference: {banned}"
