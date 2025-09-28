import importlib
import sys
from pathlib import Path

import pytest


@pytest.mark.acceptance
class TestLegacyCliRemoved:
    """Test suite verifying complete removal of legacy CLI components.

    This class contains acceptance tests ensuring that the refactored package
    has successfully eliminated all legacy CLI dependencies and structures.
    It validates the clean API design by confirming that old entry points,
    namespaces, and implicit CLI parsing are no longer accessible or functional.
    """

    def test_no_legacy_libs_namespace(self):
        """Test that legacy libs namespace is completely removed.

        This test verifies that the refactored package structure no longer exposes
        the old rawnind.libs namespace, which contained legacy implementations with
        CLI dependencies. Attempting to import it should fail immediately, ensuring
        that downstream code cannot accidentally use deprecated components.

        Expected behavior:
        - Direct import of "rawnind.libs" raises ModuleNotFoundError
        - Submodule imports like "rawnind.libs.raw" also fail with ModuleNotFoundError
        - No partial or shadowed access to legacy modules remains
        - Import errors are clean and immediate, without side effects

        Key assertions:
        - importlib.import_module("rawnind.libs") raises ModuleNotFoundError
        - All known legacy submodules raise ModuleNotFoundError
        - No exceptions or warnings during failed imports
        """
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
        """Test that legacy root modules are not importable via package namespace.

        This test ensures that legacy script modules remain outside the main
        rawnind package namespace, preventing accidental imports and maintaining
        the clean separation between historical code and the refactored production
        API. Legacy files may exist for reference but should not interfere with
        package imports.

        Expected behavior:
        - Attempts to import legacy modules via "rawnind.legacy_*" fail
        - ModuleNotFoundError is raised for each legacy module
        - No partial loading or shadowing of legacy code occurs
        - Package integrity is preserved without legacy contamination

        Key assertions:
        - importlib.import_module("rawnind.legacy_raw") raises ModuleNotFoundError
        - All specified legacy modules are inaccessible via package
        - Import failures are consistent and immediate
        """
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
        """Test that core modules contain no legacy main blocks or entry points.

        This test scans source code of critical core modules to ensure they do not
        contain legacy __main__ blocks or main() functions that could trigger
        implicit CLI execution during imports. It enforces the clean API principle
        by requiring explicit invocation rather than script-like behavior in modules.

        Expected behavior:
        - Core module source code lacks "__main__" guards
        - No "def main(" function definitions present
        - Module imports do not execute side-effect code
        - Source scanning completes without parsing errors

        Key assertions:
        - "__main__" string absent from module source
        - "def main(" pattern not found in source code
        - Specific error messages identify problematic modules
        - All core modules pass the clean code check
        """
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
