# test_legacy_cli_removed.py Passing Tests Documentation

## TestLegacyCliRemoved::test_no_legacy_libs_namespace

This test verifies that the refactored package does not expose the old `rawnind.libs` namespace or any of its submodules (`raw`, `rawds`, `rawproc`, `arbitrary_proc_fun`). It attempts to import these modules and expects `ModuleNotFoundError` to be raised, ensuring that legacy code paths have been properly removed from the package structure.

## TestLegacyCliRemoved::test_legacy_root_modules_not_under_package

This test ensures that legacy root-level modules (`legacy_raw`, `legacy_rawds`, `legacy_abstract_trainer`) are not accessible through the `rawnind` package namespace. These modules may exist in the repository for historical reference but must not be importable as part of the refactored package, maintaining clean separation between legacy and current code.

## TestLegacyCliRemoved::test_core_modules_have_no_main_blocks

This test verifies that core modules (`rawnind.training.training_loops`, `rawnind.inference.base_inference`) do not contain legacy `__main__` blocks or `main()` entry points. This ensures that CLI functionality is handled by higher-level tooling rather than implicit script execution within package modules, promoting proper API-driven design.

## TestLegacyCliRemoved::test_importing_core_modules_does_not_parse_cli

This test ensures that importing core modules is safe and does not trigger any command-line argument parsing or process exit. It simulates unknown CLI arguments and verifies that imports succeed without raising `SystemExit`, confirming that the modules are CLI-agnostic and can be safely imported in any context.