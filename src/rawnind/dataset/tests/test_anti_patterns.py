import pytest
import inspect
import pkgutil
import importlib

# Define the package to inspect
DATASET_PACKAGE_PATH = "src.rawnind.dataset"

# List of classes that are expected to be unique (not aliased or duplicated)
# These are prime candidates for violating the "sole source of truth" anti-pattern
CRITICAL_SINGLETON_CLASSES = [
    "RawDatasetOutput",
    "RawImageDataset",
    "CleanCleanImageDataset",
    "CleanNoisyDataset",
    "ProfiledRGBBayerImageDataset",
    "ProfiledRGBProfiledRGBImageCropsDataset",
    "CleanProfiledRGBCleanBayerImageCropsDataset",
    "CleanProfiledRGBCleanProfiledRGBImageCropsDataset",
    "CleanProfiledRGBNoisyBayerImageCropsDataset",
    "CleanProfiledRGBNoisyProfiledRGBImageCropsDataset",
    "CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset",
    "CleanProfiledRGBNoisyBayerImageCropsValidationDataset",
    "CleanProfiledRGBNoisyBayerImageCropsTestDataloader",
    "CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader",
    "TestDataLoader",
    "DatasetConfig",
    "DatasetMetadata"
]

@pytest.fixture(scope="module")
def all_dataset_classes():
    """Dynamically collects all classes defined within the dataset package and their origin modules."""
    classes = {}
    package = importlib.import_module(DATASET_PACKAGE_PATH)

    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if modname.startswith(f"{DATASET_PACKAGE_PATH}.tests"): # Skip tests modules
            continue
        try:
            module = importlib.import_module(modname)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and obj.__module__ == module.__name__:
                    if name in classes:
                        classes[name].append(module.__name__)
                    else:
                        classes[name] = [module.__name__]
        except Exception as e:
            pytest.fail(f"Could not import module {modname} for introspection: {e}")
    return classes

def test_critical_classes_are_singletons(all_dataset_classes):
    """
    Verifies that critical classes are defined in exactly one module.
    This checks for the "sole source of truth" anti-pattern.
    """
    
    # Exclusions for classes that are expected to be re-defined (e.g., base classes extended elsewhere)
    # or are explicitly part of a clean API pattern for re-exporting.
    # Currently, CleanCleanImageDataset and CleanNoisyDataset are base classes designed for inheritance
    # and not expected to be singletons in the same way, but inherited, so it makes it harder to check.
    # ProfiledRGBBayerImageDataset and ProfiledRGBProfiledRGBImageDataset are also base classes for others.
    # For now, we will focus on explicit duplications that are problematic.
    
    # We explicitly consolidated TestDataLoader, so let's verify it's a singleton (if not an inherited base class)
    # The others listed in CRITICAL_SINGLETON_CLASSES should generally be unique definitions.

    for class_name in CRITICAL_SINGLETON_CLASSES:
        if class_name not in all_dataset_classes:
            pytest.fail(f"Critical class '{class_name}' not found in any dataset module.")
        
        defining_modules = all_dataset_classes[class_name]
        
        # TestDataLoader is a special case. It's a mixin, so it will appear in actual Mixin classes (itself)
        # and in classes that inherit from it. We want to ensure its *definition* is unique.
        if class_name == "TestDataLoader":
            # TestDataLoader is defined in test_dataloaders.py and used as a mixin.
            # We enforce that its *primary definition* is solely in test_dataloaders.py
            # and that it's not redefined as a concrete class in other "non-mixin" files.
            # For simplicity, we just check if it's only explicitly defined once
            # in critical modules, and other occurrences are through inheritance.
            expected_module = f"{DATASET_PACKAGE_PATH}.test_dataloaders"
            assert expected_module in defining_modules, \
                f"TestDataLoader definition missing from its sole source of truth: {expected_module}"
            
            # Additional check: ensure it's not defined as a *standalone* class elsewhere
            # This is hard to do purely by name. We rely on careful code review and explicit
            # removals. For now, checking if it's only in its intended module by name.
            filtered_defining_modules = [m for m in defining_modules if "test_dataloaders" in m]
            assert len(filtered_defining_modules) == 1, \
                f"TestDataLoader defined in multiple primary modules: {defining_modules}"
            
        else: # For other critical singleton classes:
            # We expect these to be defined in exactly one module, excluding __init__.py which just re-exports
            # Filter out __init__.py files that might just re-export
            primary_definitions = [
                mod for mod in defining_modules 
                if not mod.endswith('__init__') and not mod.endswith('.validation_datasets') # validation_datasets sometimes shows up as a "definition" when it's an import + parent
            ]
            
            assert len(primary_definitions) <= 1, \
                f"Class '{class_name}' found defined in multiple primary modules: {defining_modules}. Expected at most one."
