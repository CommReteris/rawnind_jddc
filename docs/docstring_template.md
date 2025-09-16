# Google-Style Docstring Templates

## Method Docstring Template

```python
def method_name(self, param1, param2, optional_param=None):
    """Short, imperative description of what the method does.
    
    More detailed explanation of the method's purpose, behavior, and any 
    important implementation details. Explain the overall flow and logic
    of the method, especially for complex methods. Include background
    information only if directly relevant to understanding the method.
    
    Args:
        param1: Description of param1. Include type if not obvious from
            the type annotation. Use complete sentences.
        param2: Description of param2. For parameters with default values,
            mention how the default behavior works.
        optional_param: Description of optional_param. For boolean flags,
            explain what True and False mean.
            
    Returns:
        Description of the return value(s). Be specific about the return type
        and format. If returning None, state that explicitly.
        
    Raises:
        ExceptionType: Explanation of when and why this exception is raised.
        AnotherException: Description of another possible exception.
        
    Notes:
        Additional implementation notes, constraints, assumptions, or other
        details that help understand the method's behavior. This section
        is especially useful for complex algorithms.
        
    Examples:
        Basic usage examples when helpful (optional):
        
        ```python
        result = instance.method_name(param1_value, param2_value)
        ```
    """
    # Method implementation
```

## Class Docstring Template

```python
class ClassName:
    """Short description of the class's purpose and responsibility.
    
    More detailed explanation of the class's role in the system, design
    principles, usage patterns, and important behavior. Explain the class's
    main interfaces and how it interacts with other components.
    
    Attributes:
        attr1: Description of attr1. Include type if not obvious.
        attr2: Description of attr2.
        
    Note:
        Important notes about the class implementation, constraints,
        thread-safety considerations, etc.
    """

    def __init__(self, param1, param2=None):
        """Initialize the class instance.
        
        Args:
            param1: Description of initialization parameter.
            param2: Description of optional initialization parameter.
            
        Raises:
            ValueError: If param1 doesn't meet requirements.
        """
        # Implementation
```

## Function Docstring Template

```python
def function_name(param1, param2, optional_param=None):
    """Short, imperative description of what the function does.
    
    More detailed explanation of the function's purpose, behavior, and any 
    important implementation details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
        optional_param: Description of optional_param.
            
    Returns:
        Description of the return value(s).
        
    Raises:
        ExceptionType: Explanation of when this exception is raised.
        
    Examples:
        ```python
        result = function_name('value', 42)
        ```
    """
    # Function implementation
```

## Module-Level Docstring Template

```python
"""Module name and short description.

Detailed description of the module's purpose, contents, and usage.
Explain the key abstractions and components provided by the module.

Key classes:
- ClassName1: Short description
- ClassName2: Short description

Key functions:
- function_name1: Short description
- function_name2: Short description

Typical usage:
```python
from module import ClassName1

instance = ClassName1()
result = instance.method()
```

Notes:
Important module-level notes, constraints, or dependencies.
"""

```

## Example for validate_or_test Method

```python
def validate_or_test(
    self,
    dataloader: Iterable,
    test_name: str,
    sanity_check: bool = False,
    save_individual_results: bool = True,
    save_individual_images: bool = False,
):
    """Perform validation or testing on a dataset.
    
    This method runs the model on all samples from the provided dataloader,
    calculates performance metrics, and optionally saves individual results
    and output images. It supports distributed evaluation with a locking
    mechanism to prevent resource conflicts.
    
    The method performs these steps:
    1. Establishes a validation lock if needed to prevent parallel evaluations
    2. Processes each batch from the dataloader with the model
    3. Calculates losses and metrics for each sample
    4. Aggregates results and computes statistics
    5. Saves individual sample results and/or images if requested
    
    Args:
        dataloader: Iterable that yields batches of data, one image at a time
        test_name: Identifier for this validation/test run, used for file naming
        sanity_check: If True, runs a minimal validation for debugging purposes
        save_individual_results: If True, saves per-sample metrics to a YAML file
        save_individual_images: If True, saves model output images for each sample
            
    Returns:
        Dictionary with aggregated metrics and loss values
        
    Notes:
        - This method expects the dataloader to return one sample at a time
        - For progressive validation/testing, results from previous runs may be loaded
        - The validation lock prevents multiple processes from running CPU-intensive
          operations simultaneously on shared resources
    """
    # Method implementation
```