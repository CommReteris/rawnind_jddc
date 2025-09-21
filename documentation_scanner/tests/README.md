# Documentation Scanner Tests

This directory contains unit and integration tests for the documentation_scanner.py script.

## Overview

The tests cover all major components and functionality of the documentation scanner, including:

- AST parsing and docstring detection
- File scanning and analysis
- Directory traversal (with and without recursion)
- Documentation percentage calculations
- Report generation
- Command-line argument parsing
- Edge case handling

## Running the Tests

### Prerequisites

The tests require the following packages:

- pytest
- pytest-trio (for testing async functions)
- pytest-cov (optional, for test coverage reporting)

You can install these with pip:

```bash
pip install pytest pytest-trio pytest-cov
```

### Basic Test Execution

To run all tests:

```bash
# From the project root directory
python -m pytest tests/test_documentation_scanner.py -v
```

### Coverage Testing

To run tests with coverage reporting:

```bash
# From the project root directory
python -m pytest tests/test_documentation_scanner.py --cov=src.documentation_scanner --cov-report=term
```

For a detailed HTML coverage report:

```bash
python -m pytest tests/test_documentation_scanner.py --cov=src.documentation_scanner --cov-report=html
```

## Test Structure

The test suite is organized as follows:

1. **Test Fixtures**: Setup code that creates sample Python files with varying levels of documentation for testing.

2. **DocstringVisitor Tests**: Tests for the AST visitor class that identifies docstrings in modules, classes,
   functions, and methods.

3. **scan_file Tests**: Tests for the function that analyzes individual Python files.

4. **Documentation Line Counting Tests**: Tests for counting lines of documentation and code.

5. **Report Generation Tests**: Tests for generating Markdown reports.

6. **Directory Scanning Tests**: Tests for recursive and non-recursive directory scanning.

7. **Command-line Argument Tests**: Tests for the command-line interface and argument parsing.

8. **Edge Case Tests**: Tests for empty directories, non-Python files, and error conditions.

## Asynchronous Testing

Since the documentation scanner uses the Trio library for asynchronous I/O, the tests use pytest-trio to properly test
asynchronous functions. Each test for an async function is marked with the `@pytest.mark.trio` decorator.

## Mocking

For tests that would otherwise perform file system operations or have external dependencies, unittest.mock is used to
isolate the test from these dependencies. This makes the tests more reliable and faster to execute.

## Expected Output

When all tests pass, you should see output similar to:

```
============================= test session starts ==============================
collected 25 tests

tests/test_documentation_scanner.py::test_docstring_visitor_module PASSED
tests/test_documentation_scanner.py::test_docstring_visitor_no_module_docstring PASSED
... (more test results) ...
tests/test_documentation_scanner.py::test_scan_file_read_error PASSED

============================== 25 passed in 3.21s ==============================
```