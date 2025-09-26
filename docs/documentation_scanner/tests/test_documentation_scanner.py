#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unit tests for the documentation_scanner module.

This module contains unit tests for the documentation_scanner.py script, testing
all major components and functionality.
"""

import ast
import os
import sys
import tempfile

import pytest

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from docs.documentation_scanner import (
    DocstringVisitor,
    scan_file,
    scan_directory,
    count_doc_and_code_lines,
    generate_report,
    count_py_files,
    async_main,
    write_report
)
import argparse
from unittest.mock import patch, MagicMock


# Test fixtures for sample Python files
@pytest.fixture
def well_documented_file():
    """Create a temporary file with comprehensive docstrings for testing."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write('''"""This is a well-documented module.

This module demonstrates proper documentation practices with docstrings
for the module, classes, functions, and methods.
"""

class WellDocumentedClass:
    """A well-documented class.
    
    This class demonstrates proper class documentation.
    
    Attributes:
        value: A sample attribute
    """
    
    def __init__(self, value):
        """Initialize the class.
        
        Args:
            value: The initial value
        """
        self.value = value
    
    def documented_method(self, x):
        """A documented method.
        
        Args:
            x: An input parameter
            
        Returns:
            The result of the operation
        """
        return self.value + x
    
    def __repr__(self):
        # Special methods don't require docstrings
        return f"WellDocumentedClass({self.value})"


def documented_function(a, b):
    """A documented function.
    
    Args:
        a: First parameter
        b: Second parameter
        
    Returns:
        Sum of a and b
    """
    return a + b

''')
    filename = f.name
    yield filename
    os.unlink(filename)


@pytest.fixture
def partially_documented_file():
    """Create a temporary file with partial docstrings for testing."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write('''"""This is a partially documented module."""

class PartiallyDocumentedClass:
    """A documented class."""
    
    def __init__(self, value):
        # Missing docstring
        self.value = value
    
    def documented_method(self, x):
        """A documented method."""
        return self.value + x
        
    def undocumented_method(self, x):
        # Missing docstring
        return self.value * x


def undocumented_function(a, b):
    # Missing docstring
    return a - b

''')
    filename = f.name
    yield filename
    os.unlink(filename)


@pytest.fixture
def undocumented_file():
    """Create a temporary file with no docstrings for testing."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write('''
class UndocumentedClass:
    def __init__(self, value):
        self.value = value
    
    def undocumented_method(self, x):
        return self.value + x


def undocumented_function(a, b):
    return a * b

''')
    filename = f.name
    yield filename
    os.unlink(filename)


@pytest.fixture
def syntax_error_file():
    """Create a temporary file with a syntax error for testing."""
    with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as f:
        f.write('''
def function_with_syntax_error(
    # Missing closing parenthesis
    return "error"
''')
    filename = f.name
    yield filename
    os.unlink(filename)


@pytest.fixture
def test_directory():
    """Create a temporary directory with multiple Python files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple directory structure
        os.makedirs(os.path.join(temp_dir, 'package1'))
        os.makedirs(os.path.join(temp_dir, 'package2'))
        os.makedirs(os.path.join(temp_dir, 'package1', 'subpackage'))
        os.makedirs(os.path.join(temp_dir, '__pycache__'))  # Should be excluded

        # Create Python files with varying documentation levels
        files = {
            'root_file.py'                   : '''"""Root module docstring."""
def root_function(): pass
''',
            'package1/__init__.py'           : '''"""Package 1 initialization."""
''',
            'package1/module1.py'            : '''"""Module 1 docstring."""
class Class1:
    """Class 1 docstring."""
    def method1(self): 
        """Method 1 docstring."""
        pass
    def method2(self): pass
''',
            'package1/subpackage/__init__.py': '',
            'package1/subpackage/module2.py' : '''
class Class2:
    def method1(self): pass
''',
            'package2/__init__.py'           : '',
            'package2/module3.py'            : '''"""Module 3 docstring."""
def function1():
    """Function 1 docstring."""
    pass
def function2(): pass
''',
        }

        # Write the files
        for path, content in files.items():
            full_path = os.path.join(temp_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)

        yield temp_dir


# Tests for DocstringVisitor
def test_docstring_visitor_module():
    """Test that DocstringVisitor correctly identifies module docstrings."""
    code = '''"""Module docstring."""

def func(): pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "module" in visitor.documented_items["module"]
    assert len(visitor.documented_items["module"]) == 1
    assert len(visitor.undocumented_items["module"]) == 0


def test_docstring_visitor_no_module_docstring():
    """Test that DocstringVisitor correctly identifies missing module docstrings."""
    code = '''
def func(): pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "module" in visitor.undocumented_items["module"]
    assert len(visitor.documented_items["module"]) == 0
    assert len(visitor.undocumented_items["module"]) == 1


def test_docstring_visitor_class():
    """Test that DocstringVisitor correctly identifies class docstrings."""
    code = '''
class TestClass:
    """Class docstring."""
    def method(self): pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "TestClass" in visitor.documented_items["class"]
    assert len(visitor.documented_items["class"]) == 1
    assert len(visitor.undocumented_items["class"]) == 0


def test_docstring_visitor_no_class_docstring():
    """Test that DocstringVisitor correctly identifies missing class docstrings."""
    code = '''
class TestClass:
    def method(self): pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "TestClass" in visitor.undocumented_items["class"]
    assert len(visitor.documented_items["class"]) == 0
    assert len(visitor.undocumented_items["class"]) == 1


def test_docstring_visitor_function():
    """Test that DocstringVisitor correctly identifies function docstrings."""
    code = '''
def test_function():
    """Function docstring."""
    pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "test_function" in visitor.documented_items["function"]
    assert len(visitor.documented_items["function"]) == 1
    assert len(visitor.undocumented_items["function"]) == 0


def test_docstring_visitor_no_function_docstring():
    """Test that DocstringVisitor correctly identifies missing function docstrings."""
    code = '''
def test_function():
    pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "test_function" in visitor.undocumented_items["function"]
    assert len(visitor.documented_items["function"]) == 0
    assert len(visitor.undocumented_items["function"]) == 1


def test_docstring_visitor_method():
    """Test that DocstringVisitor correctly identifies method docstrings."""
    code = '''
class TestClass:
    def test_method(self):
        """Method docstring."""
        pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "TestClass.test_method" in visitor.documented_items["method"]
    assert len(visitor.documented_items["method"]) == 1
    assert len(visitor.undocumented_items["method"]) == 0


def test_docstring_visitor_no_method_docstring():
    """Test that DocstringVisitor correctly identifies missing method docstrings."""
    code = '''
class TestClass:
    def test_method(self):
        pass
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "TestClass.test_method" in visitor.undocumented_items["method"]
    assert len(visitor.documented_items["method"]) == 0
    assert len(visitor.undocumented_items["method"]) == 1


def test_docstring_visitor_special_methods():
    """Test that DocstringVisitor correctly handles special methods."""
    code = '''
class TestClass:
    def __init__(self):
        """Init docstring."""
        pass
        
    def __repr__(self):
        # Special methods (except __init__) don't need docstrings
        return "TestClass()"
'''
    tree = ast.parse(code)
    visitor = DocstringVisitor()
    visitor.visit(tree)

    assert "TestClass.__init__" in visitor.documented_items["method"]
    # __repr__ should not be counted as undocumented
    assert "TestClass.__repr__" not in visitor.undocumented_items["method"]
    assert "TestClass.__repr__" not in visitor.all_items["method"]


# Tests for scan_file function
@pytest.mark.trio
async def test_scan_file_well_documented(well_documented_file):
    """Test that scan_file correctly analyzes a well-documented file."""
    result = await scan_file(well_documented_file)

    assert result["success"] is True
    assert result["overall_percentage"] == 100.0
    assert result["total_items"] > 0
    assert result["total_documented"] == result["total_items"]

    # Check individual categories
    assert result["stats"]["module"]["percentage"] == 100.0
    assert result["stats"]["class"]["percentage"] == 100.0
    assert result["stats"]["function"]["percentage"] == 100.0
    assert result["stats"]["method"]["percentage"] == 100.0


@pytest.mark.trio
async def test_scan_file_partially_documented(partially_documented_file):
    """Test that scan_file correctly analyzes a partially documented file."""
    result = await scan_file(partially_documented_file)

    assert result["success"] is True
    assert 0 < result["overall_percentage"] < 100
    assert result["total_items"] > 0
    assert result["total_documented"] < result["total_items"]

    # Check individual categories
    assert result["stats"]["module"]["percentage"] == 100.0
    assert result["stats"]["class"]["percentage"] == 100.0
    # Some functions and methods should be undocumented
    assert result["stats"]["function"]["percentage"] < 100
    assert result["stats"]["method"]["percentage"] < 100


@pytest.mark.trio
async def test_scan_file_undocumented(undocumented_file):
    """Test that scan_file correctly analyzes an undocumented file."""
    result = await scan_file(undocumented_file)

    assert result["success"] is True
    assert result["overall_percentage"] == 0.0
    assert result["total_items"] > 0
    assert result["total_documented"] == 0

    # Check individual categories
    assert result["stats"]["module"]["percentage"] == 0.0
    assert result["stats"]["class"]["percentage"] == 0.0
    assert result["stats"]["function"]["percentage"] == 0.0
    assert result["stats"]["method"]["percentage"] == 0.0


@pytest.mark.trio
async def test_scan_file_syntax_error(syntax_error_file):
    """Test that scan_file correctly handles files with syntax errors."""
    result = await scan_file(syntax_error_file)

    assert result["success"] is False
    assert "error" in result


# Tests for documentation line counting
@pytest.mark.trio
async def test_count_doc_and_code_lines(well_documented_file, undocumented_file):
    """Test that count_doc_and_code_lines correctly counts lines."""
    # Well-documented file should have many doc lines
    doc_lines, code_lines = await count_doc_and_code_lines(well_documented_file)
    assert doc_lines > 0
    assert code_lines > 0

    # Undocumented file should have no doc lines
    doc_lines, code_lines = await count_doc_and_code_lines(undocumented_file)
    assert doc_lines == 0
    assert code_lines > 0


# Tests for report generation
@pytest.mark.trio
async def test_generate_report():
    """Test that generate_report correctly generates a Markdown report."""
    # Create sample results dictionary
    results = {
        '/path/to/file1.py': {
            'success'           : True,
            'stats'             : {
                'module'  : {'total'           : 1, 'documented': 1, 'undocumented': 0, 'percentage': 100.0,
                             'documented_items': ['module'], 'undocumented_items': []},
                'class'   : {'total'           : 1, 'documented': 1, 'undocumented': 0, 'percentage': 100.0,
                             'documented_items': ['TestClass'], 'undocumented_items': []},
                'function': {'total'           : 1, 'documented': 1, 'undocumented': 0, 'percentage': 100.0,
                             'documented_items': ['test_function'], 'undocumented_items': []},
                'method'  : {'total'           : 1, 'documented': 1, 'undocumented': 0, 'percentage': 100.0,
                             'documented_items': ['TestClass.test_method'], 'undocumented_items': []}
            },
            'overall_percentage': 100.0,
            'total_items'       : 4,
            'total_documented'  : 4
        },
        '/path/to/file2.py': {
            'success'           : True,
            'stats'             : {
                'module'  : {'total'           : 1, 'documented': 0, 'undocumented': 1, 'percentage': 0.0,
                             'documented_items': [], 'undocumented_items': ['module']},
                'class'   : {'total'           : 1, 'documented': 0, 'undocumented': 1, 'percentage': 0.0,
                             'documented_items': [], 'undocumented_items': ['TestClass']},
                'function': {'total'           : 1, 'documented': 0, 'undocumented': 1, 'percentage': 0.0,
                             'documented_items': [], 'undocumented_items': ['test_function']},
                'method'  : {'total'           : 1, 'documented': 0, 'undocumented': 1, 'percentage': 0.0,
                             'documented_items': [], 'undocumented_items': ['TestClass.test_method']}
            },
            'overall_percentage': 0.0,
            'total_items'       : 4,
            'total_documented'  : 0
        }
    }

    report = await generate_report(results, '/path/to')

    # Verify report structure
    assert "# Documentation Status Report" in report
    assert "## Overall Statistics" in report
    assert "## Documentation by Directory" in report
    assert "## Files Needing Documentation" in report

    # Verify statistics are included
    assert "Files Scanned: 2" in report.replace("**", "")
    assert "Files Successfully Parsed: 2" in report.replace("**", "")
    assert "Total Items: 8" in report.replace("**", "")
    assert "Documented Items: 4" in report.replace("**", "")
    assert "Overall Documentation Percentage: 50.0%" in report.replace("**", "")

    # Verify directories are processed
    assert "### path/to (50.0%)" in report

    # Verify files needing documentation are listed
    assert "file2.py | 0.0%" in report


# Tests for directory scanning
@pytest.mark.trio
async def test_count_py_files(test_directory):
    """Test that count_py_files correctly counts Python files."""
    # With recursion
    count = await count_py_files(test_directory, recursive=True)
    assert count == 6  # Should find all 6 Python files

    # Without recursion
    count = await count_py_files(test_directory, recursive=False)
    assert count == 1  # Should only find the root Python file


@pytest.mark.trio
async def test_scan_directory(test_directory):
    """Test that scan_directory correctly scans a directory of Python files."""
    # Test with recursion
    results = await scan_directory(test_directory, quiet=True, recursive=True)

    assert len(results) == 6  # Should find all 6 Python files
    assert all(results[f].get("success", False) for f in results)

    # Test without recursion
    results = await scan_directory(test_directory, quiet=True, recursive=False)

    assert len(results) == 1  # Should only find the root Python file
    assert all(results[f].get("success", False) for f in results)

    # Test directory exclusion
    results = await scan_directory(
        test_directory,
        exclude_dirs=['package1'],
        quiet=True,
        recursive=True
    )

    # Should exclude package1 and its subdirectories
    assert len(results) == 3  # root file + package2 files
    assert all(results[f].get("success", False) for f in results)

    # Verify no paths contain 'package1'
    assert all('package1' not in f for f in results.keys())


# Tests for the complete workflow
@pytest.mark.trio
async def test_end_to_end_workflow(test_directory, tmp_path):
    """Test the complete workflow from scanning to report generation."""
    # Output file in a temporary location
    output_file = os.path.join(tmp_path, "test_report.md")

    # Scan the directory
    results = await scan_directory(test_directory, quiet=True, recursive=True)

    # Generate and write the report
    report = await generate_report(results, test_directory)

    # Write the report to file
    with open(output_file, 'w') as f:
        f.write(report)

    # Verify the report file exists and has content
    assert os.path.exists(output_file)
    with open(output_file, 'r') as f:
        content = f.read()
        assert "# Documentation Status Report" in content
        assert "## Overall Statistics" in content
        assert "Files Scanned: " in content


# Tests for command-line argument parsing
@pytest.mark.trio
async def test_async_main_default_args():
    """Test async_main with default arguments."""
    test_dir = "test_dir"

    # Mock ArgumentParser
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = argparse.Namespace(
        directory=test_dir,
        output=None,
        exclude=None,
        quiet=False,
        recursive=False
    )

    # Mock functions
    with patch('argparse.ArgumentParser', return_value=mock_parser), \
            patch('src.documentation_scanner.scan_directory') as mock_scan, \
            patch('src.documentation_scanner.generate_report') as mock_report, \
            patch('src.documentation_scanner.write_report') as mock_write:
        # Set up return values
        mock_scan.return_value = {}
        mock_report.return_value = "# Test Report"

        # Call the function
        await async_main()

        # Verify correct function calls with default arguments
        mock_scan.assert_called_once()
        args, kwargs = mock_scan.call_args
        assert kwargs['directory'] == test_dir
        assert kwargs['exclude_dirs'] == ['__pycache__', 'venv', '.git', '.vscode']
        assert kwargs['quiet'] == False
        assert kwargs['recursive'] == False

        # Verify report generation and writing
        mock_report.assert_called_once()
        mock_write.assert_called_once_with("# Test Report", "doc_report.md")


@pytest.mark.trio
async def test_async_main_custom_args():
    """Test async_main with custom arguments."""
    test_dir = "custom_dir"
    custom_output = "custom_report.md"
    custom_exclude = ["exclude1", "exclude2"]

    # Mock ArgumentParser
    mock_parser = MagicMock()
    mock_parser.parse_args.return_value = argparse.Namespace(
        directory=test_dir,
        output=custom_output,
        exclude=custom_exclude,
        quiet=True,
        recursive=True
    )

    # Mock functions
    with patch('argparse.ArgumentParser', return_value=mock_parser), \
            patch('src.documentation_scanner.scan_directory') as mock_scan, \
            patch('src.documentation_scanner.generate_report') as mock_report, \
            patch('src.documentation_scanner.write_report') as mock_write:
        # Set up return values
        mock_scan.return_value = {}
        mock_report.return_value = "# Test Report"

        # Call the function
        await async_main()

        # Verify correct function calls with custom arguments
        mock_scan.assert_called_once()
        args, kwargs = mock_scan.call_args
        assert kwargs['directory'] == test_dir
        assert kwargs['exclude_dirs'] == custom_exclude
        assert kwargs['quiet'] == True
        assert kwargs['recursive'] == True

        # Verify report generation and writing with custom output
        mock_report.assert_called_once()
        mock_write.assert_called_once_with("# Test Report", custom_output)


# Tests for file writing
@pytest.mark.trio
async def test_write_report(tmp_path):
    """Test that write_report correctly writes content to a file."""
    # Create a test file path
    test_file = os.path.join(tmp_path, "test_report.md")
    test_content = "# Test Report\n\nThis is a test report."

    # Write the report
    await write_report(test_content, test_file)

    # Verify the file exists and contains the correct content
    assert os.path.exists(test_file)
    with open(test_file, 'r') as f:
        content = f.read()
        assert content == test_content


# Tests for edge cases
@pytest.mark.trio
async def test_empty_directory(tmp_path):
    """Test scanning an empty directory."""
    # Create an empty temporary directory
    empty_dir = os.path.join(tmp_path, "empty_dir")
    os.makedirs(empty_dir)

    # Scan the empty directory
    results = await scan_directory(empty_dir, quiet=True, recursive=True)

    # Should be empty dict since no Python files were found
    assert len(results) == 0


@pytest.mark.trio
async def test_directory_with_no_python_files(tmp_path):
    """Test scanning a directory with no Python files."""
    # Create a temporary directory with a non-Python file
    no_py_dir = os.path.join(tmp_path, "no_py_dir")
    os.makedirs(no_py_dir)

    # Create a text file
    with open(os.path.join(no_py_dir, "test.txt"), 'w') as f:
        f.write("This is not a Python file")

    # Scan the directory
    results = await scan_directory(no_py_dir, quiet=True, recursive=True)

    # Should be empty dict since no Python files were found
    assert len(results) == 0


@pytest.mark.trio
async def test_scan_file_read_error():
    """Test handling of file read errors."""
    # Create a path to a non-existent file
    non_existent_file = "this_file_does_not_exist.py"

    # Scan the non-existent file
    result = await scan_file(non_existent_file)

    # Should indicate failure
    assert result["success"] is False
    assert "error" in result
