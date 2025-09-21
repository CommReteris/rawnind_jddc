#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Documentation status scanner for Python projects.

This script scans Python files in a project to assess their documentation status,
calculating percentages of documented functions, methods, and classes.

It produces a report in Markdown format that can be used to update the documentation plan
and prioritize areas for documentation improvement.

This implementation uses the Trio library to perform asynchronous I/O operations,
which can improve performance when scanning large directories of files.
"""

import argparse
import ast
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import trio


class DocstringVisitor(ast.NodeVisitor):
    """AST Visitor to extract docstring information from Python files.
    
    Traverses abstract syntax trees of Python files to identify classes, functions,
    and methods, and checks whether they have docstrings.
    
    Attributes:
        documented_items: Dictionary tracking documented items by type
        undocumented_items: Dictionary tracking undocumented items by type
        all_items: Dictionary tracking all items by type
        current_class: Name of the class currently being processed
    """

    def __init__(self):
        """Initialize the docstring visitor."""
        self.documented_items = {"module": [], "class": [], "function": [], "method": []}
        self.undocumented_items = {"module": [], "class": [], "function": [], "method": []}
        self.all_items = {"module": [], "class": [], "function": [], "method": []}
        self.current_class = None

    def visit_Module(self, node: ast.Module):
        """Visit a module node to check for module-level docstring.
        
        Args:
            node: The AST module node
        """
        self.all_items["module"].append("module")
        if ast.get_docstring(node):
            self.documented_items["module"].append("module")
        else:
            self.undocumented_items["module"].append("module")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit a class definition to check for class docstring.
        
        Args:
            node: The AST class definition node
        """
        old_class = self.current_class
        self.current_class = node.name

        self.all_items["class"].append(node.name)
        if ast.get_docstring(node):
            self.documented_items["class"].append(node.name)
        else:
            self.undocumented_items["class"].append(node.name)

        # Visit all class contents
        for item in node.body:
            self.visit(item)

        self.current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit a function definition to check for function/method docstring.
        
        Args:
            node: The AST function definition node
        """
        # Skip special methods like __repr__ which typically don't need docstrings
        if node.name.startswith('__') and node.name.endswith('__') and node.name not in ['__init__']:
            return

        if self.current_class:
            item_type = "method"
            item_name = f"{self.current_class}.{node.name}"
        else:
            item_type = "function"
            item_name = node.name

        self.all_items[item_type].append(item_name)
        if ast.get_docstring(node):
            self.documented_items[item_type].append(item_name)
        else:
            self.undocumented_items[item_type].append(item_name)


async def scan_file(file_path: str) -> Dict[str, Any]:
    """Scan a Python file to assess its documentation status.
    
    Args:
        file_path: Path to the Python file to scan
        
    Returns:
        Dictionary containing documentation statistics and item lists
    """
    try:
        async with await trio.open_file(file_path, 'r', encoding='utf-8') as file:
            content = await file.read()
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f"Error parsing {file_path}: {e}")
                return {
                    "success": False,
                    "error"  : str(e)
                }
    except Exception as e:
        print(f"Error opening {file_path}: {e}")
        return {
            "success": False,
            "error"  : str(e)
        }

    visitor = DocstringVisitor()
    visitor.visit(tree)

    # Calculate percentages
    stats = {}
    for item_type in ["module", "class", "function", "method"]:
        total = len(visitor.all_items[item_type])
        documented = len(visitor.documented_items[item_type])
        stats[item_type] = {
            "total"             : total,
            "documented"        : documented,
            "undocumented"      : total - documented,
            "percentage"        : 100 * (documented / total) if total > 0 else 100,
            "documented_items"  : visitor.documented_items[item_type],
            "undocumented_items": visitor.undocumented_items[item_type]
        }

    # Calculate overall percentage
    total_items = sum(stats[t]["total"] for t in ["module", "class", "function", "method"])
    total_documented = sum(stats[t]["documented"] for t in ["module", "class", "function", "method"])

    return {
        "success"           : True,
        "stats"             : stats,
        "overall_percentage": 100 * (total_documented / total_items) if total_items > 0 else 100,
        "total_items"       : total_items,
        "total_documented"  : total_documented
    }


async def count_py_files(directory: str, exclude_dirs: Optional[List[str]] = None, recursive: bool = True) -> int:
    """Count the number of Python files in a directory.
    
    Args:
        directory: Root directory to scan
        exclude_dirs: List of directory names to exclude from scanning
        recursive: If True, scan subdirectories recursively; otherwise only scan the top level
        
    Returns:
        Number of Python files found
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'venv', '.git', '.vscode']

    count = 0

    if recursive:
        # Use trio_walk for recursive scanning
        async for root, dirs, files in trio_walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            count += sum(1 for file in files if file.endswith('.py'))
    else:
        # Non-recursive scan (top level only)
        try:
            # Run os.listdir in a separate thread to avoid blocking
            entries = await trio.to_thread.run_sync(os.listdir, directory)

            # Count Python files in the current directory
            for entry in entries:
                full_path = os.path.join(directory, entry)
                # Skip excluded directories
                if os.path.isdir(full_path) and entry in exclude_dirs:
                    continue
                if os.path.isfile(full_path) and entry.endswith('.py'):
                    count += 1

        except (PermissionError, FileNotFoundError) as e:
            print(f"Error accessing directory {directory}: {e}")

    return count


async def trio_walk(top):
    """Asynchronous version of os.walk using Trio.
    
    This implementation runs os.walk in a separate thread using
    trio.to_thread.run_sync to avoid blocking the event loop.
    
    Args:
        top: Directory to walk
        
    Yields:
        Tuples of (dirpath, dirnames, filenames) similar to os.walk
    """
    # Run os.walk in a separate thread to avoid blocking
    for dirpath, dirnames, filenames in await trio.to_thread.run_sync(lambda: list(os.walk(top))):
        # Filter out excluded directories if needed
        yield dirpath, dirnames, filenames


async def count_doc_and_code_lines(file_path: str) -> Tuple[int, int]:
    """Count lines of documentation and code in a Python file.
    
    Args:
        file_path: Path to the Python file to analyze
        
    Returns:
        Tuple of (documentation_lines, code_lines)
    """
    try:
        async with await trio.open_file(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, 0

    # Parse the file to get docstrings
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return 0, 0

    # Count total lines in file
    total_lines = len(content.splitlines())

    # Extract all docstrings
    docstrings = []

    # Module docstring
    if ast.get_docstring(tree):
        docstrings.append(ast.get_docstring(tree))

    # Class and function docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if docstring := ast.get_docstring(node):
                docstrings.append(docstring)

    # Count lines in all docstrings
    doc_lines = sum(len(docstring.splitlines()) for docstring in docstrings)

    # Code lines are all lines minus docstring lines
    # This is an approximation as it doesn't account for comment lines
    code_lines = total_lines - doc_lines

    return doc_lines, code_lines


def display_live_stats(stats: Dict[str, Any], num_lines: int = 6):
    """Display live statistics below the progress bar.
    
    Uses ANSI escape codes to update statistics in-place without scrolling.
    
    Args:
        stats: Dictionary containing statistics to display
        num_lines: Number of lines to use for the statistics display
    """
    # Move cursor up to overwrite previous stats
    sys.stdout.write("\033[F" * num_lines)

    # Clear each line and write new stats
    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Files documented: {stats['fully_doc_files']}/{stats['processed_files']} " +
                     f"({100 * stats['fully_doc_files'] / max(1, stats['processed_files']):.1f}%)\n")

    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Classes documented: {stats['doc_classes']}/{stats['total_classes']} " +
                     f"({100 * stats['doc_classes'] / max(1, stats['total_classes']):.1f}%)\n")

    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Functions documented: {stats['doc_functions']}/{stats['total_functions']} " +
                     f"({100 * stats['doc_functions'] / max(1, stats['total_functions']):.1f}%)\n")

    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Methods documented: {stats['doc_methods']}/{stats['total_methods']} " +
                     f"({100 * stats['doc_methods'] / max(1, stats['total_methods']):.1f}%)\n")

    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Doc lines / code lines: {stats['doc_lines']}/{stats['code_lines']} " +
                     f"({100 * stats['doc_lines'] / max(1, stats['code_lines']):.1f}%)\n")

    sys.stdout.write("\033[K")  # Clear current line
    sys.stdout.write(f"Overall documentation: {stats['total_documented']}/{stats['total_items']} " +
                     f"({100 * stats['total_documented'] / max(1, stats['total_items']):.1f}%)\n")

    # Flush to ensure immediate display
    sys.stdout.flush()


async def scan_directory(directory: str, exclude_dirs: Optional[List[str]] = None,
                         quiet: bool = False, recursive: bool = True) -> Dict[str, Dict[str, Any]]:
    """Scan Python files in a directory.
    
    Displays a progress bar and estimated time remaining, as well as live statistics
    unless quiet mode is enabled.
    
    Args:
        directory: Root directory to scan
        exclude_dirs: List of directory names to exclude from scanning
        quiet: If True, suppress progress bar and live statistics
        recursive: If True, scan subdirectories recursively; otherwise only scan the top level
        
    Returns:
        Dictionary mapping file paths to their documentation statistics
    """
    if exclude_dirs is None:
        exclude_dirs = ['__pycache__', 'venv', '.git', '.vscode']

    # First, count the total files to process for accurate progress reporting
    if not quiet:
        print("Counting Python files...")
        total_files = await count_py_files(directory, exclude_dirs, recursive)
        print(f"Found {total_files} Python files to scan")

    results = {}
    processed_files = 0
    total_items = 0
    total_documented = 0

    # Initialize additional statistics
    fully_doc_files = 0
    total_classes = 0
    doc_classes = 0
    total_functions = 0
    doc_functions = 0
    total_methods = 0
    doc_methods = 0
    doc_lines = 0
    code_lines = 0

    # Track the 5 files with lowest documentation percentage
    low_doc_files = []

    # Collect all Python files first
    all_python_files = []

    if recursive:
        # Recursive scan using trio_walk
        async for root, dirs, files in trio_walk(directory):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if file.endswith('.py'):
                    all_python_files.append(os.path.join(root, file))
    else:
        # Non-recursive scan (top level only)
        try:
            # Run os.listdir in a separate thread to avoid blocking
            entries = await trio.to_thread.run_sync(os.listdir, directory)

            # Add Python files from the current directory
            for entry in entries:
                full_path = os.path.join(directory, entry)
                # Skip excluded directories
                if os.path.isdir(full_path) and entry in exclude_dirs:
                    continue
                if os.path.isfile(full_path) and entry.endswith('.py'):
                    all_python_files.append(full_path)

        except (PermissionError, FileNotFoundError) as e:
            print(f"Error accessing directory {directory}: {e}")

    # Create progress bar for file processing
    if not quiet and all_python_files:
        # Print empty lines for stats that will be updated in place
        print("\n\n\n\n\n\n")  # 6 lines for statistics

    # Use Trio's nursery for concurrent processing
    async with trio.open_nursery() as nursery:
        # Create a processing limiter to avoid overwhelming the system
        limiter = trio.CapacityLimiter(min(8, os.cpu_count() or 4))

        # Create a lock for updating shared statistics
        stats_lock = trio.Lock()

        # Define the async task to process a file
        async def process_file(file_path):
            nonlocal processed_files, total_items, total_documented
            nonlocal fully_doc_files, total_classes, doc_classes
            nonlocal total_functions, doc_functions, total_methods, doc_methods
            nonlocal doc_lines, code_lines, low_doc_files

            # Limit concurrency
            async with limiter:
                try:
                    # Scan the file for documentation
                    result = await scan_file(file_path)

                    # Count doc and code lines
                    file_doc_lines, file_code_lines = await count_doc_and_code_lines(file_path)

                    # Update statistics with lock to prevent race conditions
                    async with stats_lock:
                        results[file_path] = result

                        if result.get("success", False):
                            processed_files += 1
                            file_total_items = result.get("total_items", 0)
                            file_documented_items = result.get("total_documented", 0)
                            file_percentage = result.get("overall_percentage", 0)

                            # Track if file is fully documented (100%)
                            if file_percentage == 100 and file_total_items > 0:
                                fully_doc_files += 1

                            # Count classes, functions, and methods
                            if "stats" in result:
                                total_classes += result["stats"]["class"]["total"]
                                doc_classes += result["stats"]["class"]["documented"]
                                total_functions += result["stats"]["function"]["total"]
                                doc_functions += result["stats"]["function"]["documented"]
                                total_methods += result["stats"]["method"]["total"]
                                doc_methods += result["stats"]["method"]["documented"]

                            # Accumulate documentation and code lines
                            doc_lines += file_doc_lines
                            code_lines += file_code_lines

                            total_items += file_total_items
                            total_documented += file_documented_items

                            # Track files with low documentation
                            if file_percentage < 70 and file_total_items > 0:
                                if len(low_doc_files) < 5:
                                    low_doc_files.append((file_path, file_percentage))
                                    low_doc_files.sort(key=lambda x: x[1])
                                elif file_percentage < low_doc_files[-1][1]:
                                    low_doc_files.pop()
                                    low_doc_files.append((file_path, file_percentage))
                                    low_doc_files.sort(key=lambda x: x[1])

                            # Update live statistics in place
                            if not quiet:
                                stats = {
                                    'processed_files' : processed_files,
                                    'fully_doc_files' : fully_doc_files,
                                    'total_classes'   : total_classes,
                                    'doc_classes'     : doc_classes,
                                    'total_functions' : total_functions,
                                    'doc_functions'   : doc_functions,
                                    'total_methods'   : total_methods,
                                    'doc_methods'     : doc_methods,
                                    'doc_lines'       : doc_lines,
                                    'code_lines'      : code_lines,
                                    'total_items'     : total_items,
                                    'total_documented': total_documented
                                }
                                display_live_stats(stats)

                                # Update progress display
                                progress = f"{processed_files}/{len(all_python_files)} files processed"
                                percent = 100 * processed_files / len(all_python_files)
                                print(
                                    f"\r\033[K{progress} ({percent:.1f}%) - Current: {os.path.basename(file_path)} ({file_percentage:.1f}%)",
                                    end="")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    async with stats_lock:
                        results[file_path] = {
                            "success": False,
                            "error"  : str(e)
                        }

        # Start processing each file
        for file_path in all_python_files:
            nursery.start_soon(process_file, file_path)

    # Clear the progress line
    if not quiet and all_python_files:
        print("\r\033[K", end="")

    if not quiet:
        # Print final statistics after all files are processed
        overall_percentage = 100 * (total_documented / total_items) if total_items > 0 else 0
        print(f"\nScanning complete! Processed {processed_files} files")
        print(f"Overall documentation: {total_documented}/{total_items} items ({overall_percentage:.1f}%)")

        if low_doc_files:
            print("\nTop files needing documentation:")
            for low_file, low_pct in low_doc_files:
                rel_path = os.path.relpath(low_file, directory)
                print(f"- {rel_path}: {low_pct:.1f}%")

    return results


async def generate_report(results: Dict[str, Dict[str, Any]], base_dir: str) -> str:
    """Generate a Markdown report of documentation status.
    
    Args:
        results: Dictionary mapping file paths to documentation statistics
        base_dir: Base directory for generating relative paths
        
    Returns:
        Markdown-formatted report string
    """
    report = "# Documentation Status Report\n\n"
    report += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Overall statistics
    successful_files = [f for f, r in results.items() if r.get("success", False)]
    total_items = sum(r.get("total_items", 0) for r in results.values() if r.get("success", False))
    total_documented = sum(r.get("total_documented", 0) for r in results.values() if r.get("success", False))
    overall_percentage = 100 * (total_documented / total_items) if total_items > 0 else 0

    report += f"## Overall Statistics\n\n"
    report += f"- **Files Scanned**: {len(results)}\n"
    report += f"- **Files Successfully Parsed**: {len(successful_files)}\n"
    report += f"- **Total Items**: {total_items}\n"
    report += f"- **Documented Items**: {total_documented}\n"
    report += f"- **Overall Documentation Percentage**: {overall_percentage:.1f}%\n\n"

    # Group files by directory
    files_by_dir = {}
    for file_path in sorted(results.keys()):
        if not results[file_path].get("success", False):
            continue

        rel_path = os.path.relpath(file_path, base_dir)
        dir_name = os.path.dirname(rel_path)
        if dir_name not in files_by_dir:
            files_by_dir[dir_name] = []
        files_by_dir[dir_name].append((file_path, rel_path))

    # Report by directory
    report += "## Documentation by Directory\n\n"

    for dir_name in sorted(files_by_dir.keys()):
        dir_files = files_by_dir[dir_name]
        dir_total_items = sum(results[f].get("total_items", 0) for f, _ in dir_files)
        dir_total_documented = sum(results[f].get("total_documented", 0) for f, _ in dir_files)
        dir_percentage = 100 * (dir_total_documented / dir_total_items) if dir_total_items > 0 else 0

        report += f"### {dir_name or 'Root'} ({dir_percentage:.1f}%)\n\n"
        report += "| File | Documentation % | Module | Classes | Functions | Methods |\n"
        report += "|------|----------------|--------|---------|-----------|--------|\n"

        for file_path, rel_path in dir_files:
            r = results[file_path]
            if not r.get("success", False):
                continue

            file_percentage = r.get("overall_percentage", 0)
            module_percentage = r["stats"]["module"]["percentage"] if "stats" in r else 0
            class_percentage = r["stats"]["class"]["percentage"] if "stats" in r else 0
            function_percentage = r["stats"]["function"]["percentage"] if "stats" in r else 0
            method_percentage = r["stats"]["method"]["percentage"] if "stats" in r else 0

            file_name = os.path.basename(rel_path)
            report += f"| {file_name} | {file_percentage:.1f}% | {module_percentage:.0f}% | {class_percentage:.0f}% | {function_percentage:.0f}% | {method_percentage:.0f}% |\n"

        report += "\n"

    # Files with lowest documentation
    report += "## Files Needing Documentation\n\n"
    report += "Files with documentation percentage below 70%, sorted by coverage:\n\n"
    report += "| File | Documentation % | Undocumented Items |\n"
    report += "|------|----------------|-------------------|\n"

    low_doc_files = []
    for file_path, r in results.items():
        if not r.get("success", False):
            continue

        file_percentage = r.get("overall_percentage", 0)
        if file_percentage < 70:
            undocumented = []
            if "stats" in r:
                for item_type in ["module", "class", "function", "method"]:
                    undocumented.extend(r["stats"][item_type]["undocumented_items"])
            rel_path = os.path.relpath(file_path, base_dir)
            low_doc_files.append((rel_path, file_percentage, undocumented))

    for rel_path, file_percentage, undocumented in sorted(low_doc_files, key=lambda x: x[1]):
        undocumented_str = ", ".join(undocumented[:5])
        if len(undocumented) > 5:
            undocumented_str += f", ... ({len(undocumented) - 5} more)"
        report += f"| {rel_path} | {file_percentage:.1f}% | {undocumented_str} |\n"

    return report


async def write_report(report_content: str, output_file: str):
    """Write report to a file using async I/O.
    
    Args:
        report_content: Content to write to the file
        output_file: Path to the output file
    """
    async with await trio.open_file(output_file, 'w', encoding='utf-8') as f:
        await f.write(report_content)


async def async_main():
    """Main async function to run the documentation scanner."""
    parser = argparse.ArgumentParser(description="Scan Python files for documentation status")
    parser.add_argument("directory", help="Directory to scan")
    parser.add_argument("--output", "-o", help="Output file for the report (default: doc_report.md)")
    parser.add_argument("--exclude", "-e", nargs="+", help="Directories to exclude from scanning")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress bar and live statistics")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Scan subdirectories recursively (default: scan only top level)")

    args = parser.parse_args()

    directory = args.directory
    output_file = args.output or "doc_report.md"
    exclude_dirs = args.exclude or ['__pycache__', 'venv', '.git', '.vscode']
    quiet = args.quiet
    recursive = args.recursive

    print(f"Scanning directory: {directory}")
    print(f"Recursive scan: {'enabled' if recursive else 'disabled'}")
    print(f"Excluding directories: {exclude_dirs}")
    print(f"Quiet mode: {'enabled' if quiet else 'disabled'}")

    start_time = time.time()
    results = await scan_directory(directory, exclude_dirs, quiet, recursive)
    end_time = time.time()

    report = await generate_report(results, directory)

    await write_report(report, output_file)

    print(f"Report written to {output_file}")
    print(f"Total scan time: {end_time - start_time:.2f} seconds")


def main():
    """Main function to run the documentation scanner using Trio."""
    trio.run(async_main)


if __name__ == "__main__":
    main()
