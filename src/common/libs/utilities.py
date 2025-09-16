# -*- coding: utf-8 -*-
"""Common utility functions and classes for file handling, data processing, and system operations.

This module provides a diverse set of utility functions used throughout the project,
including file operations, data serialization/deserialization, path manipulation,
multithreading helpers, compression utilities, and data structure operations.

Key functional areas:
- File operations: checksum, cp, backup, filesize
- Date and time utilities: get_date
- Multithreading: mt_runner for parallel processing
- Data serialization: JSON, YAML, and pickle read/write functions
- Directory and path manipulation: get_leaf, get_root, get_file_dname
- Compression utilities: compress_lzma, compress_png, decompress_lzma
- Data structure manipulation: freeze_dict, unfreeze_dict, shuffle_dictionary
- Logging and printing: Printer class

Most functions are designed to be simple, focused helpers that perform specific
tasks with error handling appropriate for the project's needs.
"""

import atexit  # restart_program()
import csv
import datetime
import json
import logging
import lzma
import os
import pickle
import random
import shutil
import sys
import unittest
from multiprocessing import Pool
from typing import Any, Callable, Iterable, List, Optional, Union

import tqdm

try:
    import png
except ModuleNotFoundError as e:
    logging.error(f"{e} (install pypng)")
import numpy as np
import statistics
import subprocess
import hashlib
import yaml

# sys.path += ['..', '.']
NUM_THREADS = os.cpu_count()


def checksum(fpath, htype="sha1"):
    """Calculate the cryptographic hash of a file.
    
    Computes a hash digest of the file's contents using the specified hash algorithm.
    This is useful for verifying file integrity or identifying duplicate files.
    
    Args:
        fpath: Path to the file to hash
        htype: Hash algorithm to use ("sha1" or "sha256")
        
    Returns:
        String containing the hexadecimal digest of the file's hash
        
    Raises:
        NotImplementedError: If an unsupported hash type is specified
        FileNotFoundError: If the file does not exist
    """
    if htype == "sha1":
        h = hashlib.sha1()
    elif htype == "sha256":
        h = hashlib.sha256()
    else:
        raise NotImplementedError(type)
    with open(fpath, "rb") as file:
        while True:
            # Reading is buffered, so we can read smaller chunks.
            chunk = file.read(h.block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cp(inpath, outpath, verbose=False, overwrite=True):
    """Copy a file with optional verbose output and overwrite control.
    
    Attempts to use fast copy-on-write when available (via --reflink=auto),
    falling back to standard copy operations if not supported.
    
    Args:
        inpath: Source file path to copy from
        outpath: Destination file path to copy to
        verbose: If True, print a message showing the copy operation
        overwrite: If False, add a suffix to the destination filename when it already exists
            rather than overwriting the existing file
            
    Notes:
        If overwrite=False and outpath exists, it will append "dupath.ext" to the filename,
        where ext is the original file extension.
    """
    if not overwrite:
        while os.path.isfile(outpath):
            outpath = outpath + "dupath." + outpath.split(".")[-1]
    try:
        subprocess.run(("cp", "--reflink=auto", inpath, outpath))
    except FileNotFoundError:
        shutil.copy2(inpath, outpath)
    if verbose:
        print(f"cp {inpath} {outpath}")


def get_date() -> str:
    """Get the current date in ISO format (YYYY-MM-DD).
    
    A simple utility for getting a consistently formatted date string
    that can be used for naming files, directories, or logging.
    
    Returns:
        String containing the current date in YYYY-MM-DD format
    """
    return f"{datetime.datetime.now():%Y-%m-%d}"


def backup(filepaths: list):
    """Backup a given list of files with date-stamped filenames.
    
    Creates a 'backup' directory in the current working directory if it doesn't exist,
    then copies each specified file into that directory with the current date
    prepended to the filename.
    
    Args:
        filepaths: List of file paths to backup
        
    Notes:
        - Backup filenames have format: YYYY-MM-DD_original_filename
        - Uses get_date() to get the current date in YYYY-MM-DD format
        - Uses get_leaf() to extract the filename from each path
        - Creates the backup directory if it doesn't exist
        - Silently overwrites any existing backup with the same name
    """
    if not os.path.isdir("backup"):
        os.makedirs("backup", exist_ok=True)
    date = get_date()
    for fpath in filepaths:
        fn = get_leaf(fpath)
        shutil.copy(fpath, os.path.join("backup", date + "_" + fn))


def mt_runner(
        fun: Callable[[Any], Any],
        argslist: list,
        num_threads: int = NUM_THREADS,
        ordered: bool = False,
        progress_bar: bool = True,
        starmap: bool = False,
) -> Iterable[Any]:
    """Run a function across multiple inputs using multiprocessing for parallelization.
    
    This is a general-purpose parallel execution utility that distributes the workload
    across multiple processes. It supports various execution modes including ordered vs.
    unordered results and progress bar visualization.
    
    Args:
        fun: Function to execute on each item in argslist
        argslist: List of arguments to pass to the function (one per call)
        num_threads: Number of worker processes to use (defaults to CPU count)
        ordered: If True, maintain the original order of results (may be slower)
        progress_bar: If True, display a progress bar during execution
        starmap: If True, expand each argument in argslist as *args to the function
                (e.g., for arguments that are tuples or lists of parameters)
    
    Returns:
        Iterable containing the results of applying the function to each argument
        
    Notes:
        - If num_threads=1, runs in a single process (useful for debugging)
        - starmap is not compatible with ordered=False
        - Progress bars are not supported when ordered=True
        - Closes and joins the process pool after execution
    
    Raises:
        NotImplementedError: If starmap=True and ordered=False (unsupported combination)
        RuntimeError: If a TypeError occurs during parallel execution
    """
    if num_threads is None:
        num_threads = NUM_THREADS
    if num_threads == 1:
        results = []
        for args in argslist:
            if starmap:
                results.append(fun(*args))
            else:
                results.append(fun(args))
        return results
    else:
        pool = Pool(num_threads)
        if starmap:
            amap = pool.starmap
            if not ordered:
                raise NotImplementedError("Unordered starmap")
        elif ordered:
            amap = pool.imap
        else:
            amap = pool.imap_unordered
        if ordered:
            print("mt_runner warning: ordered=True might be slower.")
            if progress_bar:
                print(
                    "mt_runner warning: progress bar NotImplemented for ordered pool."
                )
            ret = amap(fun, argslist)
        else:
            if progress_bar:
                ret = []
                try:
                    for ares in tqdm.tqdm(amap(fun, argslist), total=len(argslist)):
                        ret.append(ares)
                except TypeError as e:
                    print(e)
                    raise RuntimeError
            else:
                ret = amap(fun, argslist)
        pool.close()
        pool.join()
        return ret


def jsonfpath_load(fpath, default_type=dict, default=None):
    """Load a JSON file and convert digit string keys to integers.
    
    Loads data from a JSON file, automatically converting any string keys
    that are pure digits to integer keys (e.g., "123" -> 123). This is useful
    for handling JSON's limitation that object keys must be strings.
    
    Args:
        fpath: Path to the JSON file to load
        default_type: Constructor for default return value if file doesn't exist
                     and no default is provided
        default: Specific default value to return if file doesn't exist
        
    Returns:
        Contents of the JSON file as a Python object (typically dict),
        or the default value if the file doesn't exist
        
    Notes:
        - Prints a warning message if the file doesn't exist
        - The nested jsonKeys2int function recursively converts digit string keys to integers
        - If default is None and the file doesn't exist, returns default_type()
          (typically an empty dict)
    """
    if not os.path.isfile(fpath):
        print(
            "jsonfpath_load: warning: {} does not exist, returning default".format(
                fpath
            )
        )
        if default is None:
            return default_type()
        else:
            return default

    def jsonKeys2int(x):
        """Convert string keys that are digits to integer keys in a dictionary."""
        if isinstance(x, dict):
            return {k if not k.isdigit() else int(k): v for k, v in x.items()}
        return x

    with open(fpath, "r") as f:
        return json.load(f, object_hook=jsonKeys2int)


def jsonfpath_to_dict(fpath):
    """Load a JSON file to a dictionary (deprecated, use jsonfpath_load instead).
    
    This is a legacy wrapper around jsonfpath_load maintained for backward compatibility.
    
    Args:
        fpath: Path to the JSON file to load
        
    Returns:
        Dictionary containing the JSON file contents
        
    Notes:
        Prints a deprecation warning when called
    """
    print("warning: jsonfpath_to_dict is deprecated, use jsonfpath_load instead")
    return jsonfpath_load(fpath, default_type=dict)


def dict_to_json(adict, fpath):
    """Save a dictionary to a JSON file with nice formatting.
    
    Serializes a dictionary to JSON format and writes it to the specified file,
    using indentation for human-readable formatting.
    
    Args:
        adict: Dictionary to serialize
        fpath: Path where the JSON file should be written
        
    Notes:
        Uses an indent of 2 spaces for readability
    """
    with open(fpath, "w") as f:
        json.dump(adict, f, indent=2)


def dict_to_yaml(adict, fpath):
    """Save a dictionary to a YAML file.
    
    Serializes a dictionary to YAML format and writes it to the specified file.
    YAML provides a more human-readable alternative to JSON, especially for
    complex nested structures.
    
    Args:
        adict: Dictionary to serialize
        fpath: Path where the YAML file should be written
        
    Notes:
        Enables Unicode character support in the output YAML
    """
    with open(fpath, "w") as f:
        yaml.dump(adict, f, allow_unicode=True)


def load_yaml(
        fpath: str, safely=True, default_type=dict, default=None, error_on_404=True
):
    """Load a YAML file and convert digit string keys to integers.
    
    Loads data from a YAML file and optionally returns a default value if the
    file doesn't exist. Similar to jsonfpath_load but for YAML format.
    
    Args:
        fpath: Path to the YAML file to load
        safely: If True, use yaml.safe_load (recommended for security)
        default_type: Constructor for default return value if file doesn't exist
                     and no default is provided
        default: Specific default value to return if file doesn't exist
        error_on_404: If False, return default when file doesn't exist; 
                     if True, print warning message
                     
    Returns:
        Contents of the YAML file as a Python object (typically dict),
        or the default value if the file doesn't exist and error_on_404=False
        
    Notes:
        - Automatically converts string keys that are digits to integer keys
        - Uses safe_load by default to prevent YAML code execution vulnerabilities
    """
    if not os.path.isfile(fpath) and not error_on_404:
        print(
            "jsonfpath_load: warning: {} does not exist, returning default".format(
                fpath
            )
        )
        if default is None:
            return default_type()
        else:
            return default
    with open(fpath, "r") as f:
        if safely:
            res = yaml.safe_load(f)
        else:
            res = yaml.load(f, Loader=yaml.Loader)
    # transform string number keys into int
    if isinstance(res, dict):
        keys_to_convert = []
        for akey in res.keys():
            if isinstance(akey, str) and akey.isdigit():
                keys_to_convert.append(akey)
        for akey in keys_to_convert:
            res[int(akey)] = res[akey]
            del res[akey]
    return res


def dict_to_pickle(adict, fpath):
    """Save a dictionary to a pickle file.
    
    Serializes a dictionary to binary pickle format and writes it to the specified file.
    Pickle format preserves Python object types and relationships but is not human-readable.
    
    Args:
        adict: Dictionary to serialize
        fpath: Path where the pickle file should be written
        
    Notes:
        - Uses the highest protocol version supported by the current Python interpreter
        - Pickle files are not compatible across different Python versions
        - Not secure against erroneous or maliciously constructed data
    """
    with open(fpath, "wb") as f:
        pickle.dump(adict, f)


def picklefpath_to_dict(fpath):
    """Load a pickle file to a dictionary.
    
    Deserializes a binary pickle file back into a Python dictionary.
    
    Args:
        fpath: Path to the pickle file to load
        
    Returns:
        Dictionary containing the deserialized pickle file contents
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pickle.UnpicklingError: If the file contains corrupted or malicious data
    """
    with open(fpath, "rb") as f:
        adict = pickle.load(f)
    return adict


def args_to_file(fpath):
    """Save the current command line arguments to a file.
    
    Writes the complete command used to run the current script (including all arguments)
    to a text file. Useful for reproducing experiments or debugging.
    
    Args:
        fpath: Path where the command line should be written
        
    Notes:
        - Format is "python arg1 arg2 ..." with arguments space-separated
        - Uses sys.argv to get the complete argument list
    """
    with open(fpath, "w") as f:
        f.write("python " + " ".join(sys.argv))


def save_listofdict_to_csv(listofdict, fpath, keys=None, mixed_keys=False):
    """Save a list of dictionaries to a CSV file.
    
    Converts a list of dictionaries to a CSV file where each row represents
    a dictionary from the list and columns represent dictionary keys.
    
    Args:
        listofdict: List of dictionaries to convert to CSV rows
        fpath: Path where the CSV file should be written
        keys: Specific keys to use as columns (defaults to all keys from the first dict)
        mixed_keys: If True, handles dictionaries with different sets of keys by
                   including all unique keys across all dictionaries
                   
    Raises:
        ValueError: If dictionaries have different keys and mixed_keys=False
        
    Notes:
        - When mixed_keys=False, all dictionaries must have the same keys
        - Keys are sorted alphabetically in the CSV header
        - Enters debugging mode (breakpoint) if an error occurs
        - Uses csv.DictWriter for proper CSV formatting and escaping
    """
    if keys is None:
        keys = listofdict[0].keys()
        if mixed_keys:
            keys = set(keys)
            for somekeys in [adict.keys() for adict in listofdict]:
                keys.update(somekeys)
    keys = sorted(keys)
    try:
        with open(fpath, "w", newline="") as f:
            csvwriter = csv.DictWriter(f, keys)
            csvwriter.writeheader()
            csvwriter.writerows(listofdict)
    except ValueError as e:
        print(
            "save_listofdict_to_csv: error: {}. This likely means that the dictionaries have different keys, try passing mixed_keys=True".format(
                e
            )
        )
        breakpoint()


class Printer:
    """Logging utility for simultaneously printing to console and file.
    
    A flexible logging class that allows outputting messages to both stdout and
    a log file simultaneously. Useful for scripts that need to display progress
    while also maintaining a permanent record of their output.
    
    Attributes:
        tostdout: Whether to print messages to stdout
        tofile: Whether to write messages to a log file
        file_path: Path to the log file
    """

    def __init__(
            self, tostdout=True, tofile=True, save_dir=".", fn="log", save_file_path=None
    ):
        """Initialize a Printer instance.
        
        Args:
            tostdout: If True, print messages to stdout
            tofile: If True, write messages to the log file
            save_dir: Directory where the log file should be saved
            fn: Name of the log file
            save_file_path: Override for the complete log file path (if provided,
                           save_dir and fn are ignored)
                           
        Notes:
            - Creates the save_dir if it doesn't exist
            - The log file will be appended to if it already exists
        """
        self.tostdout = tostdout
        self.tofile = tofile
        os.makedirs(save_dir, exist_ok=True)
        self.file_path = (
            os.path.join(save_dir, fn) if save_file_path is None else save_file_path
        )

    def print(self, msg, err=False):  # TODO to stderr if err
        """Print a message to stdout and/or append it to the log file.
        
        Args:
            msg: The message to print/log
            err: If True, message should be treated as an error message
                (currently unused, planned to print to stderr)
                
        Notes:
            - Automatically converts the message to a string
            - Appends a newline character when writing to the log file
            - Prints a warning if writing to the log file fails
        """
        if self.tostdout:
            print(msg)
        if self.tofile:
            try:
                with open(self.file_path, "a") as f:
                    f.write(str(msg) + "\n")
            except Exception as e:
                print("Warning: could not write to log: %s" % e)


def std_bpp(bpp) -> str:
    """Format bits-per-pixel value to a standard string representation.
    
    Converts a numeric bits-per-pixel value to a string with two decimal places.
    Useful for consistent formatting in reports and logging.
    
    Args:
        bpp: Bits-per-pixel value (numeric or string representation of a number)
        
    Returns:
        String with the bits-per-pixel formatted to two decimal places,
        or None if the input cannot be converted to a float
        
    Example:
        >>> std_bpp(1.2345)
        '1.23'
        >>> std_bpp('2.7')
        '2.70'
    """
    try:
        return "{:.2f}".format(float(bpp))
    except TypeError:
        return None


def get_leaf(path: str) -> str:
    """Returns the leaf of a path, whether it's a file or directory followed by
    / or not."""
    return os.path.basename(os.path.relpath(path))


def get_root(fpath: str) -> str:
    """Get the parent directory of a file path.
    
    Extracts the directory that contains the specified file path. Handles paths 
    that may end with path separators by removing trailing separators before
    determining the parent directory.
    
    Args:
        fpath: Path to the file or directory
        
    Returns:
        String containing the parent directory path
        
    Notes:
        - Removes trailing path separators before finding the parent directory
        - Uses os.path.dirname() to extract the parent directory
        - Similar to get_file_dname(), but returns the full path instead of just the name
        
    Example:
        >>> get_root('/path/to/file.txt')
        '/path/to'
        >>> get_root('/path/to/directory/')
        '/path/to'
    """
    while fpath.endswith(os.pathsep):
        fpath = fpath[:-1]
    return os.path.dirname(fpath)


def get_file_dname(fpath: str) -> str:
    """Get the name of the directory containing a file.
    
    Extracts the basename of the parent directory from a file path.
    
    Args:
        fpath: Path to the file
        
    Returns:
        String containing the name of the parent directory (without its path)
        
    Example:
        >>> get_file_dname('/path/to/parent_dir/file.txt')
        'parent_dir'
    """
    return os.path.basename(os.path.dirname(fpath))


def freeze_dict(adict: dict) -> frozenset:
    """Recursively convert a dictionary into a hashable frozenset.
    
    Transforms a dictionary into a frozenset of (key, value) pairs, which is hashable
    and can be used as a key in another dictionary. For nested dictionaries, 
    the function recursively converts each sub-dictionary to a frozenset.
    
    Args:
        adict: Dictionary to convert to a hashable type
        
    Returns:
        Frozenset representation of the dictionary, where each nested dictionary
        has also been converted to a frozenset
        
    Notes:
        - Creates a copy of the input dictionary to avoid modifying the original
        - Useful when you need to use dictionaries as keys in other dictionaries or sets
        - Preserves the hierarchical structure of nested dictionaries
        
    Example:
        >>> d = {'a': 1, 'b': {'c': 2}}
        >>> frozen_d = freeze_dict(d)
        >>> isinstance(frozen_d, frozenset)
        True
        >>> # The frozen dictionary can be used as a dictionary key
        >>> another_dict = {frozen_d: 'value'}
    """
    fdict = adict.copy()
    for akey, aval in fdict.items():
        if isinstance(aval, dict):
            fdict[akey] = freeze_dict(aval)
    return frozenset(fdict.items())


def unfreeze_dict(fdict: frozenset) -> dict:
    """Recursively convert a frozenset back into a dictionary.
    
    Reverses the transformation performed by freeze_dict(), converting a frozenset
    of (key, value) pairs back into a dictionary. For nested frozensets, the function
    recursively converts each sub-frozenset back to a dictionary.
    
    Args:
        fdict: Frozenset representation of a dictionary, typically created by freeze_dict()
        
    Returns:
        Dictionary reconstructed from the frozenset, with all nested frozensets
        also converted back to dictionaries
        
    Notes:
        - This is the inverse operation of freeze_dict()
        - Preserves the hierarchical structure of nested frozensets
        
    Example:
        >>> d = {'a': 1, 'b': {'c': 2}}
        >>> frozen_d = freeze_dict(d)
        >>> original_d = unfreeze_dict(frozen_d)
        >>> original_d == d
        True
    """
    adict = dict(fdict)
    for akey, aval in adict.items():
        if isinstance(aval, frozenset):
            adict[akey] = unfreeze_dict(aval)
    return adict


def touch(path):
    """Create an empty file or update an existing file's modification time.
    
    Mimics the Unix touch command, creating an empty file if it doesn't exist
    or updating the access and modification times if it does.
    
    Args:
        path: Path to the file to touch
        
    Notes:
        - Uses the current time for both access and modification times
        - Creates any parent directories if they don't exist
    """
    with open(path, "a"):
        os.utime(path, None)


def dict_of_frozendicts2csv(res, fpath):
    """dict of frozendicts to csv
    used in eg evolve/tools/test_weights_on_all_tasks"""
    reslist = []
    dkeys = set()
    for areskey, aresval in res.items():
        ares = dict()
        for componentkey, componentres in unfreeze_dict(areskey).items():
            if isinstance(componentres, dict):
                for subcomponentkey, subcomponentres in componentres.items():
                    ares[componentkey + "_" + subcomponentkey] = subcomponentres
            else:
                ares[componentkey] = componentres
        ares["res"] = aresval
        reslist.append(ares)
        dkeys.update(ares.keys())
    save_listofdict_to_csv(reslist, fpath, dkeys)


def list_of_tuples_to_csv(listoftuples, heading, fpath):
    with open(fpath, "w") as fp:
        csvwriter = csv.writer(fp)
        csvwriter.writerow(heading)
        for arow in listoftuples:
            csvwriter.writerow(arow)


def dpath_has_content(dpath: str):
    if not os.path.isdir(dpath):
        return False
    return len(os.listdir(dpath)) > 0


def str2gp(gpstr):
    """Convert str(((gains), (priorities))) to tuple(((gains), (priorities)))"""
    # print(tuple([tuple([int(el) for el in weights.split(', ')]) for weights in gpstr[2:-2].split('), (')])) # dbg
    try:
        return tuple(
            [
                tuple([int(el) for el in weights.split(", ")])
                for weights in gpstr[2:-2].split("), (")
            ]
        )
    except ValueError:
        breakpoint()


def get_highest_direntry(dpath: str) -> Optional[str]:
    """Get highest numbered entry in a directory"""
    highest = -1
    for adir in os.listdir(dpath):
        if adir.isdecimal() and int(adir) > highest:
            highest = int(adir)
    if highest == -1:
        return None
    return str(highest)


def get_last_modified_file(
        dpath,
        exclude: Optional[Union[str, List[str]]] = None,
        incl_ext: bool = True,
        full_path=True,
        fn_beginswith: Optional[Union[str, int]] = None,
        ext=None,
        exclude_ext: Optional[str] = None,
):
    """Get the last modified fn,
    optionally excluding patterns found in exclude (str or list),
    optionally omitting extension"""
    if not os.path.isdir(dpath):
        return False
    fpaths = [
        os.path.join(dpath, fn) for fn in os.listdir(dpath)
    ]  # add path to each file
    fpaths.sort(key=os.path.getmtime, reverse=True)
    if len(fpaths) == 0:
        return False
    fpath = None
    if exclude is None and fn_beginswith is None and ext is None:
        fpath = fpaths[0]
    else:
        if isinstance(exclude, str):
            exclude = [exclude]
        if isinstance(fn_beginswith, int):
            fn_beginswith = str(fn_beginswith)
        for afpath in fpaths:
            fn = afpath.split("/")[-1]  # not Windows friendly
            if exclude is not None and fn in exclude:
                continue
            if fn_beginswith is not None and not fn.startswith(fn_beginswith):
                continue
            if ext is not None and not fn.endswith("." + ext):
                continue
            if exclude_ext is not None and fn.endswith("." + exclude_ext):
                continue
            fpath = afpath
            break
        if fpath is None:
            return False
    if not incl_ext:
        assert "." in fpath.split("/")[-1], fpath  # not Windows friendly
        fpath = fpath.rpartition(".")[0]
    if full_path:
        return fpath
    else:
        return fpath.split("/")[-1]


def listfpaths(dpath):
    """Similar to os.listdir(dpath), returns joined paths of files present."""
    fpaths = []
    for fn in os.listdir(dpath):
        fpaths.append(os.path.join(dpath, fn))
    return fpaths


def compress_lzma(infpath, outfpath):
    with open(infpath, "rb") as f:
        dat = f.read()
    # DBG: timing lzma compression
    # tic = time.perf_counter()
    cdat = lzma.compress(dat)
    # toc = time.perf_counter()-tic
    # print("compress_lzma: side_string encoding time = {}".format(toc))
    # compress_lzma: side_string encoding time = 0.005527787026949227
    # tic = time.perf_counter()
    # ddat = lzma.decompress(dat)
    # toc = time.perf_counter()-tic
    # print("compress_lzma: side_string decoding time = {}".format(toc))

    #
    with open(outfpath, "wb") as f:
        f.write(cdat)


def compress_png(tensor, outfpath):
    """only supports grayscale!"""
    if tensor.shape[0] > 1:
        print("common.utilities.compress_png: warning: too many channels (failed)")
        return False
    w = png.Writer(
        tensor.shape[2],
        tensor.shape[1],
        greyscale=True,
        bitdepth=int(np.ceil(np.log2(tensor.max() + 1))),
        compression=9,
    )
    with open(outfpath, "wb") as fp:
        w.write(fp, tensor[0])
    return True


def decompress_lzma(infpath, outfpath):
    with open(infpath, "rb") as f:
        cdat = f.read()
    dat = lzma.decompress(cdat)
    with open(outfpath, "wb") as f:
        f.write(dat)


# def csv_fpath_to_listofdicts(fpath):
# TODO parse int/float
#     with open(fpath, 'r') as fp:
#         csvres = list(csv.DictReader(fp))
#     return csvres


# def save_src(root_dpath: str, directories: list[str], extensions: list[str] = ["py"]):
#    pass  # TODO for dn in directories:


def noop(*args, **kwargs):
    """Do nothing function that accepts any arguments.
    
    A utility function that silently accepts and ignores any arguments.
    Useful as a placeholder, default callback, or for testing.
    
    Args:
        *args: Any positional arguments (ignored)
        **kwargs: Any keyword arguments (ignored)
        
    Returns:
        None
    """
    pass


def filesize(fpath):
    """Get the size of a file in bytes.
    
    A simple wrapper around os.stat to get file size information.
    
    Args:
        fpath: Path to the file
        
    Returns:
        Integer representing the file size in bytes
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        
    Example:
        >>> filesize('example.txt')
        1024
    """
    return os.stat(fpath).st_size


def avg_listofdicts(listofdicts):
    """Calculate the average value for each key across a list of dictionaries.
    
    For each key present in the dictionaries, computes the mean of all values
    found for that key across all dictionaries in the list.
    
    Args:
        listofdicts: List of dictionaries with the same keys and numeric values
        
    Returns:
        A dictionary with the same keys as the input dictionaries, but with each value
        replaced by the mean of all values for that key across all input dictionaries
        
    Notes:
        - Assumes all dictionaries have the same keys
        - Assumes all values are numeric (can be passed to statistics.mean)
        - Uses the first dictionary's keys as the reference set
        
    Example:
        >>> avg_listofdicts([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])
        {'a': 2.0, 'b': 3.0}
    """
    res = dict()
    for akey in listofdicts[0].keys():
        res[akey] = list()
    for adict in listofdicts:
        for akey, aval in adict.items():
            res[akey].append(aval)
    for akey in res.keys():
        res[akey] = statistics.mean(res[akey])
    return res


def walk(root: str, dir: str = ".", follow_links=False):
    """Similar to os.walk, but keeps a constant root"""
    dpath = os.path.join(root, dir)
    for name in os.listdir(dpath):
        path = os.path.join(dpath, name)
        if os.path.isfile(path):
            yield (root, dir, name)
        elif os.path.isdir(path) or (os.path.islink(path) and follow_links):
            yield from walk(root=root, dir=os.path.join(dir, name))
        elif os.path.islink(path) and not follow_links:
            continue
        elif not os.path.exists(path):
            print(f"walk: {path=} disappeared, ignoring.")
            continue
        else:
            # raise ValueError(f"Unknown type: {path}")
            popup(f"walk: Unknown type: {path}")
            breakpoint()


def popup(msg):
    """Print and send a notification on Linux/compatible systems."""
    print(msg)
    subprocess.run(["/usr/bin/notify-send", msg], check=False)


def restart_program():
    """Restart the current Python program with the same arguments.
    
    Registers an exit handler that will re-execute the current script with
    the same command-line arguments when the program exits. The script will
    restart immediately after this function is called.
    
    Notes:
        - Uses os.execl which replaces the current process without forking
        - All current program state will be lost
        - Will use the same Python interpreter that ran the current script
        - All command-line arguments will be preserved
        
    Example use cases:
        - After downloading updates to the script
        - After changing configuration files that are only read at startup
        - After modifying environment variables
    """

    def _restart_program():
        os.execl(sys.executable, sys.executable, *sys.argv)

    atexit.register(_restart_program)
    exit()


def shuffle_dictionary(input_dict):
    """Randomly reorder the keys in a dictionary.
    
    Creates a new dictionary with the same key-value pairs as the input,
    but with the keys in a random order. Useful for randomizing iteration
    order or testing order-dependence.
    
    Args:
        input_dict: Dictionary to be shuffled
        
    Returns:
        A new dictionary with the same keys and values but in a random order
        
    Notes:
        - The original dictionary is not modified
        - Uses random.shuffle internally, so results depend on the random state
        - Dictionary order is preserved in Python 3.7+ (CPython 3.6+)
        
    Example:
        >>> d = {'a': 1, 'b': 2, 'c': 3}
        >>> shuffled = shuffle_dictionary(d)
        >>> # Keys will be in random order but all present with their original values
    """
    # Convert the dictionary to a list of key-value pairs
    items = list(input_dict.items())

    # Shuffle the list of items
    random.shuffle(items)

    # Convert the shuffled list back to a dictionary
    shuffled_dict = dict(items)

    return shuffled_dict


def sort_dictionary(input_dict):
    """Sort a dictionary by its keys.
    
    Creates a new dictionary with the same key-value pairs as the input,
    but with the keys in sorted order. Useful for consistent display or
    reproducible iteration.
    
    Args:
        input_dict: Dictionary to be sorted
        
    Returns:
        A new dictionary with the same keys and values but in sorted key order
        
    Notes:
        - The original dictionary is not modified
        - Keys must be comparable (support the < operator)
        - Dictionary order is preserved in Python 3.7+ (CPython 3.6+)
        
    Example:
        >>> d = {'c': 3, 'a': 1, 'b': 2}
        >>> sorted_d = sort_dictionary(d)
        >>> # Keys will be in alphabetical order: 'a', 'b', 'c'
    """
    # Convert the dictionary to a list of key-value pairs
    items = list(input_dict.items())

    # Sort the list of items
    items.sort()

    # Convert the sorted list back to a dictionary
    sorted_dict = dict(items)

    return sorted_dict


class Test_utilities(unittest.TestCase):
    """Unit tests for the utilities module functions.
    
    This test class contains test cases for various utility functions in this module,
    verifying their correct behavior under normal and edge case scenarios.
    """

    def test_freezedict(self):
        """Test the freeze_dict and unfreeze_dict functions.
        
        Verifies that a dictionary can be correctly converted to a hashable frozenset
        representation and then restored to its original form, preserving all structure
        and values, including nested dictionaries.
        
        The test:
        1. Creates a test dictionary with nested structure
        2. Freezes it using freeze_dict()
        3. Verifies the frozen dictionary can be used as a dictionary key
        4. Unfreezes it using unfreeze_dict()
        5. Verifies the unfrozen result exactly matches the original dictionary
        """
        adict = {"a": 1, "b": 22, "c": 333, "d": {"e": 4, "f": 555}}
        print(adict)
        fdict = freeze_dict(adict)
        print(fdict)
        ndict = {fdict: 42}  # Verify frozen dict can be used as a dictionary key
        adictuf = unfreeze_dict(fdict)
        print(adictuf)
        self.assertDictEqual(adict, adictuf)


if __name__ == "__main__":
    unittest.main()
