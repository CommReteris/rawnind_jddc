# -*- coding: utf-8 -*-
"""Standard compression methods handlers.

This module provides a unified interface for various image compression methods,
allowing consistent access to different compression algorithms through a common API.
It wraps external binary executables for compression and decompression operations.

Key features:
- Abstract base class (StdCompression) with common compression/decompression pipeline
- Specialized implementations for multiple compression formats:
  - JPEG (JPG_Compression)
  - JPEG XL (JPEGXL_Compression)
  - BPG (Better Portable Graphics) (BPG_Compression)
  - JPEG XS (JPEGXS_Compression)
- Single file and batch directory processing
- Multi-threaded compression operations
- Parameter validation and command-line generation
- Temporary file management
- Consistent API across compression methods

Usage examples:
    # Compress a single file with JPEG
    JPG_Compression.file_encdec(
        infpath="input.png", 
        outfpath="output.jpg", 
        quality=75
    )
    
    # Compress all files in a directory with BPG
    BPG_Compression.dir_encdec(
        indpath="input_dir", 
        outdpath="output_dir",
        quality=30, 
        chroma_ss=444
    )
"""

import os
from typing import List, Any, Optional
import shutil
import unittest
import subprocess
import inspect
import time
import sys

<<<<<<< HEAD
from . import utilities
=======
from . import numpy_operations
>>>>>>> 9d829208844a9450effb8f515b5521749b6aed0c

# Number of threads to use for parallel compression operations
NUMTHREADS: int = 1  # Default to 1, can be set to os.cpu_count()//4*3 for parallel processing

# Chroma subsampling modes (444 = no subsampling)
CHROMA_SS: List[int] = [444]

# Valid compression parameters for all compression methods
VALID_ARGS: list = ["quality", "chroma_ss", "bitrate", "weights", "profile"]

# Extensions for lossless image formats used as output formats
LOSSLESS_IMGEXT = ["png", "ppm"]


class StdCompression:
    """Base class for standard compression methods.
    
    This abstract class defines the interface and common functionality for all
    compression method implementations. Subclasses must define specific class
    attributes and implement required methods for their compression algorithm.
    
    Required class attributes for subclasses:
        ENCBIN (str): Name of the encoding binary executable
        DECBIN (str): Name of the decoding binary executable (if different from ENCBIN)
        BINARY (str): Primary binary name (usually same as ENCBIN)
        ENCEXT (str): File extension for the compressed format (e.g., "jpg", "jxl")
        REQ_DEC (bool): Whether decompression to a lossless format is required
        
    The class provides methods for:
    - Creating temporary file paths
    - Executing encoding and decoding operations
    - Processing directories of images
    - Multi-threaded batch processing
    - Generating consistent naming conventions for compressed outputs
    """

    def __init__(self):
        """Initialize the compression handler and verify binary executable exists.
        
        Raises:
            AssertionError: If the required binary executable is not found in PATH
        """
        assert shutil.which(self.BINARY) is not None, "Missing {} binary".format(
            self.ENCBIN
        )

    # Commented out unused method
    #    @classmethod:
    #    def make_comp_fpath(cls, infpath: str, compfpath: str):
    #        breakpoint()

    @classmethod
    def make_tmp_fpath(cls, outfpath: str, tmpfpath: Optional[str] = None):
        """Determine the appropriate temporary file path for compression.
    
        This method handles the creation of temporary file paths during the
        compression process, accounting for whether the compression method
        requires a separate decompression step.
    
        Args:
            outfpath: Destination path for the final output file
            tmpfpath: Optional explicit temporary file path
        
        Returns:
            str: Path to use for the compressed temporary file
        
        Notes:
            - For methods that don't require decompression (REQ_DEC=False),
              the output path is used directly
            - For methods requiring decompression, a temporary path with the
              compression format's extension is created if not provided
        """
        if not cls.REQ_DEC:
            assert tmpfpath is None or outfpath == tmpfpath
            return outfpath
        if tmpfpath is not None:
            return tmpfpath
        if cls.REQ_DEC:
            return outfpath + "." + cls.ENCEXT

    @classmethod
    def file_encdec(
            cls,
            infpath: str,
            outfpath: Optional[str],
            tmpfpath: Optional[str] = None,
            cleanup: bool = True,
            overwrite: bool = False,
            **kwargs,
    ):
        """Compress a single image file using the specified compression method.
    
        This method handles the full compression pipeline:
        1. Determines appropriate temporary file path
        2. Validates parameters and binary availability
        3. Runs the encoding command
        4. Runs the decoding command (if required)
        5. Cleans up temporary files (if requested)
        6. Collects and returns performance metrics
    
        Args:
            infpath: Path to the input image file
            outfpath: Path where the output should be saved
            tmpfpath: Optional path for temporary compressed file
            cleanup: Whether to remove temporary files after processing
            overwrite: Whether to overwrite existing files
            **kwargs: Compression parameters (quality, bitrate, etc.)
        
        Returns:
            dict: Dictionary containing:
                - infpath: Input file path
                - outfpath: Output file path
                - tmpfpath: Temporary file path
                - enctime: Encoding time (seconds)
                - encsize: Size of encoded file (bytes)
                - dectime: Decoding time (seconds) if applicable
            
        Raises:
            AssertionError: If parameters are invalid or binary is missing
        """
        assert outfpath is not None
        tmpfpath = cls.make_tmp_fpath(outfpath, tmpfpath)
        returnvals = {"infpath": infpath, "outfpath": outfpath, "tmpfpath": tmpfpath}

        # Validate compression parameters and requirements
        assert set.issubset(set(kwargs.keys()), set(VALID_ARGS))
        assert shutil.which(cls.ENCBIN) is not None, "Missing {} binary".format(
            cls.ENCBIN
        )
        assert tmpfpath.endswith(cls.ENCEXT), tmpfpath
        assert outfpath.split(".")[-1] in LOSSLESS_IMGEXT or not cls.REQ_DEC, outfpath

        # Perform encoding
        cmd = cls.make_enc_cl(infpath, tmpfpath, **kwargs)
        if not os.path.isfile(tmpfpath) or overwrite:
            start_time = time.time()
            subprocess.run(cmd)
            returnvals["enctime"] = time.time() - start_time
        assert os.path.isfile(tmpfpath), f"{tmpfpath=}, {' '.join(cmd)}"
        returnvals["encsize"] = os.path.getsize(tmpfpath)

        # Perform decoding if needed
        if tmpfpath != outfpath:
            cmd = cls.make_dec_cl(tmpfpath, outfpath, **kwargs)
            if not os.path.isfile(outfpath) or overwrite:
                start_time = time.time()
                subprocess.run(cmd)
                returnvals["dectime"] = time.time() - start_time
            if cleanup:
                os.remove(tmpfpath)

        return returnvals

    @classmethod
    def make_cname(cls, **kwargs):
        """Generate a consistent name for a compression configuration.
    
        Creates a standardized string representation of a compression configuration
        that can be used for directory naming or identification.
    
        Args:
            **kwargs: Compression parameters (quality, bitrate, etc.)
        
        Returns:
            str: Formatted name combining compression class and parameters
        """
        cname: str = cls.__name__ + str(kwargs)
        cname = cname.replace(" ", "_")
        cname = cname.replace("'", "")
        return cname

    @classmethod
    def file_encdec_mtrunner(cls, kwargs):
        """Wrapper method for multi-threaded compression operations.
    
        This helper method unpacks a dictionary of arguments and passes them to
        the file_encdec method, making it suitable for use with numpy_operations.mt_runner.
    
        Args:
            kwargs: Dictionary of arguments for file_encdec
        
        Returns:
            dict: Result dictionary from file_encdec
        """
        return cls.file_encdec(**kwargs)

    @classmethod
    def dir_encdec(
            cls, indpath: str, outdpath: str, cleanup=True, overwrite=False, **kwargs
    ):
        """Compress all images in a directory using the specified compression method.
    
        This method processes all files in the input directory:
        1. Creates appropriate output directory structure
        2. Determines output file paths and extensions
        3. Prepares compression parameters for each file
        4. Runs compression operations in parallel using multiple threads
    
        Args:
            indpath: Path to input directory containing images
            outdpath: Path to output directory (or None to auto-generate)
            cleanup: Whether to remove temporary files after processing
            overwrite: Whether to overwrite existing output files
            **kwargs: Compression parameters (quality, bitrate, etc.)
        
        Returns:
            str: Path to the output directory
        """
        imgs = os.listdir(indpath)
        args = []

        # Create output directory path if not provided
        if outdpath is None:
            dsname = numpy_operations.get_leaf(indpath)
            outdpath = os.path.join(
                indpath, "compressed", cls.make_cname(kwargs), dsname
            )
        os.makedirs(outdpath, exist_ok=True)

        # Prepare compression arguments for each file
        for fn in imgs:
            outfpath = os.path.join(outdpath, fn)

            # Handle file extensions based on compression method
            if not cls.REQ_DEC:
                if outfpath.split(".")[-1] != cls.ENCEXT:
                    outfpath = outfpath + "." + cls.ENCEXT
            elif outfpath.split(".")[-1] not in LOSSLESS_IMGEXT:
                outfpath = outfpath + "." + LOSSLESS_IMGEXT[0]

            # Skip existing files unless overwrite is requested
            if os.path.isfile(outfpath) and not overwrite:
                continue

            args.append(
                {
                    "infpath" : os.path.join(indpath, fn),
                    "outfpath": outfpath,
                    "cleanup" : cleanup,
                    **kwargs,
                }
            )

        # Process all files in parallel
        numpy_operations.mt_runner(
            cls.file_encdec_mtrunner, args, num_threads=NUMTHREADS, ordered=False
        )

        return outdpath


class JPG_Compression(StdCompression):
    """JPEG image compression using GraphicsMagick.
    
    This class implements standard JPEG compression using the GraphicsMagick 'gm'
    command-line tool. JPEG is a widely supported lossy compression format suitable
    for photographic images.
    
    Attributes:
        ENCBIN: Encoder binary name ('gm')
        BINARY: Primary binary name (same as ENCBIN)
        ENCEXT: File extension for compressed files ('jpg')
        REQ_DEC: Whether decompression is required (False - JPEG files can be used directly)
        QUALITY_RANGE: Range of valid quality values (1-100, higher is better quality)
    """
    ENCBIN = BINARY = "gm"
    ENCEXT: str = "jpg"
    REQ_DEC: bool = False
    QUALITY_RANGE = (1, 100 + 1)  # 1-100, higher = better quality

    @classmethod
    def make_enc_cl(cls, infpath: str, outfpath: str, quality: Any, **kwargs) -> list:
        """Create the command line for JPEG encoding.
        
        Builds a GraphicsMagick command that converts an input image to JPEG format
        with the specified quality setting.
        
        Args:
            infpath: Path to the input image file
            outfpath: Path where the JPEG file should be saved
            quality: JPEG quality setting (1-100, higher is better quality)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for GraphicsMagick
            
        Raises:
            AssertionError: If quality parameter is missing
        """
        assert quality is not None
        return [
            JPG_Compression.BINARY,
            "convert",
            infpath,
            "-strip",  # Remove metadata to reduce file size
            "-quality",
            "{}%".format(quality),
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        """Create the command line for JPEG decoding.
        
        Builds a GraphicsMagick command that converts a JPEG file to another format.
        
        Args:
            infpath: Path to the input JPEG file
            outfpath: Path where the decoded image should be saved
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for GraphicsMagick
        """
        return [JPG_Compression.BINARY, "convert", infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        """Generate all valid compression argument combinations.
        
        Yields all valid quality settings for JPEG compression, producing
        a series of parameter dictionaries suitable for file_encdec.
        
        Yields:
            dict: Compression parameters with 'quality' key
        """
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class JPEGXL_Compression(StdCompression):
    """JPEG XL image compression.
    
    This class implements JPEG XL compression, a modern image format designed
    for high compression efficiency, superior quality, and fast encoding/decoding.
    JPEG XL supports both lossy and lossless compression modes.
    
    This implementation uses the reference cjxl/djxl command-line tools.
    
    Attributes:
        ENCBIN: Encoder binary name ('cjxl')
        BINARY: Primary binary name (same as ENCBIN)
        DECBIN: Decoder binary name ('djxl')
        ENCEXT: File extension for compressed files ('jxl')
        REQ_DEC: Whether decompression is required (True - JXL files need decoding)
        QUALITY_RANGE: Range of valid quality values (1-100, higher is better quality)
    """
    ENCBIN = BINARY = "cjxl"
    DECBIN = "djxl"
    ENCEXT: str = "jxl"
    REQ_DEC: bool = True
    QUALITY_RANGE = (1, 100 + 1)  # 1-100, higher = better quality

    @classmethod
    def make_enc_cl(cls, infpath: str, outfpath: str, quality: Any, **kwargs) -> list:
        """Create the command line for JPEG XL encoding.
        
        Builds a command that converts an input image to JPEG XL format
        with the specified quality setting.
        
        Args:
            infpath: Path to the input image file
            outfpath: Path where the JPEG XL file should be saved
            quality: JPEG XL quality setting (1-100, higher is better quality)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for cjxl
            
        Raises:
            AssertionError: If quality parameter is missing
        """
        assert quality is not None
        return [
            cls.BINARY,
            "--quality",  # Set the compression quality
            str(quality),
            infpath,
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        """Create the command line for JPEG XL decoding.
        
        Builds a command that converts a JPEG XL file to another format
        using the djxl decoder.
        
        Args:
            infpath: Path to the input JPEG XL file
            outfpath: Path where the decoded image should be saved
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for djxl
        """
        return [cls.DECBIN, infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        """Generate all valid compression argument combinations.
        
        Yields all valid quality settings for JPEG XL compression, producing
        a series of parameter dictionaries suitable for file_encdec.
        
        Yields:
            dict: Compression parameters with 'quality' key
        """
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class BPG_Compression(StdCompression):
    """Better Portable Graphics (BPG) image compression.
    
    This class implements BPG compression, a modern format based on HEVC/H.265
    video compression technology. BPG provides better compression efficiency
    than JPEG at similar visual quality, especially at low bitrates.
    
    This implementation uses the reference bpgenc/bpgdec command-line tools.
    
    Attributes:
        ENCBIN: Encoder binary name ('bpgenc')
        DECBIN: Decoder binary name ('bpgdec')
        ENCEXT: File extension for compressed files ('bpg')
        REQ_DEC: Whether decompression is required (True - BPG files need decoding)
        QUALITY_RANGE: Range of valid quality values (0-51, lower is better quality)
    """
    ENCBIN: str = "bpgenc"
    DECBIN: str = "bpgdec"
    ENCEXT: str = "bpg"
    REQ_DEC: bool = True
    QUALITY_RANGE = (0, 51 + 1)  # 0-51, lower = better quality (unlike JPEG)

    @classmethod
    def make_enc_cl(
            cls,
            infpath: str,
            outfpath: str,
            quality: Any,
            chroma_ss: Any = CHROMA_SS[0],
            **kwargs,
    ) -> list:
        """Create the command line for BPG encoding.
        
        Builds a command that converts an input image to BPG format
        with the specified quality and chroma subsampling settings.
        
        Args:
            infpath: Path to the input image file
            outfpath: Path where the BPG file should be saved
            quality: BPG quality setting (0-51, lower is better quality)
            chroma_ss: Chroma subsampling mode (e.g., 444 for no subsampling)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for bpgenc
            
        Raises:
            AssertionError: If quality parameter is missing
        """
        assert quality is not None
        return [
            BPG_Compression.ENCBIN,
            "-q",  # Quality parameter
            str(quality),
            "-f",  # Format (chroma subsampling)
            str(chroma_ss),
            "-o",  # Output file
            outfpath,
            infpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        """Create the command line for BPG decoding.
        
        Builds a command that converts a BPG file to another format
        using the bpgdec decoder.
        
        Args:
            infpath: Path to the input BPG file
            outfpath: Path where the decoded image should be saved
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for bpgdec
        """
        return [BPG_Compression.DECBIN, infpath, "-o", outfpath]

    @classmethod
    def get_valid_cargs(cls):
        """Generate all valid compression argument combinations.
        
        Yields all valid quality settings for BPG compression, producing
        a series of parameter dictionaries suitable for file_encdec.
        
        Note:
            Unlike JPEG, lower quality values in BPG result in better quality
            at the expense of larger file sizes.
            
        Yields:
            dict: Compression parameters with 'quality' key
        """
        for quality in range(*cls.QUALITY_RANGE):
            yield {"quality": quality}


class JPEGXS_Compression(StdCompression):
    """JPEG XS image compression.
    
    This class implements JPEG XS compression, a low-latency, visually lossless
    image compression standard designed for professional video applications,
    broadcast environments, and other scenarios requiring low encoding/decoding
    complexity with high visual quality.
    
    This implementation uses the Transcodium (tco) encoder/decoder tools.
    
    Attributes:
        ENCBIN: Encoder binary name ('tco_encoder')
        DECBIN: Decoder binary name ('tco_decoder')
        ENCEXT: File extension for compressed files ('tco')
        REQ_DEC: Whether decompression is required (True - JPEG XS files need decoding)
        PROFILE: Default compression profile (11)
        WEIGHTS: Available weighting strategies for rate control ('psnr' or 'visual')
        BITRATE_RANGE: Range of valid bitrates (start, end, step)
    """
    ENCBIN: str = "tco_encoder"
    DECBIN: str = "tco_decoder"
    ENCEXT: str = "tco"
    REQ_DEC: bool = True
    PROFILE = 11  # Default profile for JPEG XS compression
    WEIGHTS = ["psnr", "visual"]  # Rate control optimization targets
    BITRATE_RANGE = (0.36, 1.51, 0.01)  # (start, end, step) in bits per pixel

    @classmethod
    def make_enc_cl(
            cls,
            infpath: str,
            outfpath: str,
            bitrate: Any,
            profile=PROFILE,
            weights: str = WEIGHTS[0],
            **kwargs,
    ) -> list:
        """Create the command line for JPEG XS encoding.
        
        Builds a command that converts an input image to JPEG XS format
        with the specified bitrate, profile, and optimization weights.
        
        Args:
            infpath: Path to the input image file
            outfpath: Path where the JPEG XS file should be saved
            bitrate: Target bitrate in bits per pixel (bpp)
            profile: Compression profile (default: 11)
            weights: Rate control optimization target ('psnr' or 'visual')
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for tco_encoder
            
        Raises:
            AssertionError: If bitrate parameter is missing
        """
        assert bitrate is not None
        return [
            cls.ENCBIN,
            "-b",  # Bitrate parameter
            str(bitrate),
            "-p",  # Profile parameter
            str(profile),
            "-o",  # Optimization weights
            weights,
            infpath,
            outfpath,
        ]

    @classmethod
    def make_dec_cl(cls, infpath: str, outfpath: str, **kwargs) -> list:
        """Create the command line for JPEG XS decoding.
        
        Builds a command that converts a JPEG XS file to another format
        using the tco_decoder.
        
        Args:
            infpath: Path to the input JPEG XS file
            outfpath: Path where the decoded image should be saved
            **kwargs: Additional arguments (ignored)
            
        Returns:
            list: Command line arguments for tco_decoder
        """
        return [cls.DECBIN, infpath, outfpath]

    @classmethod
    def get_valid_cargs(cls):
        """Generate all valid compression argument combinations.
        
        Yields combinations of bitrate and weights settings for JPEG XS compression,
        producing a series of parameter dictionaries suitable for file_encdec.
        
        The method iterates through the bitrate range defined in BITRATE_RANGE
        and combines each bitrate with each available weighting strategy.
        
        Yields:
            dict: Compression parameters with 'bitrate', 'weights', and 'profile' keys
        """
        bitrate = cls.BITRATE_RANGE[0]
        while bitrate < cls.BITRATE_RANGE[1]:
            for weights in cls.WEIGHTS:
                yield {"bitrate": bitrate, "weights": weights, "profile": 11}
            bitrate += cls.BITRATE_RANGE[2]


# Automatically generate a list of all compression classes in the module
# This uses introspection to find all classes and then removes the base class
COMPRESSIONS = [
    acls[0] for acls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
]
COMPRESSIONS.remove("StdCompression")


class Test_numpy_operations(unittest.TestCase):
    """Test cases for compression numpy_operations.
    
    This class contains unit tests that verify the functionality of the
    compression classes using the Kodak test image dataset.
    """

    def test_compress_kodak_bpg(self):
        """Test BPG compression on the Kodak dataset.
        
        This test:
        1. Defines paths to input and output directories
        2. Creates an output directory for compressed files
        3. Runs BPG compression on all images in the Kodak dataset
        4. Verifies that output files were created
        """
        indpath = os.path.join("..", "..", "datasets", "test", "kodak")
        cname = BPG_Compression.make_cname(quality=50)
        outdpath = os.path.join(
            "..", "..", "datasets", "test", "compressed", cname, "kodak"
        )
        os.makedirs(outdpath, exist_ok=True)
        BPG_Compression.dir_encdec(
            indpath=indpath, outdpath=outdpath, quality=50, cleanup=True
        )
        # Verify that files were created in the output directory
        self.assertGreater(len(os.listdir(outdpath)), 0)

    def test_compress_kodak_jpg(self):
        """Test JPEG compression on the Kodak dataset.
        
        This test:
        1. Defines paths to input and output directories
        2. Creates an output directory for compressed files
        3. Runs JPEG compression on all images in the Kodak dataset
        4. Verifies that output files were created
        """
        indpath = os.path.join("..", "..", "datasets", "test", "kodak")
        cname = JPG_Compression.make_cname(quality=50)
        outdpath = os.path.join(
            "..", "..", "datasets", "test", "compressed", cname, "kodak"
        )
        os.makedirs(outdpath, exist_ok=True)
        JPG_Compression.dir_encdec(indpath=indpath, outdpath=outdpath, quality=50)
        # Verify that files were created in the output directory
        self.assertGreater(len(os.listdir(outdpath)), 0)


if __name__ == "__main__":
    # Run all unit tests when this file is executed directly
    unittest.main()
