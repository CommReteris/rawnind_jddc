"""Test suite for RawNIND project.

This package contains test scripts for validating and evaluating the various
components of the RawNIND project, including:

- Model evaluation scripts for denoising and compression models
- Data processing pipeline validation
- Performance benchmarking
- Quality assessment of image processing results

The test files are organized by test type and functionality:

1. Model Testing:
   - test_playraw_*.py: Tests for "playraw" (clean-clean/unpaired) image processing
   - test_manproc_*.py: Tests for manually processed image handling
   - test_progressive_*.py: Tests for progressive/incremental model improvements

2. Input/Output Type Testing:
   - *_bayer2prgb.py: Tests for Bayer pattern to profiled RGB conversion
   - *_prgb2prgb.py: Tests for profiled RGB to profiled RGB processing
   - *_proc2proc.py: Tests for processor to processor conversion

3. Task Testing:
   - *_dc_*.py: Tests for joint denoising and compression models
   - *_denoise_*.py: Tests for pure denoising models

4. Utility Scripts:
   - rawtestlib.py: Common test utilities and helper classes
   - get_*.py: Analysis and metrics computation scripts

"""


__all__ = ['rawtestlib']
