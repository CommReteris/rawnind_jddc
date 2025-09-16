"""Neural network models and architectures for raw image processing.

This package contains implementations of various neural network models and
traditional algorithms for raw image processing tasks, particularly:
    - Denoising: Removing noise from raw camera sensor data
    - Demosaicing: Converting Bayer/X-Trans patterns to RGB images
    - Compression: Reducing file size while preserving image quality

Available models:
    - raw_denoiser: Neural networks for raw image denoising (UtNet2/UtNet3)
    - bm3d_denoiser: Implementation of BM3D denoising algorithm
    - compression_autoencoders: Autoencoder-based neural compression models
    - denoise_then_compress: Pipeline combining denoising and compression steps
    - manynets_compression: ManyPriors approach to image compression
    - bitEstimator: Entropy models for compression
    - standard_compressor: Wrappers for standard compression algorithms (JPEG, JPEG XL, etc.)

Individual models can be imported directly:
    from rawnind.models import raw_denoiser, compression_autoencoders
    
Or specific classes can be imported:
    from rawnind.models.raw_denoiser import UtNet2
    from rawnind.models.compression_autoencoders import BalleEncoder
"""

# Import statements are commented out to avoid eager loading of dependencies
# Import specific models as needed in your code
# from . import raw_denoiser