"""
Model complexity analysis for denoising and compression architectures.

This script evaluates the computational complexity of various neural network models
for image denoising and compression. It uses the ptflops library to compute:
1. MACs (multiply-accumulate operations) - a measure of computational complexity
2. Parameters - a measure of model size

The analysis covers different model architectures:
- JDDC (Joint Denoising and Compression)
- DenoiseThenCompress (sequential pipeline)
- U-Net (standalone denoiser)

Each model is tested with both Bayer pattern (4-channel) and RGB (3-channel) inputs
at appropriate resolutions to represent typical use cases.

Usage:
    python get_models_complexity.py

Output:
    Prints model names with their corresponding MACs and parameter counts.
"""

import sys
import torch
import ptflops

sys.path.append("..")
from rawnind.libs import abstract_trainer
from rawnind.models import raw_denoiser
from rawnind.models import manynets_compression
from rawnind.models import denoise_then_compress
from rawnind.models import compression_autoencoders
# NOTE: This import appears to be problematic - potentially unavailable or a typo
# Keeping it commented to preserve reference while resolving the semantic error
# from nind_denoise.networks import UtNet

if __name__ == "__main__":
    # Define input dimensions for testing:
    # - 4-channel Bayer input (512x512 resolution)
    # - 3-channel RGB input (1024x1024 resolution, equivalent to debayered 512x512)
    all_megapix_dims = ((4, 512, 512), (3, 1024, 1024))
    
    # Define models to test, organized by input channel count
    models_to_test = {
        # 4-channel input models (Bayer pattern)
        4: {
            # JDDC model with BayerPSDecoder for direct Bayer processing
            "JDDC (Bayer input)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=4,
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BayerPSDecoder,  # Special decoder for Bayer
                device=torch.device("cpu"),
            ),
            
            # Sequential pipeline that first denoises then compresses
            "DenoiseThenCompress (Bayer input)": denoise_then_compress.DenoiseThenCompress(
                in_channels=4,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            
            # JDDC variant that upsamples Bayer input before processing
            "JDDC (Pre-upsample)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=4,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
                preupsample=True,  # Upsamples before processing
            ),
            
            # Standard U-Net denoiser for Bayer input
            "U-Net (Bayer input)": raw_denoiser.UtNet2(in_channels=4, funit=32),
        },
        
        # 3-channel input models (RGB)
        3: {
            # Compression model for RGB input
            "Compression or JDC (RGB input)": manynets_compression.ManyPriors_RawImageCompressor(
                in_channels=3,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            
            # Sequential pipeline for RGB input
            "DenoiseThenCompress (RGB input)": denoise_then_compress.DenoiseThenCompress(
                in_channels=3,
                device=torch.device("cpu"),
                encoder_cls=compression_autoencoders.BalleEncoder,
                decoder_cls=compression_autoencoders.BalleDecoder,
            ),
            
            # Standard U-Net denoiser for RGB input
            "U-Net (RGB input)": raw_denoiser.UtNet2(in_channels=3, funit=32),
            
            # Higher capacity U-Net with double the base feature channels
            "U-Net (RGB input, full channels)": raw_denoiser.UtNet2(
                in_channels=3, funit=64  # Double the feature channels
            ),
            
            # Legacy model reference (commented out due to import issue)
            # "oldunet": UtNet.UtNet(),
        },
    }
    
    # Run complexity analysis for each model with appropriate input dimensions
    for megapix_dims in all_megapix_dims:
        for model_name, model in models_to_test[megapix_dims[0]].items():
            print(f"Model: {model_name}")
            # Use ptflops to calculate MACs and parameters
            macs, params = ptflops.get_model_complexity_info(
                model,
                megapix_dims,
            )
            print(f"{model_name} macs: {macs}, params: {params}")

    # ALTERNATIVE TESTING APPROACH (COMMENTED OUT)
    # The following code was an alternative approach for testing different
    # U-Net configurations with varying feature channel multipliers.
    # It has been commented out but preserved for reference.
    
    # # Test different U-Net capacities with Bayer input
    # print("Bayer complexity")
    # megapixel_dims = 4, 512, 512
    # models = {
    #     "f32model": raw_denoiser.UtNet2(in_channels=4, funit=32).eval(),  # Base capacity
    #     "f48model": raw_denoiser.UtNet2(in_channels=4, funit=48).eval(),  # 1.5x capacity
    #     "f64model": raw_denoiser.UtNet2(in_channels=4, funit=64).eval(),  # 2x capacity
    #     "fUtNet3model": raw_denoiser.UtNet3(in_channels=4).eval(),        # Enhanced architecture
    # }
    # for model_name, model in models.items():
    #     macs, params = ptflops.get_model_complexity_info(
    #         model,
    #         megapixel_dims,
    #     )
    #     print(f"{model_name} macs: {macs}, params: {params}")
    
    # # Test standard U-Net with RGB input
    # print("RGB complexity")
    # megapixel_dims = 3, 1024, 1024
    # model = raw_denoiser.UtNet2(in_channels=3, funit=32).eval()
    # macs, params = ptflops.get_model_complexity_info(model, megapixel_dims)
    # print(f"RGB macs: {macs}, params: {params}")
