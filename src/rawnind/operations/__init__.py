"""Operations module for the RAWNIND pipeline."""

from .preprocessing import RawLoader, WhiteBalance, Demosaic
from .models import Encoder, Decoder, Denoiser, Compressor
from .postprocessing import ToneMapper, GammaCorrection, Saver

# Export all operations
__all__ = [
    'RawLoader',
    'WhiteBalance', 
    'Demosaic',
    'Encoder',
    'Decoder',
    'Denoiser',
    'Compressor',
    'ToneMapper',
    'GammaCorrection',
    'Saver',
]

# Mapping for the operation registry to find implementations
OPERATION_IMPLEMENTATIONS = {
    'raw_loader': RawLoader,
    'white_balance': WhiteBalance,
    'demosaic': Demosaic,
    'encoder': Encoder,
    'decoder': Decoder,
    'denoiser': Denoiser,
    'compressor': Compressor,
    'tone_mapper': ToneMapper,
    'gamma_correction': GammaCorrection,
    'saver': Saver,
}