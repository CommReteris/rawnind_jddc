"""Lightweight test helpers for training classes.

This module exposes small subclasses that override get_dataloaders() to return
None, allowing fast import-time checks and sanity tests on training pipelines
without requiring access to full datasets.
"""

import configargparse
from ..training.denoise_compress_trainer import DCTrainingBayerToProfiledRGB
from ..training.denoise_compress_trainer import DCTrainingProfiledRGBToProfiledRGB
from ..training.denoiser_trainer import DenoiserTrainingBayerToProfiledRGB
from ..training.denoiser_trainer import DenoiserTrainingProfiledRGBToProfiledRGB
from ..models import denoise_then_compress, raw_denoiser


class DCTestCustomDataloaderBayerToProfiledRGB(
    DCTrainingBayerToProfiledRGB
):
    """Test subclass for Bayer-to-RGB denoise+compress training with null dataloaders.
    
    This class inherits from DCTrainingBayerToProfiledRGB but overrides the
    get_dataloaders method to return None instead of actual dataset loaders.
    This allows for quick import testing, code validation, and development
    without requiring access to the full datasets.
    
    Use cases:
    - Rapid testing of model initialization and architecture
    - Validating command-line argument parsing
    - Checking for import errors or dependency issues
    - Development of new training functionality without dataset overhead
    
    Inherits all parameters and methods from DCTrainingBayerToProfiledRGB.
    """

    def __init__(self, launch=False, **kwargs) -> None:
        """Initialize the test class with the same parameters as the parent class.
        
        Args:
            launch: If True, triggers immediate parameter processing
            **kwargs: Additional keyword arguments passed to the parent class
        """
        if kwargs.get('test_only'):
            preset_args = kwargs.get('preset_args', {})
            kwargs.update(preset_args)
            args = configargparse.Namespace(**kwargs)
            
            self.__dict__.update(vars(args))
            super().__init__(**kwargs)
        else:
            super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        """Override parent's get_dataloaders method to return None.
        
        This method intentionally bypasses the dataset loading process
        by returning None instead of actual DataLoader objects, allowing
        for rapid testing without dataset dependencies.
        
        Returns:
            None: Instead of the DataLoader objects that the parent would return
        """
        return None

    def instantiate_model(self):
        self.model = denoise_then_compress.DenoiseThenCompress(
            self.in_channels,
            self.arch_enc,
            self.arch_dec,
            self.hidden_out_channels,
            self.bitstream_out_channels
        )


class DenoiseTestCustomDataloaderBayerToProfiledRGB(
    DenoiserTrainingBayerToProfiledRGB
):
    """Test subclass for Bayer-to-RGB denoiser training with null dataloaders.
    
    This class inherits from DenoiserTrainingBayerToProfiledRGB but overrides the
    get_dataloaders method to return None instead of actual dataset loaders.
    This allows for quick import testing, code validation, and development
    without requiring access to the full datasets.
    
    Unlike the DC (denoise+compress) test class, this one focuses specifically on
    pure denoising functionality without the compression component.
    
    Use cases:
    - Rapid testing of denoiser model initialization and architecture
    - Validating command-line argument parsing for denoisers
    - Checking for import errors or dependency issues
    - Development of new denoising functionality without dataset overhead
    
    Inherits all parameters and methods from DenoiserTrainingBayerToProfiledRGB.
    """

    def __init__(self, launch=False, **kwargs) -> None:
        """Initialize the test class with the same parameters as the parent class.
        
        Args:
            launch: If True, triggers immediate parameter processing
            **kwargs: Additional keyword arguments passed to the parent class
        """
        if kwargs.get('test_only'):
            preset_args = kwargs.get('preset_args', {})
            kwargs.update(preset_args)
            args = configargparse.Namespace(**kwargs)
            
            self.__dict__.update(vars(args))
            super().__init__(**kwargs)
        else:
            super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        """Override parent's get_dataloaders method to return None.
        
        This method intentionally bypasses the dataset loading process
        by returning None instead of actual DataLoader objects, allowing
        for rapid testing without dataset dependencies.
        
        Returns:
            None: Instead of the DataLoader objects that the parent would return
        """
        return None

    def instantiate_model(self):
        self.model = raw_denoiser.UtNet2(self.in_channels, self.funit)


class DCTestCustomDataloaderProfiledRGBToProfiledRGB(
    DCTrainingProfiledRGBToProfiledRGB
):
    """Test subclass for RGB-to-RGB denoise+compress training with null dataloaders.
    
    This class inherits from DCTrainingProfiledRGBToProfiledRGB but overrides the
    get_dataloaders method to return None instead of actual dataset loaders.
    This allows for quick import testing, code validation, and development
    without requiring access to the full datasets.
    
    Unlike the Bayer-to-RGB test class, this one works with already demosaiced RGB
    images as both input and output, skipping the Bayer pattern processing step.
    
    Use cases:
    - Rapid testing of RGB model initialization and architecture
    - Validating command-line argument parsing for RGB processing
    - Checking for import errors or dependency issues
    - Development of new RGB processing functionality without dataset overhead
    
    Inherits all parameters and methods from DCTrainingProfiledRGBToProfiledRGB.
    """

    def __init__(self, launch=False, **kwargs) -> None:
        """Initialize the test class with the same parameters as the parent class.
        
        Args:
            launch: If True, triggers immediate parameter processing
            **kwargs: Additional keyword arguments passed to the parent class
        """
        if kwargs.get('test_only'):
            preset_args = kwargs.get('preset_args', {})
            kwargs.update(preset_args)
            args = configargparse.Namespace(**kwargs)
            
            self.__dict__.update(vars(args))
            super().__init__(**kwargs)
        else:
            super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        """Override parent's get_dataloaders method to return None.
        
        This method intentionally bypasses the dataset loading process
        by returning None instead of actual DataLoader objects, allowing
        for rapid testing without dataset dependencies.
        
        Returns:
            None: Instead of the DataLoader objects that the parent would return
        """
        return None

    def instantiate_model(self):
        self.model = denoise_then_compress.DenoiseThenCompress(
            self.in_channels,
            self.arch_enc,
            self.arch_dec,
            self.hidden_out_channels,
            self.bitstream_out_channels
        )


class DenoiseTestCustomDataloaderProfiledRGBToProfiledRGB(
    DenoiserTrainingProfiledRGBToProfiledRGB
):
    """Test subclass for RGB-to-RGB denoiser training with null dataloaders.
    
    This class inherits from DenoiserTrainingProfiledRGBToProfiledRGB but overrides the
    get_dataloaders method to return None instead of actual dataset loaders.
    This allows for quick import testing, code validation, and development
    without requiring access to the full datasets.
    
    This class combines two key characteristics:
    1. Pure denoising (without compression) like DenoiseTestCustomDataloaderBayerToProfiledRGB
    2. RGB-to-RGB processing (skipping Bayer pattern) like DCTestCustomDataloaderProfiledRGBToProfiledRGB
    
    Use cases:
    - Rapid testing of RGB denoiser model initialization and architecture
    - Validating command-line argument parsing for RGB denoisers
    - Checking for import errors or dependency issues
    - Development of new RGB denoising functionality without dataset overhead
    
    Inherits all parameters and methods from DenoiserTrainingProfiledRGBToProfiledRGB.
    """

    def __init__(self, launch=False, **kwargs) -> None:
        """Initialize the test class with the same parameters as the parent class.
        
        Args:
            launch: If True, triggers immediate parameter processing
            **kwargs: Additional keyword arguments passed to the parent class
        """
        if kwargs.get('test_only'):
            preset_args = kwargs.get('preset_args', {})
            kwargs.update(preset_args)
            args = configargparse.Namespace(**kwargs)
            
            self.__dict__.update(vars(args))
            super().__init__(**kwargs)
        else:
            super().__init__(launch=launch, **kwargs)

    def get_dataloaders(self) -> None:
        """Override parent's get_dataloaders method to return None.
        
        This method intentionally bypasses the dataset loading process
        by returning None instead of actual DataLoader objects, allowing
        for rapid testing without dataset dependencies.
        
        Returns:
            None: Instead of the DataLoader objects that the parent would return
        """
        return None

    def instantiate_model(self):
        self.model = raw_denoiser.UtNet2(self.in_channels, self.funit)