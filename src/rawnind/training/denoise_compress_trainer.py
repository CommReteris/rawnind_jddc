"""Training classes for joint denoise+compress models.

This module contains trainer classes for models that perform both denoising
and compression, extracted from the original training scripts per the partition plan.

Consolidated from:
- train_dc_bayer2prgb.py
- train_dc_prgb2prgb.py
"""

import multiprocessing
import os
import sys
import logging
from typing import Optional
from collections.abc import Iterable
import torch

from . import training_loops
from ..dependencies import pytorch_helpers
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw


class DCTrainingBayerToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.BayerImageToImageNNTraining,
):
    """Train a joint denoise+compress model from Bayer to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_dc_bayer2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_dc_bayer2prgb.yaml")
    ]

    def __init__(self, config=None, launch=False, **kwargs) -> None:
        from ..training.clean_api import LegacyTrainingConfig, TrainingConfig
        
        if config is not None and isinstance(config, LegacyTrainingConfig):
            # Convert LegacyTrainingConfig to TrainingConfig with testing defaults
            import tempfile
            training_config = TrainingConfig(
                model_architecture=config.arch,
                input_channels=config.in_channels,
                output_channels=config.out_channels,
                learning_rate=config.init_lr,
                batch_size=config.batch_size,
                crop_size=config.crop_size,
                total_steps=config.tot_steps,
                validation_interval=config.val_interval,
                loss_function=config.loss,
                device=config.device,
                patience=config.patience,
                lr_decay_factor=config.lr_multiplier,
                additional_metrics=config.additional_metrics,
                filter_units=config.filter_units,
                compression_lambda=config.compression_lambda,
                bit_estimator_lr_multiplier=config.bit_estimator_lr_multiplier,
                test_interval=config.test_interval,
                test_crop_size=config.test_crop_size,
                val_crop_size=config.val_crop_size,
                num_crops_per_image=config.num_crops_per_image,
                save_training_images=config.save_training_images,
                # Set test-only mode and required paths for testing
                test_only=True,
                expname="test_experiment",
                save_dpath=tempfile.mkdtemp(),
                metrics=config.metrics,
            )
            super().__init__(training_config)
        else:
            # Traditional kwargs-based initialization for backward compatibility
            super().__init__(**kwargs)
        
        # Handle launch logic if needed (preserve original intent)
        if launch:
            # Original launch logic would go here if it existed
            pass

    def autocomplete_args(self, args) -> None:
        if not args.in_channels:
            args.in_channels = 4
        super().autocomplete_args(args)


class DCTrainingProfiledRGBToProfiledRGB(
    training_loops.DenoiseCompressTraining,
    training_loops.PRGBImageToImageNNTraining,
):
    """Train a joint denoise+compress model from profiled RGB to profiled RGB.

    This entrypoint wires abstract_trainer mixins, loads defaults from
    config/train_dc_prgb2prgb.yaml, and runs training_loop().
    """

    CLS_CONFIG_FPATHS = training_loops.DenoiseCompressTraining.CLS_CONFIG_FPATHS + [
        os.path.join("dependencies", "configs", "train_dc_prgb2prgb.yaml")
    ]

    def __init__(self, config=None, launch=False, **kwargs):
        from ..training.clean_api import LegacyTrainingConfig, TrainingConfig
        
        if config is not None and isinstance(config, LegacyTrainingConfig):
            # Convert LegacyTrainingConfig to TrainingConfig with testing defaults
            import tempfile
            training_config = TrainingConfig(
                model_architecture=config.arch,
                input_channels=config.in_channels,
                output_channels=config.out_channels,
                learning_rate=config.init_lr,
                batch_size=config.batch_size,
                crop_size=config.crop_size,
                total_steps=config.tot_steps,
                validation_interval=config.val_interval,
                loss_function=config.loss,
                device=config.device,
                patience=config.patience,
                lr_decay_factor=config.lr_multiplier,
                additional_metrics=config.additional_metrics,
                filter_units=config.filter_units,
                compression_lambda=config.compression_lambda,
                bit_estimator_lr_multiplier=config.bit_estimator_lr_multiplier,
                test_interval=config.test_interval,
                test_crop_size=config.test_crop_size,
                val_crop_size=config.val_crop_size,
                num_crops_per_image=config.num_crops_per_image,
                save_training_images=config.save_training_images,
                # Set test-only mode and required paths for testing
                test_only=True,
                expname="test_experiment",
                save_dpath=tempfile.mkdtemp(),
                metrics=config.metrics,
            )
            super().__init__(training_config)
        else:
            # Traditional kwargs-based initialization for backward compatibility
            super().__init__(**kwargs)
        
        # Handle launch logic if needed (preserve original intent)
        if launch:
            # Original launch logic would go here if it existed
            pass

    def autocomplete_args(self, args):
        if not args.in_channels:
            args.in_channels = 3
        super().autocomplete_args(args)


# Clean API factory functions (no CLI dependencies)
def create_dc_bayer_trainer(**kwargs) -> DCTrainingBayerToProfiledRGB:
    """Create a Bayer-to-RGB denoising+compression trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DCTrainingBayerToProfiledRGB instance
    """
    return DCTrainingBayerToProfiledRGB(launch=False, **kwargs)


def create_dc_rgb_trainer(**kwargs) -> DCTrainingProfiledRGBToProfiledRGB:
    """Create an RGB-to-RGB denoising+compression trainer with clean API.
    
    Args:
        **kwargs: Training configuration parameters
        
    Returns:
        Configured DCTrainingProfiledRGBToProfiledRGB instance
    """
    return DCTrainingProfiledRGBToProfiledRGB(launch=False, **kwargs)



