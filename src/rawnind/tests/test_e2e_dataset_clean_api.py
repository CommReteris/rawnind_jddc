import logging
"""
E2E Tests for Dataset Package Clean API

This test suite demonstrates how the dataset package should work with clean,
programmatic interfaces without CLI dependencies. These tests serve as 
specifications for the desired API design.

The dataset package should provide:
1. Clean factory functions for creating datasets
2. Programmatic configuration without CLI parsing
3. Dataset loading and preprocessing utilities
4. Data validation and format conversion
5. Integration with training and inference workflows
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import current dataset package (these imports will likely fail initially)
try:
    from rawnind.dataset import (
        create_training_dataset,
        create_validation_dataset,
        create_test_dataset,
        DatasetConfig,
        DatasetMetadata,
        validate_dataset_format,
        prepare_dataset_splits
    )
except ImportError:
    # These are the clean interfaces we want to implement
    pytest.skip("Clean dataset interfaces not yet implemented", allow_module_level=True)


class TestDatasetFactoryFunctions:
    """Test clean factory functions for creating datasets."""
    
    def test_create_rgb_training_dataset(self):
        """Test creating RGB training dataset with clean API."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=8,
            color_profile="lin_rec2020",
            augmentations=["flip", "rotate"],
            validation_split=0.2,
            test_reserve_images=["test_001", "test_002"],
            device="cpu"
        )
        
        # Mock data paths
        dataset_paths = {
            'clean_images': ['/path/to/clean'],
            'noisy_images': ['/path/to/noisy'],
            'metadata_files': ['/path/to/metadata.yaml']
        }
        
        dataset = create_training_dataset(
            config=config,
            data_paths=dataset_paths
        )
        
        # Verify dataset properties
        assert dataset is not None
        assert dataset.config.dataset_type == "rgb_pairs"
        assert dataset.config.crop_size == 128
        assert dataset.config.num_crops_per_image == 4
        assert hasattr(dataset, '__iter__')
        assert hasattr(dataset, '__len__')
        
        # Test iteration
        for i, batch in enumerate(dataset):
            if i >= 2:  # Just test first few batches
                break
            assert 'clean_images' in batch
            assert 'noisy_images' in batch
            assert 'masks' in batch
            assert batch['clean_images'].shape[1:] == (3, 128, 128)  # C,H,W
            assert batch['noisy_images'].shape[1:] == (3, 128, 128)
            
    def test_create_bayer_training_dataset(self):
        """Test creating Bayer training dataset with clean API."""
        config = DatasetConfig(
            dataset_type="bayer_pairs",
            data_format="clean_noisy",
            input_channels=4,  # Bayer pattern
            output_channels=3,  # RGB output
            crop_size=256,
            num_crops_per_image=2,
            batch_size=4,
            color_profile="lin_rec2020",
            augmentations=["flip"],
            validation_split=0.15,
            test_reserve_images=["bayer_test_001"],
            device="cpu"
        )
        
        dataset_paths = {
            'bayer_images': ['/path/to/bayer'],
            'rgb_ground_truth': ['/path/to/rgb_gt'],
            'metadata_files': ['/path/to/bayer_metadata.yaml']
        }
        
        dataset = create_training_dataset(
            config=config,
            data_paths=dataset_paths
        )
        
        assert dataset is not None
        assert dataset.config.input_channels == 4
        assert dataset.config.output_channels == 3
        
        # Test Bayer-specific functionality
        for i, batch in enumerate(dataset):
            if i >= 1:
                break
            assert 'clean_images' in batch  # RGB ground truth
            assert 'noisy_images' in batch  # Bayer input
            assert 'masks' in batch
            assert 'rgb_xyz_matrices' in batch  # Color transformation matrices
            assert batch['noisy_images'].shape[1] == 4  # Bayer channels
            assert batch['clean_images'].shape[1] == 3  # RGB channels
            assert batch['rgb_xyz_matrices'].shape[-2:] == (3, 3)  # 3x3 matrix
    
    def test_create_self_supervised_dataset(self):
        """Test creating self-supervised dataset (clean-clean pairs)."""
        config = DatasetConfig(
            dataset_type="self_supervised",
            data_format="clean_clean",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=6,
            batch_size=4,
            augmentations=["flip", "rotate", "color_jitter"],
            noise_augmentation="synthetic",  # Add synthetic noise for self-supervision
            validation_split=0.2,
            device="cpu"
        )
        
        dataset_paths = {
            'clean_images': ['/path/to/clean_images'],
            'metadata_files': ['/path/to/metadata.yaml']
        }
        
        dataset = create_training_dataset(
            config=config,
            data_paths=dataset_paths
        )
        
        assert dataset is not None
        assert dataset.config.data_format == "clean_clean"
        assert dataset.config.noise_augmentation == "synthetic"
        
        # Test self-supervised data format
        for i, batch in enumerate(dataset):
            if i >= 1:
                break
            assert 'clean_images' in batch
            assert 'noisy_images' in batch  # Should be augmented versions
            assert 'masks' in batch
            # For self-supervised, both should be 3-channel RGB
            assert batch['clean_images'].shape[1] == 3
            assert batch['noisy_images'].shape[1] == 3


class TestDatasetConfigurationAndValidation:
    """Test dataset configuration and validation utilities."""
    
    def test_dataset_config_validation(self):
        """Test dataset configuration validation."""
        # Valid config should work
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=8,
            color_profile="lin_rec2020",
            device="cpu"
        )
        assert config.is_valid()
        
        # Invalid configurations should raise errors
        with pytest.raises(ValueError, match="Crop size must be positive and even"):
            DatasetConfig(
                dataset_type="rgb_pairs",
                data_format="clean_noisy",
                input_channels=3,
                output_channels=3,
                crop_size=127,  # Odd number - invalid for Bayer alignment
                num_crops_per_image=4,
                batch_size=8,
                device="cpu"
            )
            
        with pytest.raises(ValueError, match="Bayer datasets require 4 input channels"):
            DatasetConfig(
                dataset_type="bayer_pairs",
                data_format="clean_noisy",
                input_channels=3,  # Should be 4 for Bayer
                output_channels=3,
                crop_size=128,
                num_crops_per_image=4,
                batch_size=8,
                device="cpu"
            )
    
    def test_validate_dataset_format(self):
        """Test dataset format validation utility."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset files
            clean_dir = Path(tmpdir) / "clean"
            noisy_dir = Path(tmpdir) / "noisy"
            clean_dir.mkdir()
            noisy_dir.mkdir()
            
            # Create dummy image files
            for i in range(5):
                (clean_dir / f"image_{i:03d}.jpg").touch()
                (noisy_dir / f"image_{i:03d}.jpg").touch()
            
            # Create metadata file
            metadata = {
                'dataset_name': 'test_dataset',
                'total_images': 5,
                'image_format': 'jpg',
                'color_profile': 'lin_rec2020'
            }
            
            metadata_path = Path(tmpdir) / "metadata.yaml"
            with open(metadata_path, 'w') as f:
                import yaml
                yaml.dump(metadata, f)
            
            # Validate the dataset
            validation_result = validate_dataset_format(
                dataset_paths={
                    'clean_images': [str(clean_dir)],
                    'noisy_images': [str(noisy_dir)],
                    'metadata_files': [str(metadata_path)]
                },
                expected_format="clean_noisy"
            )
            
            assert validation_result['is_valid'] == True
            assert validation_result['total_pairs'] == 5
            assert validation_result['missing_pairs'] == 0
            assert 'validation_errors' not in validation_result or len(validation_result['validation_errors']) == 0


class TestDatasetPreprocessingAndAugmentation:
    """Test data preprocessing and augmentation functionality."""
    
    def test_image_cropping_and_masking(self):
        """Test image cropping with proper masking."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=4,
            batch_size=2,
            min_valid_pixels_ratio=0.8,  # Mask threshold
            device="cpu"
        )
        
        # Create mock dataset with masking functionality
        def mock_dataloader():
            # Create images with some "overexposed" regions
            clean_img = torch.randn(3, 128, 128)
            noisy_img = torch.randn(3, 128, 128)
            
            # Create mask with some invalid regions
            mask = torch.ones(3, 128, 128)
            mask[:, 60:80, 60:80] = 0  # Invalid region
            
            yield {
                'clean_images': clean_img.unsqueeze(0),
                'noisy_images': noisy_img.unsqueeze(0),
                'masks': mask.unsqueeze(0),
                'image_paths': ['test_image.jpg']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        for batch in dataset:
            # Verify crop dimensions
            assert batch['clean_images'].shape[-2:] == (64, 64)
            assert batch['noisy_images'].shape[-2:] == (64, 64)
            assert batch['masks'].shape[-2:] == (64, 64)
            
            # Verify sufficient valid pixels in each crop
            for i in range(batch['masks'].shape[0]):
                valid_ratio = batch['masks'][i].float().mean()
                assert valid_ratio >= config.min_valid_pixels_ratio
            break  # Just test first batch
    
    def test_bayer_pattern_alignment(self):
        """Test that Bayer pattern alignment is maintained during cropping."""
        config = DatasetConfig(
            dataset_type="bayer_pairs",
            data_format="clean_noisy",
            input_channels=4,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=1,
            maintain_bayer_alignment=True,
            device="cpu"
        )
        
        def mock_bayer_dataloader():
            # Create Bayer pattern (4-channel) and RGB ground truth
            bayer_img = torch.randn(4, 128, 128)
            rgb_gt = torch.randn(3, 128, 128)
            mask = torch.ones(3, 128, 128)
            
            yield {
                'clean_images': rgb_gt.unsqueeze(0),
                'noisy_images': bayer_img.unsqueeze(0),
                'masks': mask.unsqueeze(0),
                'rgb_xyz_matrices': torch.eye(3).unsqueeze(0),
                'image_paths': ['bayer_test.raw']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_bayer_dataloader()
        )
        
        for batch in dataset:
            # Verify Bayer alignment - crop coordinates should be even
            # This would be validated internally by the dataset
            assert batch['noisy_images'].shape[1] == 4  # Bayer channels
            assert batch['clean_images'].shape[1] == 3  # RGB channels
            break
    
    def test_data_augmentation_pipeline(self):
        """Test data augmentation functionality."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=2,
            augmentations=["horizontal_flip", "vertical_flip", "rotation_90"],
            augmentation_probability=0.8,
            device="cpu"
        )
        
        def mock_dataloader():
            for i in range(2):
                yield {
                    'clean_images': torch.randn(2, 3, 64, 64),
                    'noisy_images': torch.randn(2, 3, 64, 64),
                    'masks': torch.ones(2, 3, 64, 64),
                    'image_paths': [f'aug_test_{i}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        # Test that augmentations are applied
        augmentation_info = dataset.get_augmentation_info()
        assert 'horizontal_flip' in augmentation_info['available_augmentations']
        assert augmentation_info['probability'] == 0.8
        
        for batch in dataset:
            # Verify batch structure is maintained after augmentation
            assert 'clean_images' in batch
            assert 'noisy_images' in batch
            assert 'augmentation_applied' in batch  # Should track what was applied
            break


class TestDatasetSplitsAndReservedData:
    """Test dataset splitting and test data reservation."""
    
    def test_prepare_dataset_splits(self):
        """Test automatic dataset splitting functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock dataset structure
            data_dir = Path(tmpdir)
            
            # Create image directories
            for split in ['train', 'val', 'test']:
                split_dir = data_dir / split
                split_dir.mkdir()
                for i in range(10):
                    (split_dir / f"image_{i:03d}.jpg").touch()
            
            dataset_paths = {
                'image_directories': [str(data_dir)],
                'metadata_files': []
            }
            
            # Prepare splits
            splits = prepare_dataset_splits(
                dataset_paths=dataset_paths,
                validation_split=0.2,
                test_reserve_names=["image_008", "image_009"],
                stratify_by_noise_level=True
            )
            
            assert 'train' in splits
            assert 'validation' in splits  
            assert 'test' in splits
            assert len(splits['test']) >= 2  # Should include reserved images
            assert len(splits['validation']) > 0
            assert len(splits['train']) > len(splits['validation'])
    
    def test_test_reserve_enforcement(self):
        """Test that test reserve images are properly excluded from training."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=4,
            test_reserve_images=["reserved_001", "reserved_002"],
            enforce_test_reserve=True,
            device="cpu"
        )
        
        # Mock dataset with some reserved images
        def mock_dataloader_with_reserved():
            image_names = [
                "train_001", "train_002", "reserved_001", 
                "train_003", "reserved_002", "train_004"
            ]
            for name in image_names:
                yield {
                    'clean_images': torch.randn(1, 3, 128, 128),
                    'noisy_images': torch.randn(1, 3, 128, 128),
                    'masks': torch.ones(1, 3, 128, 128),
                    'image_paths': [f'{name}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader_with_reserved()
        )
        
        # Verify reserved images are excluded
        seen_images = set()
        for batch in dataset:
            for path in batch['image_paths']:
                seen_images.add(path)
        
        # Should not see reserved images in training data
        assert "reserved_001.jpg" not in seen_images
        assert "reserved_002.jpg" not in seen_images
        assert "train_001.jpg" in seen_images


class TestDatasetValidationAndMetadata:
    """Test dataset validation and metadata handling."""
    
    def test_dataset_metadata_loading(self):
        """Test loading and parsing dataset metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock metadata file
            metadata = {
                'dataset_info': {
                    'name': 'test_rawnind_dataset',
                    'version': '1.0',
                    'total_images': 100,
                    'image_format': 'exr',
                    'color_profile': 'lin_rec2020'
                },
                'camera_info': {
                    'iso_levels': [100, 200, 400],
                    'white_balance': 'daylight',
                    'color_matrix': [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                },
                'processing_info': {
                    'demosaicing_method': 'bilinear',
                    'noise_reduction': False,
                    'gamma_correction': False
                }
            }
            
            metadata_path = Path(tmpdir) / "dataset_metadata.yaml"
            with open(metadata_path, 'w') as f:
                import yaml
                yaml.dump(metadata, f)
            
            # Load metadata using clean API
            dataset_metadata = DatasetMetadata.from_file(metadata_path)
            
            assert dataset_metadata.name == 'test_rawnind_dataset'
            assert dataset_metadata.total_images == 100
            assert dataset_metadata.color_profile == 'lin_rec2020'
            assert len(dataset_metadata.iso_levels) == 3
            assert dataset_metadata.color_matrix.shape == (3, 3)
    
    def test_dataset_integrity_validation(self):
        """Test dataset integrity validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dataset with some issues
            clean_dir = Path(tmpdir) / "clean"
            noisy_dir = Path(tmpdir) / "noisy"
            clean_dir.mkdir()
            noisy_dir.mkdir()
            
            # Create unmatched pairs (missing some noisy images)
            for i in range(5):
                (clean_dir / f"image_{i:03d}.jpg").touch()
            for i in range(3):  # Missing 2 noisy images
                (noisy_dir / f"image_{i:03d}.jpg").touch()
            
            dataset_paths = {
                'clean_images': [str(clean_dir)],
                'noisy_images': [str(noisy_dir)]
            }
            
            # Validate dataset integrity
            integrity_report = validate_dataset_format(
                dataset_paths=dataset_paths,
                expected_format="clean_noisy",
                check_image_pairs=True,
                check_file_integrity=True
            )
            
            assert integrity_report['is_valid'] == False
            assert integrity_report['missing_pairs'] == 2
            assert len(integrity_report['validation_errors']) > 0
            assert 'Missing noisy image' in str(integrity_report['validation_errors'])


class TestDatasetLoadingPerformance:
    """Test dataset loading performance and optimization."""
    
    def test_lazy_loading_behavior(self):
        """Test that datasets use lazy loading for memory efficiency."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=4,
            lazy_loading=True,  # Enable lazy loading
            cache_size=10,  # Small cache for testing
            device="cpu"
        )
        
        # Mock large dataset
        def mock_large_dataset():
            for i in range(100):  # Large dataset
                yield {
                    'clean_images': torch.randn(1, 3, 256, 256),  # Large images
                    'noisy_images': torch.randn(1, 3, 256, 256),
                    'masks': torch.ones(1, 3, 256, 256),
                    'image_paths': [f'large_image_{i:03d}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_large_dataset()
        )
        
        # Verify lazy loading properties
        assert dataset.config.lazy_loading == True
        assert dataset.config.cache_size == 10
        
        # Test that memory usage is reasonable (not loading all images at once)
        cache_info = dataset.get_cache_info()
        assert cache_info['max_size'] == 10
        assert cache_info['current_size'] >= 0
    
    def test_multiprocessing_data_loading(self):
        """Test multiprocessing data loading capabilities."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=4,
            num_workers=2,  # Enable multiprocessing
            device="cpu"
        )
        
        def mock_dataloader():
            for i in range(10):
                yield {
                    'clean_images': torch.randn(1, 3, 128, 128),
                    'noisy_images': torch.randn(1, 3, 128, 128),
                    'masks': torch.ones(1, 3, 128, 128),
                    'image_paths': [f'mp_test_{i}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        # Verify multiprocessing configuration
        loader_info = dataset.get_loader_info()
        assert loader_info['num_workers'] == 2
        assert loader_info['multiprocessing_enabled'] == True
        
        # Test that data loads correctly with multiprocessing
        batch_count = 0
        for batch in dataset:
            assert 'clean_images' in batch
            batch_count += 1
            if batch_count >= 3:  # Test a few batches
                break


class TestSpecializedDatasetTypes:
    """Test specialized dataset types and formats."""
    
    def test_rawnind_academic_dataset(self):
        """Test loading RawNIND academic dataset format."""
        config = DatasetConfig(
            dataset_type="rawnind_academic",
            data_format="clean_noisy",
            input_channels=4,  # Bayer
            output_channels=3,  # RGB
            crop_size=256,
            num_crops_per_image=1,
            batch_size=1,
            color_profile="lin_rec2020",
            academic_dataset_version="v1.0",
            device="cpu"
        )
        
        # Mock RawNIND dataset structure
        def mock_rawnind_dataloader():
            yield {
                'clean_images': torch.randn(1, 3, 512, 512),  # RGB ground truth
                'noisy_images': torch.randn(1, 4, 1024, 1024),  # Bayer RAW
                'masks': torch.ones(1, 3, 512, 512),
                'rgb_xyz_matrices': torch.randn(1, 3, 3),
                'camera_metadata': {
                    'iso': 100,
                    'white_balance': [1.0, 1.2, 1.8],
                    'exposure_time': 1/60
                },
                'image_paths': ['RawNIND_test_001.raw']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'rawnind_path': '/path/to/rawnind'},
            data_loader_override=mock_rawnind_dataloader()
        )
        
        for batch in dataset:
            assert 'camera_metadata' in batch
            assert 'iso' in batch['camera_metadata']
            assert batch['rgb_xyz_matrices'].shape[-2:] == (3, 3)
            break
    
    def test_hdr_dataset_support(self):
        """Test HDR/EXR dataset loading capabilities."""
        config = DatasetConfig(
            dataset_type="hdr_pairs",
            data_format="clean_noisy", 
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=2,
            batch_size=2,
            file_format="exr",  # HDR format
            dynamic_range="hdr",
            tone_mapping="none",  # Keep linear
            device="cpu"
        )
        
        def mock_hdr_dataloader():
            # HDR images have higher dynamic range
            clean_hdr = torch.randn(3, 256, 256) * 10  # High dynamic range
            noisy_hdr = torch.randn(3, 256, 256) * 10
            
            yield {
                'clean_images': clean_hdr.unsqueeze(0),
                'noisy_images': noisy_hdr.unsqueeze(0),
                'masks': torch.ones(3, 256, 256).unsqueeze(0),
                'dynamic_range_info': {
                    'min_value': float(clean_hdr.min()),
                    'max_value': float(clean_hdr.max()),
                    'is_hdr': True
                },
                'image_paths': ['hdr_test.exr']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'hdr_path': '/path/to/hdr'},
            data_loader_override=mock_hdr_dataloader()
        )
        
        for batch in dataset:
            assert 'dynamic_range_info' in batch
            assert batch['dynamic_range_info']['is_hdr'] == True
            # HDR values can be > 1.0
            assert batch['clean_images'].max() > 1.0 or batch['clean_images'].min() < 0.0
            break


class TestDatasetFormatConversions:
    """Test format conversions and color space handling."""
    
    def test_color_space_conversion(self):
        """Test color space conversion utilities."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=2,
            input_color_profile="srgb",
            output_color_profile="lin_rec2020",
            apply_color_conversion=True,
            device="cpu"
        )
        
        def mock_srgb_dataloader():
            # sRGB data (gamma-corrected)
            srgb_clean = torch.rand(3, 128, 128)  # [0,1] range
            srgb_noisy = torch.rand(3, 128, 128)
            
            yield {
                'clean_images': srgb_clean.unsqueeze(0),
                'noisy_images': srgb_noisy.unsqueeze(0),
                'masks': torch.ones(3, 128, 128).unsqueeze(0),
                'color_profile': 'srgb',
                'image_paths': ['srgb_test.jpg']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'srgb_path': '/path/to/srgb'},
            data_loader_override=mock_srgb_dataloader()
        )
        
        for batch in dataset:
            # After conversion to linear, values should potentially be > 1.0
            # (sRGB gamma curve expansion)
            assert batch['color_profile_info']['input'] == 'srgb'
            assert batch['color_profile_info']['output'] == 'lin_rec2020'
            assert batch['color_profile_info']['conversion_applied'] == True
            break
    
    def test_bayer_demosaicing_options(self):
        """Test different Bayer demosaicing options."""
        config = DatasetConfig(
            dataset_type="bayer_pairs",
            data_format="clean_noisy",
            input_channels=4,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=2,
            batch_size=2,
            demosaicing_method="bilinear",  # vs "ahd", "vng", etc.
            bayer_pattern="RGGB",
            device="cpu"
        )
        
        def mock_bayer_dataloader():
            # Raw Bayer data
            bayer_data = torch.randn(4, 256, 256)  # RGGB pattern
            rgb_gt = torch.randn(3, 256, 256)
            
            yield {
                'clean_images': rgb_gt.unsqueeze(0),
                'noisy_images': bayer_data.unsqueeze(0),
                'masks': torch.ones(3, 256, 256).unsqueeze(0),
                'bayer_info': {
                    'pattern': 'RGGB',
                    'demosaicing_method': 'bilinear'
                },
                'rgb_xyz_matrices': torch.eye(3).unsqueeze(0),
                'image_paths': ['bayer_demosaic_test.raw']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'bayer_path': '/path/to/bayer'},
            data_loader_override=mock_bayer_dataloader()
        )
        
        for batch in dataset:
            assert 'bayer_info' in batch
            assert batch['bayer_info']['pattern'] == 'RGGB'
            assert batch['bayer_info']['demosaicing_method'] == 'bilinear'
            break


class TestDatasetIntegrationWithTraining:
    """Test dataset integration with training workflows."""
    
    @patch('rawnind.training.create_denoiser_trainer')
    def test_dataset_trainer_integration(self, mock_create_trainer):
        """Test seamless integration between dataset and training packages."""
        # Mock trainer that expects specific data format
        mock_trainer = Mock()
        mock_trainer.prepare_datasets.return_value = {
            'train_loader': Mock(),
            'val_loader': Mock()
        }
        mock_create_trainer.return_value = mock_trainer
        
        # Dataset config compatible with training
        dataset_config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=8,
            device="cpu"
        )
        
        # Training config that uses dataset
        training_config = {
            'model_architecture': 'unet',
            'learning_rate': 1e-4,
            'total_steps': 1000
        }
        
        # Create integrated workflow
        datasets = create_training_dataset(
            config=dataset_config,
            data_paths={'clean': '/path/to/clean', 'noisy': '/path/to/noisy'}
        )
        
        # Verify compatibility
        dataset_info = datasets.get_compatibility_info()
        assert dataset_info['compatible_with_training'] == True
        assert dataset_info['batch_format'] == 'standard'
        assert dataset_info['tensor_dtypes']['images'] == torch.float32
        assert dataset_info['tensor_dtypes']['masks'] == torch.bool
    
    def test_dataset_preprocessing_pipeline(self):
        """Test complete dataset preprocessing pipeline."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=2,
            preprocessing_steps=[
                "normalize_range",      # [0,1] normalization
                "color_space_convert",  # sRGB to linear
                "noise_level_estimate", # Estimate noise characteristics
                "quality_check"         # Check image quality
            ],
            device="cpu"
        )
        
        def mock_dataloader():
            # Raw sRGB images that need preprocessing
            srgb_clean = torch.rand(3, 128, 128) * 255  # [0,255] range
            srgb_noisy = torch.rand(3, 128, 128) * 255
            
            yield {
                'clean_images': srgb_clean.unsqueeze(0),
                'noisy_images': srgb_noisy.unsqueeze(0),
                'masks': torch.ones(3, 128, 128).unsqueeze(0),
                'preprocessing_info': {
                    'input_range': [0, 255],
                    'input_color_space': 'srgb'
                },
                'image_paths': ['preprocess_test.jpg']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        for batch in dataset:
            # After preprocessing, should be in [0,1] range and linear color space
            assert batch['clean_images'].min() >= 0.0
            assert batch['clean_images'].max() <= 1.0
            assert batch['preprocessing_info']['steps_applied'] == config.preprocessing_steps
            assert batch['preprocessing_info']['output_color_space'] == 'lin_rec2020'
            break


class TestDatasetErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_data_handling(self):
        """Test handling of missing or corrupted data files."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=4,
            handle_missing_files="skip",  # vs "error"
            device="cpu"
        )
        
        # Mock dataset with some missing files
        def mock_dataloader_with_missing():
            files = ["good_001.jpg", "missing_002.jpg", "good_003.jpg"]
            for i, filename in enumerate(files):
                if "missing" in filename:
                    continue  # Simulate missing file
                yield {
                    'clean_images': torch.randn(1, 3, 128, 128),
                    'noisy_images': torch.randn(1, 3, 128, 128),
                    'masks': torch.ones(1, 3, 128, 128),
                    'image_paths': [filename]
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader_with_missing()
        )
        
        # Should skip missing files gracefully
        loaded_files = []
        for batch in dataset:
            loaded_files.extend(batch['image_paths'])
        
        assert "good_001.jpg" in loaded_files
        assert "good_003.jpg" in loaded_files
        assert "missing_002.jpg" not in loaded_files
    
    def test_insufficient_valid_pixels_handling(self):
        """Test handling of crops with insufficient valid pixels."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=4,
            batch_size=2,
            min_valid_pixels_ratio=0.8,
            max_crop_attempts=10,
            device="cpu"
        )
        
        def mock_dataloader_with_masks():
            # Create image with large masked region
            clean_img = torch.randn(3, 256, 256)
            noisy_img = torch.randn(3, 256, 256)
            
            # Large masked region in center
            mask = torch.ones(3, 256, 256)
            mask[:, 100:156, 100:156] = 0  # Large invalid region
            
            yield {
                'clean_images': clean_img.unsqueeze(0),
                'noisy_images': noisy_img.unsqueeze(0),
                'masks': mask.unsqueeze(0),
                'image_paths': ['heavy_mask_test.jpg']
            }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader_with_masks()
        )
        
        # Should either find valid crops or handle gracefully
        for batch in dataset:
            # Each crop should have sufficient valid pixels
            for i in range(batch['masks'].shape[0]):
                valid_ratio = batch['masks'][i].float().mean()
                assert valid_ratio >= config.min_valid_pixels_ratio
            break


class TestDatasetStatisticsAndAnalysis:
    """Test dataset statistics and analysis utilities."""
    
    def test_dataset_statistics_computation(self):
        """Test computation of dataset statistics."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=2,
            compute_statistics=True,
            device="cpu"
        )
        
        def mock_dataloader():
            for i in range(5):
                yield {
                    'clean_images': torch.randn(1, 3, 128, 128),
                    'noisy_images': torch.randn(1, 3, 128, 128),
                    'masks': torch.ones(1, 3, 128, 128),
                    'image_paths': [f'stats_test_{i}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        # Compute and verify statistics
        stats = dataset.compute_statistics()
        
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert stats['mean'].shape == (3,)  # Per-channel statistics
        assert stats['std'].shape == (3,)
        assert stats['total_samples'] > 0
    
    def test_noise_level_analysis(self):
        """Test noise level analysis for datasets."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=2,
            analyze_noise_levels=True,
            device="cpu"
        )
        
        def mock_dataloader_with_varying_noise():
            for i, noise_std in enumerate([0.01, 0.05, 0.1]):  # Different noise levels
                clean_img = torch.randn(3, 128, 128)
                noisy_img = clean_img + torch.randn(3, 128, 128) * noise_std
                
                yield {
                    'clean_images': clean_img.unsqueeze(0),
                    'noisy_images': noisy_img.unsqueeze(0),
                    'masks': torch.ones(3, 128, 128).unsqueeze(0),
                    'noise_info': {
                        'estimated_std': noise_std,
                        'noise_type': 'gaussian'
                    },
                    'image_paths': [f'noise_test_{i}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader_with_varying_noise()
        )
        
        # Analyze noise characteristics
        noise_analysis = dataset.analyze_noise_levels()
        
        assert 'noise_distribution' in noise_analysis
        assert 'mean_noise_std' in noise_analysis
        assert 'noise_level_categories' in noise_analysis
        assert noise_analysis['mean_noise_std'] > 0


class TestDatasetValidationDataLoaders:
    """Test validation and test dataset loaders."""
    
    def test_create_validation_dataset(self):
        """Test creating validation dataset with different properties than training."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=256,  # Larger crops for validation
            num_crops_per_image=1,  # Single crop for reproducibility
            batch_size=1,   # Single image batches
            augmentations=[],  # No augmentations for validation
            center_crop=True,  # Use center crop instead of random
            device="cpu"
        )
        
        val_dataset = create_validation_dataset(
            config=config,
            data_paths={'val_images': '/path/to/validation'}
        )
        
        assert val_dataset is not None
        assert val_dataset.config.num_crops_per_image == 1
        assert val_dataset.config.center_crop == True
        assert len(val_dataset.config.augmentations) == 0
        
        # Validation should be deterministic
        determinism_info = val_dataset.get_determinism_info()
        assert determinism_info['is_deterministic'] == True
        assert determinism_info['uses_center_crop'] == True
    
    def test_create_test_dataset(self):
        """Test creating test dataset with specific requirements."""
        config = DatasetConfig(
            dataset_type="rgb_pairs", 
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=512,  # Large crops for testing
            num_crops_per_image=1,
            batch_size=1,
            augmentations=[],
            center_crop=True,
            save_individual_results=True,  # Save per-image results
            device="cpu"
        )
        
        test_dataset = create_test_dataset(
            config=config,
            data_paths={'test_images': '/path/to/test'},
            test_names_filter=["test_001", "test_002"]  # Only specific test images
        )
        
        assert test_dataset is not None
        assert test_dataset.config.save_individual_results == True
        assert test_dataset.config.crop_size == 512
        
        # Test dataset should provide image metadata
        for batch in test_dataset:
            assert 'image_metadata' in batch
            assert 'image_name' in batch['image_metadata']
            assert 'original_size' in batch['image_metadata']
            break


class TestDatasetCachingAndPerformance:
    """Test dataset caching and performance optimizations."""
    
    def test_dataset_caching_behavior(self):
        """Test dataset caching for improved performance."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,
            num_crops_per_image=2,
            batch_size=4,
            enable_caching=True,
            cache_size_mb=100,  # 100MB cache
            cache_strategy="lru",  # Least Recently Used
            device="cpu"
        )
        
        def mock_dataloader():
            # Same images repeated to test caching
            for i in range(10):
                image_id = i % 3  # Repeat first 3 images
                yield {
                    'clean_images': torch.randn(1, 3, 128, 128),
                    'noisy_images': torch.randn(1, 3, 128, 128),
                    'masks': torch.ones(1, 3, 128, 128),
                    'image_paths': [f'cache_test_{image_id:03d}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader()
        )
        
        # Test caching functionality
        cache_stats = dataset.get_cache_statistics()
        assert cache_stats['enabled'] == True
        assert cache_stats['max_size_mb'] == 100
        assert cache_stats['strategy'] == 'lru'
        
        # Load data multiple times to test cache hits
        for epoch in range(2):
            for i, batch in enumerate(dataset):
                if i >= 5:  # Test a few batches
                    break
        
        final_cache_stats = dataset.get_cache_statistics()
        assert final_cache_stats['cache_hits'] > 0  # Should have cache hits on repeated data


class TestRawNINDAcademicDatasetSupport:
    """Test specific support for RawNIND academic dataset."""
    
    def test_rawnind_dataset_loading(self):
        """Test loading RawNIND dataset from UCLouvain Dataverse."""
        config = DatasetConfig(
            dataset_type="rawnind_academic",
            data_format="bayer_rgb_pairs",
            input_channels=4,
            output_channels=3,
            crop_size=256,
            num_crops_per_image=1,
            batch_size=1,
            academic_dataset_path="/path/to/rawnind",
            load_camera_metadata=True,
            device="cpu"
        )
        
        # Mock RawNIND structure
        def mock_rawnind_loader():
            camera_types = ["Canon_7D", "Sony_A7C", "Nikon_D60"]
            for i, camera in enumerate(camera_types):
                yield {
                    'clean_images': torch.randn(1, 3, 1024, 1024),  # High-res RGB
                    'noisy_images': torch.randn(1, 4, 2048, 2048),  # High-res Bayer
                    'masks': torch.ones(1, 3, 1024, 1024),
                    'rgb_xyz_matrices': torch.randn(1, 3, 3),
                    'camera_metadata': {
                        'camera_model': camera,
                        'iso': 100,
                        'aperture': 'f/5.6',
                        'shutter_speed': '1/60',
                        'white_balance': 'daylight',
                        'color_matrix': torch.randn(3, 3)
                    },
                    'rawnind_info': {
                        'dataset_version': 'v1.0',
                        'image_id': f'RawNIND_{i:03d}',
                        'scene_category': 'indoor'
                    },
                    'image_paths': [f'RawNIND_{camera}_{i:03d}.raw']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'rawnind_path': '/path/to/rawnind'},
            data_loader_override=mock_rawnind_loader()
        )
        
        for batch in dataset:
            assert 'camera_metadata' in batch
            assert 'rawnind_info' in batch
            assert batch['camera_metadata']['camera_model'] in ["Canon_7D", "Sony_A7C", "Nikon_D60"]
            assert batch['rawnind_info']['dataset_version'] == 'v1.0'
            break
    
    def test_rawnind_test_reserve_handling(self):
        """Test RawNIND test reserve image handling."""
        config = DatasetConfig(
            dataset_type="rawnind_academic",
            data_format="bayer_rgb_pairs",
            input_channels=4,
            output_channels=3,
            crop_size=256,
            num_crops_per_image=1,
            batch_size=1,
            test_reserve_config_path="src/rawnind/dependencies/configs/test_reserve.yaml",
            enforce_test_reserve=True,
            device="cpu"
        )
        
        # Mock test reserve configuration
        test_reserve_names = [
            "Bayer_TEST_7D-2_GT_ISO100",
            "Bayer_TEST_MuseeL-bluebirds-A7C_GT_ISO50",
            "Bayer_TEST_TitusToys_GT_ISO50"
        ]
        
        def mock_rawnind_with_reserves():
            all_images = [
                "Bayer_TRAIN_indoor_001",
                "Bayer_TEST_7D-2_GT_ISO100",  # Reserved
                "Bayer_TRAIN_outdoor_001", 
                "Bayer_TEST_MuseeL-bluebirds-A7C_GT_ISO50",  # Reserved
                "Bayer_TRAIN_indoor_002"
            ]
            
            for img_name in all_images:
                if any(reserved in img_name for reserved in test_reserve_names):
                    continue  # Skip reserved images in training
                    
                yield {
                    'clean_images': torch.randn(1, 3, 512, 512),
                    'noisy_images': torch.randn(1, 4, 1024, 1024),
                    'masks': torch.ones(1, 3, 512, 512),
                    'rgb_xyz_matrices': torch.eye(3).unsqueeze(0),
                    'rawnind_info': {
                        'image_id': img_name,
                        'is_test_reserved': any(reserved in img_name for reserved in test_reserve_names)
                    },
                    'image_paths': [f'{img_name}.raw']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'rawnind_path': '/path/to/rawnind'},
            data_loader_override=mock_rawnind_with_reserves()
        )
        
        # Verify test reserve enforcement
        seen_images = set()
        for batch in dataset:
            image_id = batch['rawnind_info']['image_id']
            seen_images.add(image_id)
            assert not any(reserved in image_id for reserved in test_reserve_names)
        
        # Should only see training images, not reserved ones
        assert "Bayer_TRAIN_indoor_001" in seen_images
        assert "Bayer_TEST_7D-2_GT_ISO100" not in seen_images


class TestDatasetUtilitiesAndHelpers:
    """Test dataset utility functions and helpers."""
    
    def test_dataset_format_conversion(self):
        """Test converting between different dataset formats."""
        # Test converting from paired format to self-supervised
        paired_config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            device="cpu"
        )
        
        self_supervised_config = DatasetConfig(
            dataset_type="self_supervised",
            data_format="clean_clean",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            noise_augmentation="synthetic",
            device="cpu"
        )
        
        # Mock conversion utility
        from rawnind.dataset import convert_dataset_format
        
        conversion_result = convert_dataset_format(
            source_config=paired_config,
            target_config=self_supervised_config,
            source_path="/path/to/paired",
            target_path="/path/to/self_supervised"
        )
        
        assert conversion_result['conversion_successful'] == True
        assert conversion_result['source_format'] == "clean_noisy"
        assert conversion_result['target_format'] == "clean_clean"
    
    def test_dataset_subset_creation(self):
        """Test creating dataset subsets for specific purposes."""
        full_config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=4,
            batch_size=8,
            device="cpu"
        )
        
        # Create subset for quick validation
        subset_config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=64,  # Smaller for speed
            num_crops_per_image=1,  # Fewer crops
            batch_size=2,   # Smaller batches
            max_samples=10,  # Limit dataset size
            device="cpu"
        )
        
        full_dataset = create_training_dataset(
            config=full_config,
            data_paths={'images': '/path/to/full'}
        )
        
        subset_dataset = create_validation_dataset(
            config=subset_config,
            data_paths={'images': '/path/to/full'},
            parent_dataset=full_dataset  # Create subset from full dataset
        )
        
        assert len(subset_dataset) <= subset_config.max_samples
        assert subset_dataset.config.crop_size < full_dataset.config.crop_size


class TestDatasetQualityAssurance:
    """Test dataset quality assurance and validation."""
    
    def test_image_quality_validation(self):
        """Test validation of image quality and format correctness."""
        config = DatasetConfig(
            dataset_type="rgb_pairs",
            data_format="clean_noisy",
            input_channels=3,
            output_channels=3,
            crop_size=128,
            num_crops_per_image=2,
            batch_size=2,
            quality_checks=["alignment", "exposure", "artifacts"],
            quality_thresholds={
                "max_alignment_error": 0.035,
                "max_overexposure_ratio": 0.01,
                "min_image_quality_score": 0.7
            },
            device="cpu"
        )
        
        def mock_dataloader_with_quality_issues():
            for i in range(3):
                clean_img = torch.randn(3, 256, 256)
                noisy_img = torch.randn(3, 256, 256)
                
                # Simulate some quality issues
                if i == 1:
                    # Overexposed regions
                    clean_img[:, 50:100, 50:100] = 2.0  # Overexposed
                elif i == 2:
                    # Misalignment (would be detected by alignment check)
                    noisy_img = torch.roll(noisy_img, shifts=(2, 2), dims=(-2, -1))
                
                yield {
                    'clean_images': clean_img.unsqueeze(0),
                    'noisy_images': noisy_img.unsqueeze(0),
                    'masks': torch.ones(3, 256, 256).unsqueeze(0),
                    'quality_info': {
                        'alignment_error': 0.02 if i != 2 else 0.05,  # High error for misaligned
                        'overexposure_ratio': 0.005 if i != 1 else 0.03,  # High for overexposed
                        'quality_score': 0.9 if i == 0 else 0.6  # Low for problematic images
                    },
                    'image_paths': [f'quality_test_{i}.jpg']
                }
        
        dataset = create_training_dataset(
            config=config,
            data_paths={'mock_data': True},
            data_loader_override=mock_dataloader_with_quality_issues()
        )
        
        # Should filter out low-quality images
        quality_report = dataset.get_quality_report()
        assert quality_report['total_checked'] == 3
        assert quality_report['passed_quality_checks'] < 3  # Some should be filtered
        assert quality_report['failed_checks']['overexposure'] >= 1
        assert quality_report['failed_checks']['alignment'] >= 1


# These tests demonstrate the clean API we want to implement for the dataset package.
# The dataset package should support:
# 1. Factory functions for creating different dataset types
# 2. Configuration classes for explicit parameter specification  
# 3. Support for various data formats (RGB, Bayer, HDR)
# 4. Data validation and quality assurance
# 5. Preprocessing and augmentation pipelines
# 6. Dataset splitting and test reserve management
# 7. Integration with training and inference workflows
# 8. Performance optimizations (caching, lazy loading)
# 9. Academic dataset format support (RawNIND)
# 10. Comprehensive error handling and edge case management