"""
Integration tests for the `rawnind.dataset` package.

This module contains integration tests that validate the end-to-end
functionality of the dataset creation and loading process, ensuring
that the clean API works as expected and that the domain logic
is preserved.
"""

import pytest
import torch
import yaml
from pathlib import Path

# Import the new clean API
from typing import Optional
from ..clean_api import create_training_dataset
from ..clean_api import CleanDataset
from ..dataset_config import DatasetConfig, BayerDatasetConfig, RgbDatasetConfig

# Import dependencies
from ...dependencies import pytorch_helpers as pt_helpers_dep
from unittest.mock import patch, MagicMock
import numpy as np


# Mock content_fpath for hermetic testing
@pytest.fixture
def mock_content_fpath_rawnind(tmp_path):
    """Create a mock YAML file path for RAWNIND content."""
    yaml_path = tmp_path / 'rawnind_content.yaml'
    yaml_content = {
        'images': [
            {'img_id': 'MuseeL-Bobo-alt-A7C', 'gt_path': str(tmp_path / 'gt_bobo.exr'), 'raw_path': str(tmp_path / 'raw_bobo.exr'),
             'is_bayer': True, 'image_set': 'MuseeL-Bobo-alt-A7C', 'best_alignment_loss': 0.01, 'mask_mean': 0.9,
             'rgb_msssim_score': 0.95, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'raw_gain': 1.5,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'gt_bobo.exr'), 'f_bayer_fpath': str(tmp_path / 'raw_bobo.exr'), 'mask_fpath': str(tmp_path / 'mask_bobo.exr')}]},
            {'img_id': 'MuseeL-yombe-A7C', 'gt_path': str(tmp_path / 'gt_yombe.exr'), 'raw_path': str(tmp_path / 'raw_yombe.exr'),
             'is_bayer': True, 'image_set': 'MuseeL-yombe-A7C', 'best_alignment_loss': 0.02, 'mask_mean': 0.85,
             'rgb_msssim_score': 0.80, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'raw_gain': 1.2,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'gt_yombe.exr'), 'f_bayer_fpath': str(tmp_path / 'raw_yombe.exr'), 'mask_fpath': str(tmp_path / 'mask_yombe.exr')}]}
        ]
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files that the RawImageDataset would expect
    (tmp_path / 'gt_bobo.exr').touch()
    (tmp_path / 'raw_bobo.exr').touch()
    (tmp_path / 'mask_bobo.exr').touch()
    (tmp_path / 'gt_yombe.exr').touch()
    (tmp_path / 'raw_yombe.exr').touch()
    (tmp_path / 'mask_yombe.exr').touch()
    return str(yaml_path)

@pytest.fixture
def mock_content_fpath_extraraw(tmp_path):
    """Create a mock YAML file path for EXTRARAW content."""
    yaml_path = tmp_path / 'extraraw_content.yaml'
    yaml_content = {
        'images': [
            {'img_id': 'image_exr1', 'gt_path': str(tmp_path / 'image_exr1.exr'), 'raw_path': str(tmp_path / 'image_exr1.exr'),
             'is_bayer': False, 'image_set': 'image_exr1', 'best_alignment_loss': 0.01, 'mask_mean': 0.9,
             'rgb_msssim_score': 0.9, 'rgb_xyz_matrix': [[1,0,0],[0,1,0],[0,0,1],[0,0,0]], 'rgb_gain': 1.0,
             'crops': [{'coordinates': [0,0], 'gt_linrec2020_fpath': str(tmp_path / 'image_exr1.exr'), 'gt_bayer_fpath': str(tmp_path / 'image_exr1.exr'), 'mask_fpath': str(tmp_path / 'mask_exr1.exr')}]}
        ]
    }
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f)
    # Create dummy image files
    (tmp_path / 'image_exr1.exr').touch()
    (tmp_path / 'mask_exr1.exr').touch()
    return [str(yaml_path)]

@pytest.fixture(autouse=True)
def patch_rawproc_paths(monkeypatch):
    """Patch file-reading functions to avoid actual I/O during tests."""
    def mock_fpath_to_tensor(fpath):
        if 'bayer' in fpath:
            return torch.ones(4, 128, 128)  # Bayer is 4ch, half resolution
        elif 'gt' in fpath or 'f_linrec2020' in fpath:  # RGB image
            return torch.ones(3, 256, 256)
        elif 'mask' in fpath:
            return torch.ones(1, 256, 256, dtype=torch.bool)
        elif 'image_exr' in fpath:  # for extraraw, both raw and gt are RGB
            return torch.ones(3, 256, 256)
        return torch.empty(0)  # Default for unexpected paths

    def mock_random_crops(self, ximg, yimg, whole_img_mask):
        num_crops = self.num_crops
        crop_size = self.crop_size
        if yimg.shape[-3] == 4:  # Bayer
            return (torch.ones(num_crops, 3, crop_size, crop_size), torch.ones(num_crops, 4, crop_size // 2, crop_size // 2), torch.ones(num_crops, 3, crop_size, crop_size, dtype=torch.bool))
        else:  # RGB
            return (torch.ones(num_crops, 3, crop_size, crop_size), torch.ones(num_crops, 3, crop_size, crop_size), torch.ones(num_crops, 3, crop_size, crop_size, dtype=torch.bool))

    monkeypatch.setattr(pt_helpers_dep, 'fpath_to_tensor', mock_fpath_to_tensor)

    # We also need to mock the functions being used in base_dataset
    with patch('src.rawnind.dataset.base_dataset.cv2.imread', return_value=np.ones((256,256,3), dtype=np.float32)), \
         patch('src.rawnind.dataset.base_dataset.rp.imread', return_value=np.ones((256,256), dtype=np.uint16)), \
         patch('src.rawnind.dataset.base_dataset.RawImageDataset.random_crops', mock_random_crops):
        yield

@pytest.mark.parametrize(
    "dataset_type, config_class, content_fpath_fixture, extra_params, expected_x_shape, expected_y_shape",
    [
        # Clean-Clean Bayer
        ("clean-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True}, (1, 3, 256, 256), (1, 4, 128, 128)),
        # Clean-Clean RGB
        ("clean-rgb", RgbDatasetConfig, "mock_content_fpath_extraraw", {'is_bayer': False}, (1, 3, 256, 256), (1, 3, 256, 256)),
        # Noisy Bayer
        ("noisy-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True, 'bayer_only': True}, (1, 3, 256, 256), (1, 4, 128, 128)),
        # Noisy RGB
        ("noisy-rgb", RgbDatasetConfig, "mock_content_fpath_extraraw", {'is_bayer': False}, (1, 3, 256, 256), (1, 3, 256, 256)),
    ]
)
def test_create_training_dataset_shapes(
    dataset_type, config_class, content_fpath_fixture, extra_params, expected_x_shape, expected_y_shape, request
):
    """Test that `create_training_dataset` produces datasets with correct tensor shapes."""
    content_fpaths = request.getfixturevalue(content_fpath_fixture)
    if not isinstance(content_fpaths, list):
         content_fpaths = [content_fpaths]

    config = DatasetConfig(
        dataset_type=dataset_type,
        data_format="clean-clean" if "clean" in dataset_type else "clean-noisy",
        input_channels=3,
        output_channels=4 if "bayer" in dataset_type else 3,
        content_fpaths=content_fpaths,
        num_crops_per_image=1,
        crop_size=256,
        batch_size=1,
        test_reserve_images=[],
        match_gain=extra_params.pop('match_gain', False),
        config=config_class(**extra_params)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})

    assert len(ds) > 0

    image_batch = next(iter(ds))

    assert image_batch["noisy_images"].shape == expected_x_shape
    assert image_batch["clean_images"].shape == expected_y_shape
    assert image_batch["masks"].shape == (expected_x_shape[0], expected_x_shape[1], expected_x_shape[2], expected_x_shape[3])

    if "bayer" in dataset_type:
        assert "bayer_info" in image_batch
@pytest.mark.parametrize("crop_size, num_crops, use_yimg, expected_x_shape, expected_y_shape, expected_mask_shape", [
    (128, 4, True, (4, 3, 128, 128), (4, 4, 64, 64), (4, 3, 128, 128)), # Bayer scenario
    (128, 4, False, (4, 3, 128, 128), None, (4, 3, 128, 128)),
    (256, 2, True, (2, 3, 256, 256), (2, 4, 128, 128), (2, 3, 256, 256)), # Bayer scenario
])
def test_rawimagedataset_random_crops(
    crop_size, num_crops, use_yimg,
    expected_x_shape, expected_y_shape, expected_mask_shape, mock_content_fpath_rawnind
):
    """Test RawImageDataset.random_crops for correct output shapes and logic."""
    config = DatasetConfig(
        dataset_type="clean-bayer",
        data_format="clean-clean",
        input_channels=3,
        output_channels=4,
        content_fpaths=[mock_content_fpath_rawnind],
        num_crops_per_image=num_crops,
        crop_size=crop_size,
        batch_size=1,
        test_reserve_images=[],
        config=BayerDatasetConfig(is_bayer=True, bayer_only=True)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': [mock_content_fpath_rawnind]})

    image_batch = next(iter(ds))

    assert image_batch["noisy_images"].shape == expected_x_shape
    if use_yimg:
        assert image_batch["clean_images"].shape == expected_y_shape
    assert image_batch["masks"].shape == expected_mask_shape
@pytest.mark.parametrize("crop_size, use_yimg, input_is_rgb, expected_x_shape, expected_y_shape, expected_mask_shape", [
    (128, True, True, (3, 128, 128), (3, 128, 128), (3, 128, 128)), # RGB scenario
    (128, True, False, (3, 128, 128), (4, 64, 64), (3, 128, 128)), # Bayer scenario
    (256, False, True, (3, 256, 256), None, (3, 256, 256)),
])
def test_rawimagedataset_center_crop(
    crop_size, use_yimg, input_is_rgb,
    expected_x_shape, expected_y_shape, expected_mask_shape,  mock_content_fpath_rawnind, mock_content_fpath_extraraw
):
    """Test RawImageDataset.center_crop for correct output shapes and logic."""
    if input_is_rgb:
        dataset_type = "clean-rgb"
        content_fpaths = mock_content_fpath_extraraw
        config_class = RgbDatasetConfig
        extra_params = {'is_bayer': False}
        output_channels = 3

    else:
        dataset_type = "clean-bayer"
        content_fpaths = [mock_content_fpath_rawnind]
        config_class = BayerDatasetConfig
        extra_params = {'is_bayer': True, 'bayer_only': True}
        output_channels = 4

    config = DatasetConfig(
        dataset_type=dataset_type,
        data_format="clean-clean",
        input_channels=3,
        output_channels=output_channels,
        content_fpaths=content_fpaths,
        num_crops_per_image=1,
        crop_size=crop_size,
        batch_size=1,
        test_reserve_images=[],
        center_crop=True,
        config=config_class(**extra_params)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})
    image_batch = next(iter(ds))

    assert image_batch["noisy_images"][0].shape == expected_x_shape
    if use_yimg:
        assert image_batch["clean_images"][0].shape == expected_y_shape
    assert image_batch["masks"][0].shape == expected_mask_shape
class TestDataLoader:
    """Mixin-like helper that yields processed images without using PyTorch DataLoader.

    Classes inheriting this should implement get_images(), which yields dictionaries
    with keys like x_crops, y_crops, mask_crops, and optionally rgb_xyz_matrix.
    """
    OUTPUTS_IMAGE_FILES = False

    def __init__(self, **kwargs):
        """Accept arbitrary keyword arguments for configuration; subclasses may consume them."""
        pass

    def __getitem__(self, i):
        """Disabled random access; use the iterator or get_images() instead."""
        raise TypeError(
            f"{type(self).__name__} is its own data loader: "
            "call get_images instead of __getitem__ (or use built-in __iter__)."
        )

    def __iter__(self):
        """Iterator alias for get_images()."""
        return self.get_images()

    def batched_iterator(self):
        """Yield batched tensors by adding a batch dimension when needed.

        If get_images() yields per-image tensors of shape [C,H,W], they are expanded
        to [1,C,H,W]; if they already include [N,C,H,W], they are passed through.
        """
        single_to_batch = lambda x: torch.unsqueeze(x, 0)
        identity = lambda x: x
        if hasattr(
                self, "get_images"
        ):  # TODO should combine this ifelse with an iterator selection
            for res in self.get_images():
                batch_fun = single_to_batch if res["y_crops"].dim() == 3 else identity
                res["y_crops"] = batch_fun(res["y_crops"]).float()
                res["x_crops"] = batch_fun(res["x_crops"]).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res
        else:
            for i in range(len(self._dataset)):
                res = self.__getitem__(i)
                batch_fun = single_to_batch if res["y_crops"].dim() == 3 else identity
                res["y_crops"] = batch_fun(res["y_crops"]).float()
                res["x_crops"] = batch_fun(res["x_crops"]).float()
                res["mask_crops"] = batch_fun(res["mask_crops"])
                if "rgb_xyz_matrix" in res:
                    res["rgb_xyz_matrix"] = batch_fun(res["rgb_xyz_matrix"])
                yield res

    @staticmethod
    def _content_fpaths_to_test_reserve(content_fpaths: list[str]) -> list[str]:
        """Extract test reserve directory names from dataset content files.

        Parses YAML content files to extract directory names (excluding 'gt' directories)
        that should be reserved for testing purposes, ensuring proper train/test splits.

        Args:
            content_fpaths: List of paths to YAML files containing dataset metadata.

        Returns:
            List of directory names to reserve for testing.
        """
        # add all images to test_reserve:
        test_reserve = []
        for content_fpath in content_fpaths:
            for image in yaml.safe_load(Path(content_fpath).read_text()):
                # get the directory name of the image (not the full path)
                dn = os.path.basename(os.path.dirname(image["f_fpath"]))
                if dn == "gt":
                    continue
                test_reserve.append(dn)
        return test_reserve
class CleanProfiledRGBNoisyBayerImageCropsTestDataloader(
    TestDataLoader, CleanDataset
):
    """Dataloader of clean (profiled RGB) - noisy (Bayer) images crops from rawNIND."""

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = 0.035,
            mask_mean_min: float = 0.8,
            toy_dataset=False,
            match_gain: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        config = DatasetConfig(
            dataset_type="noisy-bayer",
            data_format="clean-noisy",
            input_channels=3,
            output_channels=4,
            content_fpaths=content_fpaths,
            num_crops_per_image=1,
            crop_size=crop_size,
            batch_size=1,
            test_reserve_images=test_reserve,
            config=BayerDatasetConfig(is_bayer=True, bayer_only=bayer_only)
        )
        super().__init__(config=config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})


class CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader(
    TestDataLoader, CleanDataset
):
    """Dataloader of clean (profiled RGB) - noisy (profiled RGB) images crops from rawNIND."""

    def __init__(
            self,
            content_fpaths: list[str],
            crop_size: int,
            test_reserve,
            bayer_only: bool,
            alignment_max_loss: float = 0.035,
            mask_mean_min: float = 0.8,
            toy_dataset=False,
            match_gain: bool = False,
            arbitrary_proc_method: bool = False,
            min_msssim_score: Optional[float] = 0.0,
            max_msssim_score: Optional[float] = 1.0,
    ):
        config = DatasetConfig(
            dataset_type="noisy-rgb",
            data_format="clean-noisy",
            input_channels=3,
            output_channels=3,
            content_fpaths=content_fpaths,
            num_crops_per_image=1,
            crop_size=crop_size,
            batch_size=1,
            test_reserve_images=test_reserve,
            config=RgbDatasetConfig(is_bayer=False)
        )
        super().__init__(config=config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})
@pytest.mark.parametrize(
    "dataset_type, config_class, content_fpath_fixture, extra_params, expected_gain",
    [
        # Noisy Bayer
        ("noisy-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True, 'bayer_only': True, 'match_gain': False}, 1.5),
        # Noisy RGB
        ("noisy-rgb", RgbDatasetConfig, "mock_content_fpath_extraraw", {'is_bayer': False, 'match_gain': False}, 1.0),
        # Clean-Clean Bayer with match_gain=True (gain should be 1.0)
        ("clean-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True, 'match_gain': True}, 1.0),
    ]
)
def test_create_training_dataset_gain(
    dataset_type, config_class, content_fpath_fixture, extra_params, expected_gain, request
):
    """Test that `create_training_dataset` correctly handles gain."""
    content_fpaths = request.getfixturevalue(content_fpath_fixture)
    if not isinstance(content_fpaths, list):
        content_fpaths = [content_fpaths]

    config = DatasetConfig(
        dataset_type=dataset_type,
        data_format="clean-clean" if "clean" in dataset_type else "clean-noisy",
        input_channels=3,
        output_channels=4 if "bayer" in dataset_type else 3,
        content_fpaths=content_fpaths,
        num_crops_per_image=1,
        crop_size=256,
        batch_size=1,
        match_gain=match_gain,
        test_reserve_images=[],
        config=config_class(**extra_params)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})

    image_batch = next(iter(ds))

    assert "gain" in image_batch
    assert image_batch["gain"] == pytest.approx(expected_gain)
@pytest.mark.parametrize(
    "dataset_type, config_class, content_fpath_fixture, extra_params, expected_gain",
    [
        # Noisy Bayer
        # Noisy Bayer
        ("noisy-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True, 'bayer_only': True, 'match_gain': False}, 1.5),
        # Noisy RGB
        ("noisy-rgb", RgbDatasetConfig, "mock_content_fpath_extraraw", {'is_bayer': False, 'match_gain': False}, 1.0),
        # Clean-Clean Bayer with match_gain=True (gain should be 1.0)
        ("clean-bayer", BayerDatasetConfig, "mock_content_fpath_rawnind", {'is_bayer': True, 'match_gain': True}, 1.0),
    ]
)
def test_create_training_dataset_gain(
    dataset_type, config_class, content_fpath_fixture, extra_params, expected_gain, request
):
    """Test that `create_training_dataset` correctly handles gain."""
    content_fpaths = request.getfixturevalue(content_fpath_fixture)
    if not isinstance(content_fpaths, list):
        content_fpaths = [content_fpaths]

    config = DatasetConfig(
        dataset_type=dataset_type,
        data_format="clean-clean" if "clean" in dataset_type else "clean-noisy",
        input_channels=3,
        output_channels=4 if "bayer" in dataset_type else 3,
        content_fpaths=content_fpaths,
        num_crops_per_image=1,
        crop_size=256,
        batch_size=1,
        test_reserve_images=[],
        match_gain=extra_params.pop('match_gain', False),
        config=config_class(**extra_params)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': content_fpaths})

    image_batch = next(iter(ds))

    assert "gain" in image_batch
    assert image_batch["gain"] == pytest.approx(expected_gain)
@pytest.mark.parametrize(
    "min_msssim, max_msssim, expected_len",
    [
        (0.0, 1.0, 2),  # No filtering
        (0.9, 1.0, 1),  # Filter out one image
        (0.96, 1.0, 0), # Filter out both images
    ]
)
def test_create_training_dataset_msssim_filter(
    min_msssim, max_msssim, expected_len, mini_test_dataset
):
    """Test that `create_training_dataset` correctly filters by MS-SSIM score."""

    config = DatasetConfig(
        dataset_type="noisy-bayer",
        data_format="clean-noisy",
        input_channels=3,
        output_channels=4,
        content_fpaths=[mini_test_dataset['yaml_path']],
        num_crops_per_image=1,
        crop_size=256,
        batch_size=1,
        test_reserve_images=[],
        quality_thresholds={
            'min_image_quality_score': min_msssim,
            'max_image_quality_score': max_msssim
        },
        config=BayerDatasetConfig(is_bayer=True, bayer_only=True)
    )

    ds = create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': [mini_test_dataset['yaml_path']]})

    assert len(ds) >= expected_len
def test_create_training_dataset_no_images(tmp_path):
    """Test that `create_training_dataset` raises ValueError when no images are found."""

    # Create an empty yaml file
    yaml_path = tmp_path / 'empty_content.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump({'images': []}, f)

    config = DatasetConfig(
        dataset_type="noisy-bayer",
        data_format="clean-noisy",
        input_channels=3,
        output_channels=4,
        content_fpaths=[str(yaml_path)],
        num_crops_per_image=1,
        crop_size=256,
        batch_size=1,
        test_reserve_images=[],
        config=BayerDatasetConfig(is_bayer=True, bayer_only=True)
    )

    with pytest.raises(ValueError):
        create_training_dataset(config, data_paths={'noise_dataset_yamlfpaths': [str(yaml_path)]})
