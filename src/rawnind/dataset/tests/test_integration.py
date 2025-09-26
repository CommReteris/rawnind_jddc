import pytest
from unittest.mock import patch

from rawnind.dataset.clean_api import ConfigurableDataset, DatasetConfig
from rawnind.dependencies import load_yaml
from rawnind.dependencies.raw_processing import shift_images
from rawnind.dependencies.image_analysis import get_mask


@pytest.fixture
def mock_image_data():
    """Mock image data structure for YAML contents."""
    return {
        'f_bayer_fpath': 'dummy_noisy_bayer.exr',
        'gt_linrec2020_fpath': 'dummy_clean_rgb.exr',
        'mask_fpath': 'dummy_mask.exr',
        'best_alignment': [0, 0],
        'best_alignment_loss': 0.01,
        'mask_mean': 0.9,
        'rgb_msssim_score': 0.95,
        'rgb_xyz_matrix': [[0.7034, -0.0804, -0.1014], [-0.4420, 1.2564, 0.2058], [-0.0851, 0.1994, 0.5758]],
        'raw_gain': 1.5,
        'crops': [{'coordinates': [0, 0], 'gt_linrec2020_fpath': 'dummy_crop_clean.exr', 'f_bayer_fpath': 'dummy_crop_noisy.exr'}],
        'image_set': 'train',
        'is_bayer': True
    }


@pytest.fixture
def mock_config():
    """Mock DatasetConfig."""
    return DatasetConfig(
        dataset_type='bayer_pairs',
        data_format='clean_noisy',
        input_channels=4,
        output_channels=3,
        crop_size=64,
        num_crops_per_image=1,
        batch_size=8,
        data_pairing='x_y',
        match_gain=False,
        test=False,
        bayer_only=True,
        alignment_max_loss=0.035,
        mask_mean_min=0.8,
        min_msssim_score=0.0,
        max_msssim_score=1.0
    )


def test_yaml_loading_for_clean_noisy_pairs(mock_image_data, mock_config):
    """Test YAML metadata loading for clean-noisy pairs in ConfigurableDataset."""
    with patch('rawnind.dependencies.load_yaml') as mock_load:
        mock_load.return_value = [mock_image_data]

        dataset = ConfigurableDataset(
            config=mock_config,
            yaml_paths=['dummy.yaml'],
            test_reserve=[],
            toy_dataset=False
        )

        # Verify loading
        assert len(dataset) == 1
        assert len(dataset._dataset) == 1

        image = dataset._dataset[0]
        assert 'f_bayer_fpath' in image
        assert 'rgb_xyz_matrix' in image
        assert image['raw_gain'] == 1.5


def test_image_alignment_shift_images():
    """Test image alignment with shift_images in raw_processing."""
    # Create test images with known misalignment
    img_gt = torch.zeros(4, 4)
    img_gt[0, 0] = 1.0  # Mark position (0,0) in ground truth
    img_noisy = torch.zeros(4, 4)
    img_noisy[1, 1] = 1.0  # Misaligned by [1,1]

    # Expected after shifting noisy by [-1, -1] to align with gt
    expected_aligned = torch.zeros(4, 4)
    expected_aligned[0, 0] = 1.0

    # Test shift by [1, 1] (should move noisy to align with gt)
    aligned_gt, aligned_noisy = shift_images(img_gt, img_noisy, [1, 1])

    assert torch.allclose(aligned_gt, img_gt)  # gt unchanged
    assert torch.allclose(aligned_noisy, expected_aligned)  # noisy shifted to match gt at (0,0)

    # Test no shift
    aligned_gt_no, aligned_noisy_no = shift_images(img_gt, img_noisy, [0, 0])
    assert torch.allclose(aligned_gt_no, img_gt)
    assert torch.allclose(aligned_noisy_no, img_noisy)

    # Test negative shift
    expected_neg = torch.zeros(4, 4)
    expected_neg[2, 2] = 1.0  # Shift by [-1,-1] moves to (2,2)
    aligned_gt_neg, aligned_noisy_neg = shift_images(img_gt, img_noisy, [-1, -1])
    assert torch.allclose(aligned_gt_neg, img_gt)
    assert torch.allclose(aligned_noisy_neg, expected_neg)


def test_overexposure_masking():
    """Test overexposure masking in image_analysis.get_mask."""
    # Mock image with some pixels above and below threshold
    img_rgb = torch.tensor([[[0.5, 0.5, 0.5], [1.0, 1.0, 1.0]], [[0.8, 0.8, 0.8], [0.2, 0.2, 0.2]]])  # Shape (2, 3, 2, 2)
    metadata = {'overexposure_lb': 0.9}

    mask = get_mask(img_rgb, metadata)
    # Expected: True where max channel >= 0.9 (saturated), False otherwise
    expected_mask = torch.tensor([[[False, False], [True, False]], [[False, False], [False, False]]])
    assert torch.all(mask == expected_mask)

    # Test for Bayer image (4 channels)
    img_bayer = torch.tensor([[[[0.5], [1.0]], [[0.8], [0.2]]]])  # Shape (1, 4, 2, 2)
    mask_bayer = get_mask(img_bayer, metadata)
    # Expected: Interpolate to RGB, then mask
    expected_bayer_mask = torch.tensor([[[False, True], [False, False]]]).repeat(3, 1, 1)
    assert torch.all(mask_bayer == expected_bayer_mask)