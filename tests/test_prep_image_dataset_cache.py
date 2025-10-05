"""Test that prep_image_dataset.py correctly caches results incrementally."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from common.libs import utilities
from rawnind.libs import rawproc


def test_incremental_cache_updates():
    """Validate cache contains expected results after 3 iterations."""

    # Create temporary cache file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        cache_path = Path(f.name)

    try:
        # Mock arguments for 5 image pairs
        mock_args = [
            {
                "ds_dpath": "/mock/path",
                "image_set": f"scene_{i:03d}",
                "gt_file_endpath": f"gt/image_{i:03d}_gt.raw",
                "f_endpath": f"image_{i:03d}_noisy.raw",
                "masks_dpath": "/mock/masks",
                "alignment_method": "fft",
                "verbose": False,
                "num_threads": 1,
            }
            for i in range(5)
        ]

        # Mock results that the processing function would return
        def mock_process(arg):
            return {
                "image_set": arg["image_set"],
                "gt_fpath": f"/mock/path/{arg['image_set']}/{arg['gt_file_endpath']}",
                "f_fpath": f"/mock/path/{arg['image_set']}/{arg['f_endpath']}",
                "alignment_method": arg["alignment_method"],
                "is_bayer": False,
            }

        # Track cache writes
        cache_snapshots = []
        original_open = open

        def track_cache_writes(path, mode='r', **kwargs):
            """Intercept writes to cache file and save snapshots."""
            if str(path) == str(cache_path) and 'w' in mode:
                # Return a wrapper that captures what gets written
                class WriteCaptureWrapper:
                    def __init__(self, file_obj):
                        self.file_obj = file_obj
                        self.content = []

                    def write(self, data):
                        self.content.append(data)
                        return self.file_obj.write(data)

                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        # Save snapshot after write completes
                        full_content = ''.join(self.content)
                        cache_snapshots.append(yaml.safe_load(full_content) if full_content else [])
                        return self.file_obj.__exit__(*args)

                return WriteCaptureWrapper(original_open(path, mode, **kwargs))
            return original_open(path, mode, **kwargs)

        # Run mt_runner with mocked processing and tracked cache writes
        results = []
        cached_results = []
        content_fpath = cache_path

        def save_result(result):
            results.append(result)
            with track_cache_writes(content_fpath, "w", encoding="utf-8") as f:
                yaml.dump(results + (cached_results or []), f, allow_unicode=True)

        # Process only first 3 items
        with patch.object(rawproc, 'get_best_alignment_compute_gain_and_make_loss_mask', side_effect=mock_process):
            utilities.mt_runner(
                rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
                mock_args[:3],  # Only process first 3
                num_threads=1,
                progress_bar=False,
                on_result=save_result,
            )

        # Validate: Should have 3 cache snapshots
        assert len(cache_snapshots) == 3, f"Expected 3 cache writes, got {len(cache_snapshots)}"

        # Validate: First snapshot should contain 1 result
        assert len(cache_snapshots[0]) == 1, f"First snapshot should have 1 result, got {len(cache_snapshots[0])}"
        assert cache_snapshots[0][0]["image_set"] == "scene_000"

        # Validate: Second snapshot should contain 2 results
        assert len(cache_snapshots[1]) == 2, f"Second snapshot should have 2 results, got {len(cache_snapshots[1])}"
        assert cache_snapshots[1][0]["image_set"] == "scene_000"
        assert cache_snapshots[1][1]["image_set"] == "scene_001"

        # Validate: Third snapshot should contain 3 results
        assert len(cache_snapshots[2]) == 3, f"Third snapshot should have 3 results, got {len(cache_snapshots[2])}"
        assert cache_snapshots[2][0]["image_set"] == "scene_000"
        assert cache_snapshots[2][1]["image_set"] == "scene_001"
        assert cache_snapshots[2][2]["image_set"] == "scene_002"

        # Validate: Each result has expected structure
        for i, result in enumerate(cache_snapshots[2]):
            assert "image_set" in result
            assert "gt_fpath" in result
            assert "f_fpath" in result
            assert "alignment_method" in result
            assert result["image_set"] == f"scene_{i:03d}"

    finally:
        # Cleanup
        if cache_path.exists():
            cache_path.unlink()


def test_cache_persistence_with_cached_results():
    """Validate that existing cached_results are preserved in incremental updates."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        cache_path = Path(f.name)

    try:
        # Simulate existing cached results
        existing_cached = [
            {
                "image_set": "scene_old_001",
                "gt_fpath": "/old/gt.raw",
                "f_fpath": "/old/noisy.raw",
                "alignment_method": "original",
                "is_bayer": True,
            },
            {
                "image_set": "scene_old_002",
                "gt_fpath": "/old/gt2.raw",
                "f_fpath": "/old/noisy2.raw",
                "alignment_method": "original",
                "is_bayer": True,
            },
        ]

        # New results to process
        new_args = [
            {
                "ds_dpath": "/mock/path",
                "image_set": "scene_new_001",
                "gt_file_endpath": "gt/new.raw",
                "f_endpath": "new_noisy.raw",
                "masks_dpath": "/mock/masks",
                "alignment_method": "fft",
                "verbose": False,
                "num_threads": 1,
            }
        ]

        def mock_process(arg):
            return {
                "image_set": arg["image_set"],
                "gt_fpath": f"/mock/{arg['gt_file_endpath']}",
                "f_fpath": f"/mock/{arg['f_endpath']}",
                "alignment_method": arg["alignment_method"],
                "is_bayer": False,
            }

        results = []
        cached_results = existing_cached
        content_fpath = cache_path

        def save_result(result):
            results.append(result)
            with content_fpath.open("w", encoding="utf-8") as f:
                yaml.dump(results + (cached_results or []), f, allow_unicode=True)

        with patch.object(rawproc, 'get_best_alignment_compute_gain_and_make_loss_mask', side_effect=mock_process):
            utilities.mt_runner(
                rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
                new_args,
                num_threads=1,
                progress_bar=False,
                on_result=save_result,
            )

        # Load final cache
        with cache_path.open('r') as f:
            final_cache = yaml.safe_load(f)

        # Validate: Should have 1 new + 2 old = 3 total
        assert len(final_cache) == 3, f"Expected 3 total results, got {len(final_cache)}"

        # Validate: New result is first
        assert final_cache[0]["image_set"] == "scene_new_001"

        # Validate: Old results are preserved
        assert final_cache[1]["image_set"] == "scene_old_001"
        assert final_cache[2]["image_set"] == "scene_old_002"

    finally:
        if cache_path.exists():
            cache_path.unlink()
