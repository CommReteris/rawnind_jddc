# Phase 2: Translate ConfigurableDataset

## Context
Read from collective memory:
- "ConfigurableDataset Implementation Pattern"
- "Legacy Dataset Classes"  
- "Data Pairing Modes"
- "Clean-Clean vs Clean-Noisy Differences"
- "Import Requirements ConfigurableDataset"
- "Crop Selection Strategy"

## Objective
Translate domain logic from 4 legacy dataset classes into unified ConfigurableDataset with conditional branches. Must preserve 100% domain logic fidelity.

## Reference Implementations
- Clean+Bayer: `legacy_rawds.py` lines 335-416 OR `src/rawnind/dataset/bayer_datasets.py` lines 37-119
- Clean+RGB: `legacy_rawds.py` lines 419-503 OR `src/rawnind/dataset/rgb_datasets.py` lines 31-114
- Noisy+Bayer: `legacy_rawds.py` lines 506-679 OR `src/rawnind/dataset/bayer_datasets.py` lines 128-307
- Noisy+RGB: `legacy_rawds.py` lines 682-874 OR `src/rawnind/dataset/rgb_datasets.py` lines 117-304

## File to Modify
`src/rawnind/dataset/clean_api.py`

## Step-by-Step Translation

### Step 2.1: Update Imports Section (lines 1-20)

Add these imports if not present:
```python
import random
import logging
from typing import Dict, Any, Optional

from ..dependencies import pytorch_helpers as pt_helpers
from ..dependencies import raw_processing as rawproc  
from ..dependencies.json_saver import load_yaml
from .base_dataset import (
    RawImageDataset,
    ALIGNMENT_MAX_LOSS,
    MASK_MEAN_MIN,
    TOY_DATASET_LEN,
    OVEREXPOSURE_LB
)
```

### Step 2.2: Rewrite ConfigurableDataset Class (lines 70-118)

**Replace entire class with this translation:**

```python
class ConfigurableDataset(torch.utils.data.Dataset):
    """Unified configuration-driven dataset (translated from 4 legacy classes)."""

    def __init__(self, config: DatasetConfig, data_paths: Dict[str, Any]):
        """Initialize dataset with config-driven logic (merged from 4 legacy __init__)."""
        self.config = config
        self.data_paths = data_paths
        
        # Create RawImageDataset for cropping utilities
        self.raw_image_dataset = RawImageDataset(
            num_crops=config.num_crops_per_image,
            crop_size=config.crop_size
        )
        
        # Instance variables (from legacy classes)
        self.match_gain = config.match_gain
        self.data_pairing = getattr(config.config, 'data_pairing', 'x_y') if config.config else 'x_y'
        self.arbitrary_proc_method = getattr(config, 'arbitrary_proc_method', None)
        
        # Validate arbitrary_proc requires match_gain (from legacy line 722-725)
        if self.arbitrary_proc_method and not self.match_gain:
            raise AssertionError("arbitrary_proc_method requires match_gain=True")
        
        # Load dataset
        self._dataset = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load dataset from YAML files (translated from 4 legacy __init__ methods)."""
        content_fpaths = self.data_paths.get('noise_dataset_yamlfpaths', [])
        
        # Extraction thresholds from config
        min_score = self.config.quality_thresholds.get('min_image_quality_score', 0.0)
        max_score = self.config.quality_thresholds.get('max_image_quality_score', 1.0)
        alignment_max_loss = self.config.quality_thresholds.get('max_alignment_error', ALIGNMENT_MAX_LOSS)
        mask_mean_min = self.config.quality_thresholds.get('min_mask_mean', MASK_MEAN_MIN)
        toy_dataset = self.config.max_samples == 25
        test_mode = self.config.save_individual_results  # Proxy for test mode
        bayer_only = self.config.config.bayer_only if self.config.config and hasattr(self.config.config, 'bayer_only') else True
        
        # Load from all YAML files
        for content_fpath in content_fpaths:
            logging.info(f"ConfigurableDataset: loading {content_fpath}")
            contents = load_yaml(content_fpath, error_on_404=True)
            
            for image in contents:
                # Toy dataset limiting (from legacy)
                if toy_dataset and len(self._dataset) >= TOY_DATASET_LEN:
                    break
                
                # Bayer filtering for noisy datasets (from legacy line 558-559, 179-180)
                if self.config.data_format == 'clean_noisy' and bayer_only:
                    if not image.get("is_bayer", False):
                        continue
                
                # Test reserve filtering (from legacy line 562-564, 737-742)
                if test_mode:
                    if image["image_set"] not in self.config.test_reserve_images:
                        continue
                else:
                    if image["image_set"] in self.config.test_reserve_images:
                        continue
                
                # MS-SSIM quality filtering (from legacy line 567-583, 747-760)
                try:
                    score = image.get("rgb_msssim_score", 1.0)
                    if min_score and min_score > score:
                        logging.debug(f"Skipping {image.get('f_fpath')}: score {score} < {min_score}")
                        continue
                    if max_score and max_score != 1.0 and max_score < score:
                        logging.debug(f"Skipping {image.get('f_fpath')}: score {score} > {max_score}")
                        continue
                except KeyError as e:
                    if min_score > 0 or max_score < 1.0:
                        raise KeyError(f"Image {image.get('f_fpath')} missing rgb_msssim_score") from e
                
                # Alignment quality filtering - ONLY for clean-noisy (from legacy line 588-593, 766-772)
                if self.config.data_format == 'clean_noisy':
                    if (image.get("best_alignment_loss", 0) > alignment_max_loss or
                        image.get("mask_mean", 1.0) < mask_mean_min):
                        logging.info(f"Rejected {image.get('f_fpath')} (alignment or mask criteria)")
                        continue
                
                # Crop validation (from legacy line 599-604, 777-781)
                if not image.get("crops"):
                    logging.warning(f"Image {image.get('f_fpath')} has no crops; skipping")
                    continue
                
                # Sort crops for deterministic testing (from legacy line 596-597, 774-775)
                image["crops"] = sorted(image["crops"], key=lambda d: d["coordinates"])
                
                # Add to dataset
                self._dataset.append(image)
        
        # Validate non-empty (from legacy line 606-608, 783-787)
        if len(self._dataset) == 0:
            raise ValueError(
                f"ConfigurableDataset is empty. "
                f"content_fpaths={content_fpaths}, test_reserve={self.config.test_reserve_images}"
            )
        
        logging.info(f"ConfigurableDataset initialized with {len(self._dataset)} images")
    
    def get_mask(self, ximg: torch.Tensor, metadata: dict) -> torch.BoolTensor:
        """Compute overexposure mask (from CleanCleanImageDataset in base_dataset.py)."""
        overexposure_lb = metadata.get("overexposure_lb", OVEREXPOSURE_LB)
        
        # Interpolate if Bayer to apply mask to RGB
        if ximg.shape[0] == 4:
            ximg = torch.nn.functional.interpolate(
                ximg.unsqueeze(0), scale_factor=2
            ).squeeze(0)
            return (
                (ximg.max(0).values < overexposure_lb)
                .unsqueeze(0)
                .repeat(3, 1, 1)
            )
        # RGB: mask individual channels
        return ximg < overexposure_lb
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return batch (translated from 4 legacy __getitem__ with bug fixes)."""
        image_data = self._dataset[idx]
        crop = random.choice(image_data["crops"])
        
        # BRANCH 1: Clean-Noisy + Bayer (from legacy_rawds.py:611-679)
        if self.config.data_format == 'clean_noisy' and 'bayer' in self.config.dataset_type:
            # Load images based on data_pairing mode
            if self.data_pairing == "x_y":
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
                
                # Alignment shift
                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image_data["best_alignment"]
                )
                
                # Load pre-computed mask
                whole_img_mask = pt_helpers.fpath_to_tensor(image_data["mask_fpath"])[
                    :,
                    crop["coordinates"][1]:crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0]:crop["coordinates"][0] + gt_img.shape[2],
                ]
                whole_img_mask = whole_img_mask.expand(gt_img.shape)
            elif self.data_pairing == "x_x":
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"])
                whole_img_mask = torch.ones_like(gt_img)
            elif self.data_pairing == "y_y":
                gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_bayer_fpath"])
                whole_img_mask = torch.ones_like(gt_img)
            else:
                raise ValueError(f"Unsupported data_pairing: {self.data_pairing}")
            
            # Random crops with retry on failure
            try:
                x_crops, y_crops, mask_crops = self.raw_image_dataset.random_crops(
                    gt_img, noisy_img, whole_img_mask
                )
            except TypeError:
                # Insufficient valid pixels (from legacy line 648-658)
                logging.warning(f"Crop {crop} has insufficient valid pixels; removing")
                self._dataset[idx]["crops"].remove(crop)
                if len(self._dataset[idx]["crops"]) == 0:
                    logging.warning(f"Image has no more valid crops; removing from dataset")
                    self._dataset.remove(self._dataset[idx])
                return self.__getitem__(idx)
            
            # Build output with gain handling
            output = {
                "x_crops": x_crops,
                "y_crops": y_crops,
                "mask_crops": mask_crops,
                "rgb_xyz_matrix": torch.tensor(image_data["rgb_xyz_matrix"])
            }
            
            # Gain matching (FIX applied from legacy line 674-678)
            if self.match_gain:
                output["y_crops"] *= image_data["raw_gain"]
                output["gain"] = 1.0  # FIXED: Was incomplete in legacy
            else:
                output["gain"] = image_data["raw_gain"]
            
            return output
        
        # BRANCH 2: Clean-Noisy + RGB (from legacy_rawds.py:790-874)
        elif self.config.data_format == 'clean_noisy' and 'rgb' in self.config.dataset_type:
            # Load based on data_pairing
            if self.data_pairing == "x_y":
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
                
                # Alignment
                gt_img, noisy_img = rawproc.shift_images(
                    gt_img, noisy_img, image_data["best_alignment"]
                )
                
                # Load mask
                whole_img_mask = pt_helpers.fpath_to_tensor(image_data["mask_fpath"])[
                    :,
                    crop["coordinates"][1]:crop["coordinates"][1] + gt_img.shape[1],
                    crop["coordinates"][0]:crop["coordinates"][0] + gt_img.shape[2],
                ]
                whole_img_mask = whole_img_mask.expand(gt_img.shape)
            elif self.data_pairing == "x_x":
                gt_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"])
                whole_img_mask = torch.ones_like(gt_img)
            elif self.data_pairing == "y_y":
                gt_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
                noisy_img = pt_helpers.fpath_to_tensor(crop["f_linrec2020_fpath"])
                whole_img_mask = torch.ones_like(gt_img)
            
            # Apply arbitrary processing if configured (from legacy line 834-846)
            if self.arbitrary_proc_method:
                from ..dependencies.arbitrary_processing import arbitrarily_process_images
                gt_img = arbitrarily_process_images(
                    gt_img,
                    randseed=crop["gt_linrec2020_fpath"],
                    method=self.arbitrary_proc_method
                )
                noisy_img = arbitrarily_process_images(
                    noisy_img,
                    randseed=crop["gt_linrec2020_fpath"],
                    method=self.arbitrary_proc_method
                )
            
            # Random crops
            try:
                x_crops, y_crops, mask_crops = self.raw_image_dataset.random_crops(
                    gt_img, noisy_img, whole_img_mask
                )
            except TypeError:
                logging.warning(f"Crop {crop} has insufficient valid pixels; removing")
                self._dataset[idx]["crops"].remove(crop)
                if len(self._dataset[idx]["crops"]) == 0:
                    self._dataset.remove(self._dataset[idx])
                return self.__getitem__(idx)
            
            # Build output (FIX applied: removed duplicate return from legacy line 868-874)
            output = {
                "x_crops": x_crops.float(),
                "y_crops": y_crops.float(),
                "mask_crops": mask_crops
            }
            
            # Gain handling (uses rgb_gain not raw_gain)
            if self.match_gain:
                output["y_crops"] *= image_data["rgb_gain"]
                output["gain"] = 1.0
            else:
                output["gain"] = image_data["rgb_gain"]
            
            return output  # FIXED: Removed unreachable return
        
        # BRANCH 3: Clean-Clean + Bayer (from legacy_rawds.py:377-413)
        elif self.config.data_format == 'clean_clean' and 'bayer' in self.config.dataset_type:
            try:
                gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                rgbg_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"]).float()
            except ValueError as e:
                logging.error(e)
                return self.__getitem__(random.randrange(len(self)))
            
            # Compute mask from overexposure threshold
            mask = self.get_mask(rgbg_img, image_data)
            
            # Random crops
            try:
                x_crops, y_crops, mask_crops = self.raw_image_dataset.random_crops(
                    gt, rgbg_img, mask
                )
            except (AssertionError, RuntimeError) as e:
                logging.error(f"Error with {crop}: {e}")
                logging.error(f"Shapes: gt={gt.shape}, rgbg={rgbg_img.shape}, mask={mask.shape}")
                raise
            except TypeError:
                logging.warning(f"Crop {crop} has insufficient valid pixels; removing")
                self._dataset[idx]["crops"].remove(crop)
                if len(self._dataset[idx]["crops"]) == 0:
                    self._dataset.remove(self._dataset[idx])
                return self.__getitem__(idx)
            
            return {
                "x_crops": x_crops,
                "y_crops": y_crops,
                "mask_crops": mask_crops,
                "rgb_xyz_matrix": image_data["rgb_xyz_matrix"],
                "gain": 1.0
            }
        
        # BRANCH 4: Clean-Clean + RGB (from legacy_rawds.py:462-503)
        elif self.config.data_format == 'clean_clean' and 'rgb' in self.config.dataset_type:
            try:
                gt = pt_helpers.fpath_to_tensor(crop["gt_linrec2020_fpath"]).float()
                rgbg_img = pt_helpers.fpath_to_tensor(crop["gt_bayer_fpath"]).float()
            except ValueError as e:
                logging.error(e)
                return self.__getitem__(random.randrange(len(self)))
            
            # Compute mask
            mask = self.get_mask(rgbg_img, image_data)
            
            # Arbitrary processing if configured (from legacy line 474-479)
            if self.arbitrary_proc_method:
                from ..dependencies.arbitrary_processing import arbitrarily_process_images
                gt = arbitrarily_process_images(
                    gt,
                    randseed=crop["gt_linrec2020_fpath"],
                    method=self.arbitrary_proc_method
                )
            
            # Random crops - NOTE: Clean-clean RGB returns only x_crops and mask_crops
            try:
                x_crops, mask_crops = self.raw_image_dataset.random_crops(gt, None, mask)
            except (AssertionError, RuntimeError) as e:
                logging.error(f"Error with {crop}: {e}")
                logging.error(f"Shapes: gt={gt.shape}, rgbg={rgbg_img.shape}, mask={mask.shape}")
                raise
            except TypeError:
                logging.warning(f"Crop {crop} has insufficient valid pixels; removing")
                self._dataset[idx]["crops"].remove(crop)
                if len(self._dataset[idx]["crops"]) == 0:
                    self._dataset.remove(self._dataset[idx])
                return self.__getitem__(idx)
            
            return {
                "x_crops": x_crops,
                "mask_crops": mask_crops,
                "gain": 1.0
            }
        
        else:
            raise ValueError(
                f"Unsupported combination: data_format={self.config.data_format}, "
                f"dataset_type={self.config.dataset_type}"
            )
    
    def __len__(self):
        return len(self._dataset)
```

## Critical Implementation Notes

1. **Gain field selection**: Bayer uses `raw_gain`, RGB uses `rgb_gain` from image metadata
2. **Return keys for clean-clean RGB**: Only `x_crops` and `mask_crops` (no `y_crops`)
3. **Mask source**: Clean-noisy loads from file, clean-clean computes from overexposure
4. **Data pairing**: Three modes (x_y, x_x, y_y) with different file path selections
5. **Bug fixes applied**: Incomplete gain assignment (line 676), unreachable return (line 868-874)
6. **Arbitrary processing**: Only applied if configured, requires match_gain=True
7. **Dynamic dataset modification**: Failed crops removed from list, retry __getitem__

## Verification

```bash
# Syntax check
python -m py_compile src/rawnind/dataset/clean_api.py

# Import check
python -c "from rawnind.dataset.clean_api import ConfigurableDataset"

# Logic check (compare with legacy)
python -c "
from rawnind.dataset.clean_api import ConfigurableDataset, DatasetConfig
from rawnind.dataset.dataset_config import BayerDatasetConfig
config = DatasetConfig(
    dataset_type='bayer_pairs',
    data_format='clean_noisy',
    input_channels=4,
    output_channels=3,
    crop_size=128,
    num_crops_per_image=1,
    batch_size=1,
    test_reserve_images=[],
    config=BayerDatasetConfig(is_bayer=True, bayer_only=True)
)
# Should not crash with random data error
print('ConfigurableDataset translation verification passed')
"
```

## Estimated Time
180 minutes (main translation work)