# RawNIND Clean API Refactoring - Execution Guide

## Overview

This refactoring translates legacy dataset classes into a unified `ConfigurableDataset` with 100% domain logic fidelity. After completion, all legacy dataset files will be deleted.

## Before Starting

### 1. Read Collective Memory
Use the collectiveMemory MCP server to access the knowledge graph:
```
Read entities:
- RawNIND Clean API Refactoring
- ConfigurableDataset
- Legacy Dataset Classes
- Dataset Package Issues
- Critical Code Errors
- Integration Architecture
- All other related entities
```

### 2. Understand the Architecture

**Current State**: Legacy classes in 4 files (with duplicates and bugs)
**Target State**: Single ConfigurableDataset in clean_api.py with all logic

**Translation Pattern**:
```
4 Legacy Classes → 1 ConfigurableDataset with 4 conditional branches
├── Clean-Clean + Bayer branch
├── Clean-Clean + RGB branch  
├── Clean-Noisy + Bayer branch
└── Clean-Noisy + RGB branch
```

## Execution Sequence

Execute phases in strict order:

### Phase 1: Syntax Fixes (10 min)
**Prompt**: Read `phase1_syntax_fixes.md`
**Tasks**:
- Fix incomplete gain assignment in noisy_datasets.py:198
- Remove unreachable code in rgb_datasets.py:300-305
- Add missing `import random` in clean_api.py

**Verification**: All .py files compile without SyntaxError

### Phase 2: Translate ConfigurableDataset (180 min) - MAIN WORK
**Prompt**: Read `phase2_translate_configurabledataset.md`
**Tasks**:
- Update imports in clean_api.py
- Rewrite ConfigurableDataset.__init__() 
- Rewrite ConfigurableDataset._load_dataset()
- Rewrite ConfigurableDataset.__getitem__() with 4 branches
- Add ConfigurableDataset.get_mask() method
- Fix all bugs during translation

**Verification**: ConfigurableDataset loads real data, not random tensors

### Phase 3: Integration Function (45 min)
**Prompt**: Read `phase3_integration_function.md`
**Tasks**:
- Implement create_training_datasets() in clean_api.py
- Update __init__.py to export function

**Verification**: Function callable from training code

### Phase 4: Delete Legacy Files (15 min)
**Prompt**: Read `phase4_delete_legacy_files.md`
**Tasks**:
- Delete clean_datasets.py, noisy_datasets.py (duplicates)
- Delete bayer_datasets.py, rgb_datasets.py (reference impls)
- Update __init__.py imports
- Review validation_datasets.py and test_dataloaders.py

**Verification**: No duplicate definitions, all imports work

### Phase 5: Simplify CleanDataset (10 min)  
**Prompt**: Read `phase5_simplify_cleandataset.md`
**Tasks**:
- Simplify CleanDataset._create_underlying_dataset()
- Verify _standardize_batch_format() works

**Verification**: Batch format conversion works correctly

## Total Estimated Time
~4.5 hours focused work

## Success Criteria Checklist

After all phases complete:

- [ ] No syntax errors in any .py file
- [ ] ConfigurableDataset contains all logic from 4 legacy classes
- [ ] ConfigurableDataset loads real images (not random data)
- [ ] All bugs fixed (gain assignment, unreachable code)
- [ ] create_training_datasets() implemented and exported
- [ ] All legacy dataset files deleted
- [ ] No duplicate class definitions
- [ ] Single source of truth for all dataset types
- [ ] __init__.py exports only from clean_api and base_dataset
- [ ] Batch format standardization working
- [ ] All domain logic preserved: alignment, masks, crops, gains, quality filtering
- [ ] Tests pass (user validates)

## Critical Domain Logic to Verify

After Phase 2, verify ConfigurableDataset preserves:

1. ✓ YAML loading with quality score filtering
2. ✓ Test reserve split (training vs testing)
3. ✓ Alignment quality filtering (alignment_max_loss, mask_mean_min)
4. ✓ Bayer-only filtering for mixed datasets
5. ✓ Image alignment with shift_images()
6. ✓ Mask loading from files (clean-noisy) or computation (clean-clean)
7. ✓ Random crop selection with retry on insufficient pixels
8. ✓ Dynamic dataset modification (removing bad crops)
9. ✓ Gain matching (raw_gain for Bayer, rgb_gain for RGB)
10. ✓ Data pairing modes (x_y, x_x, y_y)
11. ✓ Arbitrary processing for experiments
12. ✓ Color matrix (rgb_xyz_matrix) for Bayer
13. ✓ Bayer alignment (even coordinates)
14. ✓ Clean-clean RGB returns only x_crops and mask_crops

## Troubleshooting

### If ConfigurableDataset still returns random data:
- Check that _load_dataset() actually loads from YAML files
- Verify content_fpaths not empty
- Check __getitem__() uses pt_helpers.fpath_to_tensor()

### If import errors after deletion:
- Update __init__.py to remove deleted file imports
- Verify validation_datasets.py and test_dataloaders.py updated or deleted

### If batch format wrong:
- Check _standardize_batch_format() key mapping
- Verify x_crops → clean_images, y_crops → noisy_images, mask_crops → masks

### If tests fail:
- Compare ConfigurableDataset output with legacy_rawds.py output on same input
- Check all 4 conditional branches implemented
- Verify data_pairing modes work correctly