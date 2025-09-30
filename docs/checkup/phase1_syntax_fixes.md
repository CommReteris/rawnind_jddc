# Phase 1: Syntax Fixes - ALREADY COMPLETE ✅

## Context
Read from collective memory: "Phase 1 Syntax Fixes", "Critical Code Errors"

## Objective
Fix 3 blocking syntax errors preventing code compilation.

## Tasks

### Task 1.1: Fix Incomplete Gain Assignment
**File**: `src/rawnind/dataset/noisy_datasets.py`
**Line**: 198
**Current**:
```python
output["gain"]
```
**Change to**:
```python
output["gain"] = 1.0
```

### Task 1.2: Remove Unreachable Code  
**File**: `src/rawnind/dataset/rgb_datasets.py`
**Lines**: 300-305
**Action**: Delete lines 300-305 entirely (keep line 299 `return output`)

### Task 1.3: Add Missing Import
**File**: `src/rawnind/dataset/clean_api.py`
**Location**: Near other imports at top of file (around line 10)
**Add**:
```python
import random
```

## Verification
```bash
python -m py_compile src/rawnind/dataset/noisy_datasets.py
python -m py_compile src/rawnind/dataset/rgb_datasets.py  
python -m py_compile src/rawnind/dataset/clean_api.py
```
All should complete without SyntaxError.

## Estimated Time
10 minutes