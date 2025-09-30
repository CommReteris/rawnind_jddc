# Suggested Commands
- **Run full test suite**: `python -m pytest src/rawnind/tests`
- **Run package-specific tests**: `python -m pytest src/rawnind/dataset/tests` (or substitute `inference`, `training`, `dependencies`)
- **Lint (when required)**: `pylint src/rawnind`
- **Prepare dataset crops**: `python tools/crop_datasets.py --dataset rawnind`
- **Precompute alignment/masks**: `python tools/prep_image_dataset.py`
- **Train denoise+compression model**: `python training/train_dc_bayer2prgb.py --config dependencies/configs/train_dc_bayer2prgb.yaml`
- **Train denoiser model**: `python training/train_denoiser_bayer2prgb.py --config dependencies/configs/train_denoise_bayer2prgb.yaml`
- **Run inference on images**: `python tools/denoise_image.py --config dependencies/configs/test_reserve.yaml --load_path <model_dir>`