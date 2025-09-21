# RawNIND: Modular Image Processing Framework

RawNIND provides a modular framework for joint image denoising, demosaicing, and compression using deep learning. The codebase has been refactored into a clean, maintainable architecture organized into four main packages:

- **inference/**: Model loading, prediction, and deployment tools
- **training/**: Training loops, optimization, and experiment management
- **dataset/**: Dataset loading, preprocessing, and validation
- **dependencies/**: Shared utilities, configurations, and core libraries

## Installation

### Core Dependencies

```bash
# PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core processing libraries
pip install colour-science pytorch-msssim ConfigArgParse rawpy imageio
pip install opencv-python numpy scipy matplotlib tqdm requests pyyaml ptflops
```

### Optional Dependencies

For OpenEXR support:
```bash
pip install OpenEXR
# Alternative for some systems:
pip install https://github.com/jamesbowman/openexrpython/archive/master.zip
```

For development:
```bash
pip install pytest pytest-cov black isort mypy
```

### System-specific Installation

#### Arch Linux:
```bash
# PyTorch with ROCm or CUDA
pacman -S python-pytorch-opt-rocm python-torchvision-rocm  # or cuda versions

# Additional libraries
pacman -S libraw python-rawpy python-openexr python-opencv python-colour-science
pacman -S python-pytorch-msssim-git python-configargparse python-pytorch-piqa-git
pacman -S python-tqdm python-colorspacious python-ptflops openimageio
```

#### Slurm Environment:
```bash
module load GCC CUDA LibTIFF PyYAML PyTorch libpng libjpeg-turbo
```


## Datasets

### Gathering

RawNIND is available on https://dataverse.uclouvain.be/dataset.xhtml?persistentId=doi:10.14428/DVN/DEQCIM

It can be downloaded with the following commands: `curl -s "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId=doi:10.14428/DVN/DEQCIM" | jq -r '.data.latestVersion.files[] | "https://dataverse.uclouvain.be/api/access/datafile/\(.dataFile.id)"' | wget -c -i -`

This will download a flat structure. The data loaders and pre-processors in this work expect the structure described in the following subsection (datasets/src/<Bayer or X-Trans>/<SET_NAME>/<"gt/" if applicable><IMAGE.EXT>).

## Dataset Preparation

### RawNIND Dataset (Clean-noisy / paired images)

RawNIND files are organized as follows in `../../datasets/RawNIND`:

- `src/<Bayer or X-Trans>/<SET_NAME>/<IMAGEY>.<EXT>`
- `src/<Bayer or X-Trans>/<SET_NAME>/gt/<IMAGEX>.<EXT>`
- `proc/lin_rec2020/<SET_NAME>/<IMAGEY.EXT>`
- `proc/lin_rec2020/<SET_NAME>/gt/<IMAGEX>.<EXT>`
- `proc/lin_rec2020/<SET_NAME>/gt/<IMAGEX>.<EXT>.xmp` (processing pipeline for testing)
- `proc/dt/<SET_NAME>/<IMAGEY>_aligned_to_<IMAGEX>.<EXT>` (manually processed test images)
- `proc/dt/<SET_NAME>/gt/<IMAGEX>_aligned_to_<IMAGEY>.<EXT>`
- `masks/<IMAGEX-IMAGEY.EXT>.png`
- `metadata/xProfiledRGB_yBayer.yaml`
- `metadata/xProfiledRGB_yProfiledRGB.yaml`

```bash
# Optional: full-size images (not needed by the model)
# python -m src.rawnind.tools.make_hdr_rawnind_files

# Demosaic X-Trans files and convert to pRGB OpenEXR with darktable-cli
python -m src.rawnind.tools.xtrans_to_openexr_dataset

# Pre-crop dataset images for faster training loading
python -m src.rawnind.tools.crop_datasets --dataset rawnind

# Compute images alignment, masks, gains
python -m src.rawnind.tools.prep_image_dataset

# For additional camera/test images
# python -m src.rawnind.tools.prep_image_dataset --dataset RawNIND_Bostitch

# Copy mask overwrites
cp ../../datasets/RawNIND/masks_overwrite/* ../../datasets/RawNIND/masks/
```

#### Optional: Compute MS-SSIM Loss (for filtered testing)

```bash
python -m src.rawnind.tools.add_msssim_score_to_dataset_yaml_descriptor
```

### RawNIND Manual Processing Test Images

```bash
# Generate dataset descriptor
python -m src.rawnind.dataset.manual_processing

# Compute MS-SSIM losses for filtered testing
python -m src.rawnind.tools.add_msssim_score_to_dataset_yaml_descriptor \
    --dataset_descriptor_fpath ../../datasets/RawNIND/manproc_test_descriptor.yaml
```

### Additional Camera Train/Test Images

#### Training Requirements:
```bash
# Pre-crop dataset images
python -m src.rawnind.tools.crop_datasets --dataset RawNIND_Bostitch

# Compute alignment, masks, gains
python -m src.rawnind.tools.prep_image_dataset --dataset RawNIND_Bostitch

# Optional: full-size debayered images
python -m src.rawnind.tools.make_hdr_rawnind_files \
    --data_dpath ../../datasets/RawNIND_Bostitch
```

#### Testing Requirements:
```bash
# Create manual processing descriptor
python -m src.rawnind.dataset.manual_processing \
    --test_descriptor_fpath ../../datasets/RawNIND_Bostitch/manproc_test_descriptor.yaml \
    --rawnind_content_fpath ../../datasets/RawNIND_Bostitch/RawNIND_Bostitch_masks_and_alignments.yaml \
    --test_reserve_fpath config/test_reserve_extdata.yaml

# Compute MS-SSIM scores
python -m src.rawnind.tools.add_msssim_score_to_dataset_yaml_descriptor \
    --dataset_descriptor_fpath ../../datasets/RawNIND_Bostitch/manproc_test_descriptor.yaml
```




### ExtraRaw Dataset (Clean-clean / unpaired images) for Training

#### PIXL.US Dataset

```bash
# Download from PIXL.US (run multiple times until ~41GB)
cd <TEMPORARY_DIRECTORY>
rsync -avL rsync://raw.pixls.us/data/ raw-pixls-us-data/

# Process downloaded images
python -m src.rawnind.tools.gather_raw_gt_images \
    --orig_dpath <TEMPORARY_DIRECTORY>/raw-pixls-us-data/ \
    --orig_name raw-pixls
```

#### Custom Raw Images (e.g., trougnouf-ISO_LE_100)

```bash
# Initial run
python -m src.rawnind.tools.gather_raw_gt_images \
    --orig_name trougnouf \
    --orig_dpath /path/to/your/raw/images

# Update with new images
python -m src.rawnind.tools.gather_raw_gt_images \
    --overwrite \
    --orig_name trougnouf \
    --orig_dpath '/path/to/new/images/2022/'
```

#### Dataset Cleanup and Processing

Remove duplicate files:
```bash
cd ../../datasets/extraraw
rmlint . -S l
./rmlint.sh sh:remove  # Use -d for non-interactive mode
rm rmlint.*  # Clean up rmlint files
cd ../../src/rawnind/
```

Process ground-truth images to linear Rec.2020:
```bash
python -m src.rawnind.tools.make_hdr_extraraw_files
# Review and delete any files that couldn't be read
bash logs/make_hdr_extraraw_files.py.log
```

Validate dataset integrity:
```bash
python -m src.rawnind.tools.check_dataset
```

Pre-crop for faster training:
```bash
# Crop extraraw dataset
python -m src.rawnind.tools.crop_datasets --dataset extraraw
python -m src.rawnind.tools.prep_image_dataset_extraraw
```

**ExtraRaw file organization** in `../../datasets/extraraw`:
- `<SET_NAME>/src/<Bayer or X-Trans>/<IMAGE.EXT>`
- `<SET_NAME>/src/proc/lin_rec2020/<IMAGE.EXT>.exr`

### ExtraRaw PlayRaw (Unpaired) Manual Processing for Testing

```bash
# Generate linear Rec.2020 images and crop list
python -m src.rawnind.tools.make_hdr_extraraw_files
python -m src.rawnind.tools.prep_image_dataset_extraraw

# Create manual processing dataset descriptor
python -m src.rawnind.dataset.manual_processing \
    --rawnind_content_fpath ../../datasets/extraraw/play_raw_test/crops_metadata.yaml \
    --test_descriptor_fpath ../../datasets/extraraw/play_raw_test/manproc_test_descriptor.yaml \
    --unpaired_images \
    --test_reserve_fpath ""
```


## Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest

# Run specific package tests
pytest src/rawnind/inference/tests/
pytest src/rawnind/training/tests/
pytest src/rawnind/dataset/tests/
pytest src/rawnind/dependencies/tests/

# Run with coverage
pytest --cov=src/rawnind --cov-report=html
```

## Model Evaluation and Plotting

Test all trained models (this may take several days):

```bash
# Run comprehensive model testing
bash scripts/test_all_needed.sh

# Generate evaluation plots
python -m src.rawnind.tests.grapher
```

## Usage Examples

### Training a Model

```bash
# Train a denoiser for Bayer-to-profiledRGB conversion
python -m src.rawnind.train_denoiser_bayer2prgb

# Train a denoiser for profiledRGB-to-profiledRGB conversion
python -m src.rawnind.train_denoiser_prgb2prgb

# Train joint denoising and compression
python -m src.rawnind.train_dc_bayer2prgb
python -m src.rawnind.train_dc_prgb2prgb
```

### Running Inference

```bash
# Denoise an image using a trained model
python -m src.rawnind.inference.image_denoiser --input image.raw --model path/to/model
```

## Project Structure

```
src/rawnind/
├── inference/          # Model inference and deployment
│   ├── __init__.py
│   ├── base_inference.py
│   ├── image_denoiser.py
│   ├── inference_engine.py
│   ├── model_factory.py
│   ├── model_loader.py
│   └── tests/
├── training/           # Training loops and optimization
│   ├── __init__.py
│   ├── denoise_compress_trainer.py
│   ├── denoiser_trainer.py
│   ├── experiment_manager.py
│   ├── training_loops.py
│   └── tests/
├── dataset/            # Dataset handling and preprocessing
│   ├── __init__.py
│   ├── base_dataset.py
│   ├── bayer_datasets.py
│   ├── clean_datasets.py
│   ├── manual_processing.py
│   ├── rgb_datasets.py
│   ├── test_dataloaders.py
│   ├── validation_datasets.py
│   └── tests/
├── dependencies/       # Shared utilities and configurations
│   ├── __init__.py
│   ├── config_manager.py
│   ├── json_saver.py
│   ├── pt_losses.py
│   ├── pytorch_helpers.py
│   ├── utilities.py
│   └── tests/
├── models/             # Neural network architectures
├── config/             # Configuration files
├── tools/              # Utility scripts
└── tests/              # Integration tests
```

## Troubleshooting

### OpenEXR Installation Issues

If `pip install OpenEXR` fails:

1. Try installing from the AUR repository with patches:
   ```bash
   # On Arch Linux
   yay -S python-openexr
   ```

2. For manual installation:
   - Install OpenEXR system libraries
   - Add local OpenEXR paths to library and include directories in setup.py
   - Include arrays in setup.py configuration
   - Run `pip install .`

3. For local installations, you may need to set:
   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/lib
   ```

### Dataset Issues

- **X-Trans conversion filename bug**: Images converted from X-Trans format may be saved with incorrect filenames in YAML dataset descriptors (e.g., `DSCF1735.RAF.exr` saved as `DSCF1735.exr`)

### Performance Tips

- Pre-crop datasets for faster training loading
- Use appropriate batch sizes for your GPU memory
- Consider using mixed precision training for faster convergence
