### A Chronical Detailing the Journey From Blank Slate to Training-Ready Batches

This guide does its best to identify and document every step required to go from a blank dev environment, to seeing your first batches get loaded onto your GPU for training the RawNIND models. Like a lot of the documentation I've added to this repo, It's mostly for my benefit as I navigate the codebase and wrap my head around all the moving parts. A predictable result is that there _are_ going to be errors and falsehoods because I simply do not have as much knowledge as (A) I wish I had, and (B) I think I have. Usually. Also, in the nascient era of ChatGPT & the like - these tools are simply _too good_ to not use for tasks like code comprehension and rapidly drafting _ad hoc_ backgrounders on subjects I lack familiarity with - the problem, of course, is that because of the way we train our AI, they all seem to learn that the best way to be seen as successful in the eyes of humanity, is to spout falsehoods and make shit up when you don't have facts, and the _real_ key to success is just to be _really convincing._

So just - read it all with a healthy dose of skepticism. It's by and large pretty darn accurate. 

Rengo

---

### Step 1: System Requirements & Environment Setup

#### 1.1 On the Reading and the Comprehending of the Read Me 
- It is located you know what if you can't find it I'm sorry this probably won't work out root folder of the repository

...
skipping right along
...

### Step 3: Dataset Acquisition & Preparation

#### 3.1 Download the Dataset
The RawNIND dataset may be downloaded from the official source:

```bash
cd src/rawnind
python -m rawnind.dataset download #TODO: this, for example, is total bullshit and did not exist at the time I had the AI write up instructions. Alas; it was a good idea so I should probably implement it.
```
Or, if you prefer your commands to actually be grounded in reality, you'll try this:

```console
echo("I have not read the README yet so this will make me")
cat README.md
```

This downloads the complete dataset to `src/rawnind/datasets/RawNIND/`. #TODO change to just datasets/RawNIND

#### 3.2 Dataset Structure Overview
After download, the dataset is organized as:
```
datasets/RawNIND/
├── src/
│   └── dataset_index.yaml          # Master index file
├── Bayer/                           # Bayer sensor images
│   ├── scene_001/
│   │   ├── clean/
│   │   │   └── img_001.dng
│   │   └── noisy/
│   │       ├── img_001_iso_100.dng
│   │       ├── img_001_iso_200.dng
│   │       └── ...
│   └── scene_002/
│       └── ...
└── RGB/                             # RGB sensor images (if applicable)
    └── ...
```

#### 3.3 Dataset Index Structure
The `dataset_index.yaml` file contains metadata for each scene:
```yaml
Bayer:
  scene_001:
    clean:
      - filename: img_001.dng
        sha1: abc123...
    noisy:
      - filename: img_001_iso_100.dng
        sha1: def456...
        iso: 100
    test_reserve: false
    unknown_sensor: false
```

---

### Step 4: Dataset Pipeline Components

The dataset preparation follows this pipeline (from `pipeline_diagram.md`):

```
DataIngestor → FileScanner → Downloader → Verifier → SceneIndexer → MetadataEnricher
```

#### 4.1 Pipeline Component Details

**DataIngestor**
- Entry point for dataset preparation
- Coordinates the entire pipeline
- Manages state and error handling

**FileScanner**
- Scans the dataset directory structure
- Identifies all image files (clean/noisy pairs)
- Outputs `SceneInfo` objects containing file paths

**Downloader**
- Downloads missing files from remote source
- Validates checksums during download
- Handles network failures and retries

**Verifier**
- Verifies file integrity using SHA1 checksums
- Compares against `dataset_index.yaml`
- Flags corrupted or missing files

**SceneIndexer**
- Builds internal index of all scenes
- Maps scene IDs to file locations
- Creates training/validation/test splits

**MetadataEnricher**
- Extracts EXIF data from RAW files
- Adds ISO, exposure, white balance info
- Enriches `ImageInfo` objects

---

### Step 5: Understanding Dataset Classes

The (legacy) project provides multiple (only two actually) dataset classes in `src/rawnind/libs/rawds.py`:

#### 5.1 Base Dataset Classes
What do these do? AH! a good question. They load the data you have downloaded and prepared (i.e., processed masks and aligned and augmented ) into the training framework! Good question. Dunno why that was omitted. When do you use these? Well I think you mostly don't if you just want to run inference, but the CleanNoisyDataset is what you would use for training.

**`RawImageDataset`**
- Base class for loading RAW images
- Handles .dng, .CR2, .NEF formats via rawpy
- Returns raw Bayer data or RGB images

**`CleanNoisyDataset`**
- Pairs clean and noisy images
- Used for supervised denoising training
- Returns (clean, noisy) tuples

#### 5.2 Specialized Dataset Classes
These are for inference.
**`CleanProfiledRGBNoisyBayerImageCropsDataset`**
- **Purpose**: Joint denoising + demosaicing *from Bayer* to profiled RGB
- **Input**: Noisy Bayer RAW images
- **Output**: Clean profiled RGB images (gamma-corrected)
- **Features**:
  - Random crop extraction (configurable crop size)
  - Multiple crops per image
  - Noise profiling and calibration
  - Data augmentation (flips, rotations)
  - Transfer function application (gamma correction)

**`CleanProfiledRGBNoisyProfiledRGBImageCropsDataset`**
- **Purpose**: Denoising in *profiled RGB* space
- **Input**: Noisy profiled RGB images
- **Output**: Clean profiled RGB images
- Both input and output in same color space

These are  test like test reserve; not unit test. Used as part of training.
**Test Dataset Classes:**
- `TestRawDataloader`: Loads full-resolution test images
- `TestDataIngestor`: Manages test dataset workflow

---

### Step 6: Configuration Files

Training is controlled by YAML configuration files in `src/rawnind/config/`. If you skip this step, stuff will break.

#### 6.1 Example: `train_denoise_bayer2prgb.yaml`
```yaml
crop_size: 256                    # Size of randomly cropped images used for training 
val_crop_size: 1024              # " "                                ...for validation
test_crop_size: 1024             # " "                                ...for testing 
num_crops_per_image: 4           #How many crops to extract per image per epoch
batch_size_clean: 1              # Number of clean "ground truth" or "GT" images per batch
batch_size_noisy: 4              # Number of noisy images per batch
transfer_function: gamma22       # Color transfer function - `gamma22`, `srgb` - these do stuff #TODO go become more certain of their papplication 
arch: unet                        # Network architecture. You probably just want `unet` but you could pick `resnet` or something more exotic
```

### Step 7: Training Scripts 


**`train_denoiser_bayer2prgb.py`**
- Trains denoising + demosaicing (Bayer → Profiled RGB)
- Uses `CleanProfiledRGBNoisyBayerImageCropsDataset`
- Default input channels: 4 (Bayer RGGB)

**`train_denoiser_prgb2prgb.py`**
- Trains denoising in RGB space (Profiled RGB → Profiled RGB)
- Uses `CleanProfiledRGBNoisyProfiledRGBImageCropsDataset`
- Default input channels: 3 (RGB)

**`train_dc_bayer2prgb.py`**
- Trains demosaicing only (clean Bayer → Profiled RGB)
- No noise, focuses on color interpolation

**`train_dc_prgb2prgb.py`**
- Trains compression artifact removal (RGB → RGB)

---

### Step 8: The Process By Which Datasets Are Loaded for Training ("TPBWDALfTRAIN")

#### 8.1 Training Class Hierarchy (This is *not* the process, it's a class hierarchy?)
```
AbstractTrainer (base class)
  ↓
ImageToImageNNTraining
  ↓
BayerImageToImageNNTraining
  ↓
DenoiserTraining
  ↓
BayerDenoiser
  ↓
DenoiserTrainingBayerToProfiledRGB (final training class)
```

#### 8.2 Dataset Initialization Flow (The actual process)

When you run a training script, this sequence occurs:

**Step 1: Training Class Instantiation**
```python
trainer = DenoiserTrainingBayerToProfiledRGB(args)
```

**Step 2: Configuration Loading**
- Loads `train_denoise_bayer2prgb.yaml`
- Parses command-line arguments
- Sets hyperparameters

**Step 3: Dataset Creation**
The abstract trainer classes implement methods like:
- `get_training_dataset()` - Creates training dataset
- `get_validation_dataset()` - Creates validation dataset
- `get_test_dataset()` - Creates test dataset

**Step 4: Dataset Instantiation**
For Bayer→RGB denoising:
```python
train_dataset = CleanProfiledRGBNoisyBayerImageCropsDataset(
    dataset_root='datasets/RawNIND',
    sensor_type='Bayer',
    crop_size=256,
    num_crops_per_image=4,
    transfer_function='gamma22',
    split='train'
)
```

**Step 5: DataLoader Creation**
PyTorch DataLoader wraps the dataset:
```python
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)
```

---

### Step 9: Batch Loading Details

#### 9.1 What Happens During Batch Loading
But where does this happen?

**For each batch iteration:**

1. **Scene Selection**: Random scenes chosen from dataset index
2. **Image Loading**: RAW files loaded via `rawpy`
3. **Preprocessing**:
   - Extract Bayer pattern (RGGB)
   - Normalize to [0, 1] range #ghastly, that
   - Apply black level correction
   - Apply white balance
4. **Crop Extraction**:
   - Random 256×256 crops extracted
   - Multiple crops per image (e.g., 4)
5. **Augmentation**:
   - Random horizontal/vertical flips
   - Random 90° rotations
6. **Target Generation**:
   - Load corresponding clean image
   - Apply same crop location
   - Convert to profiled RGB
   - Apply transfer function (gamma correction)
7. **Batch Assembly**:
   - Stack crops into batch tensors
   - Input: `[B, 4, 256, 256]` (Bayer)
   - Target: `[B, 3, 256, 256]` (RGB)

#### 9.2 Batch Tensor Shapes

For `batch_size=4` with Bayer→RGB denoising:
- **Input batch**: `torch.Tensor[4, 4, 256, 256]`
  - 4 samples
  - 4 channels (R, G1, G2, B)
  - 256×256 spatial dimensions
- **Target batch**: `torch.Tensor[4, 3, 256, 256]`
  - 4 samples
  - 3 channels (R, G, B)
  - 256×256 spatial dimensions

---

### Step 10: Running Training

#### 10.1 Start Training
```bash
cd src/rawnind
python train_denoiser_bayer2prgb.py \
    --config config/train_denoise_bayer2prgb.yaml \
    --dataset_root datasets/RawNIND \
    --output_dir results/bayer2prgb \
    --epochs 100 \
    --lr 0.0001 \
    --gpu 0
```

#### 10.2 Training Loop Flow

**For each epoch:**
1. **Training Phase**:
   - Iterate through training DataLoader
   - Load batches (clean/noisy pairs)
   - Forward pass through model
   - Compute loss (MSE, perceptual, etc.)
   - Backward pass and optimizer step
   - Log training metrics

2. **Validation Phase**:
   - Iterate through validation DataLoader
   - Load larger crops (1024×1024)
   - Forward pass (no gradients)
   - Compute validation metrics (PSNR, SSIM)
   - Log validation results

3. **Checkpointing**:
   - Save model weights
   - Save optimizer state
   - Save training progress

---