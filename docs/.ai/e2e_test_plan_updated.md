### Analysis of Integration Tests in test_e2e_image_processing_pipeline*.py

The project includes two key test files for end-to-end (E2E) image processing pipeline validation:

1. **test_e2e_image_processing_pipeline.py** (main file, 300+ lines): Focuses on unit-like integration with synthetic data and heavy mocking. It tests pipeline components (e.g., bayer_to_prgb, process_image_base, denoise_image_compute_metrics, load_image/save_image) using torch.rand-generated images and Mock objects for models/inference. It verifies shapes, ranges, device transfers, metrics (MSE, MS-SSIM), error handling (invalid channels, NaNs), and performance (memory, scalability). However, it does not execute the full pipeline with real data/models/dependencies—mocks bypass actual inference, and synthetic data avoids I/O/external libs.

2. **test_e2e_image_processing_pipeline_real.py** (secondary file, ~200 lines): Attempts real execution but is incomplete/broken. It loads models from weights dir (but skips if no "config.yaml"—actual configs are args.yaml/trainres.yaml), uses synthetic EXR for I/O, and mocks some parts. Tests cover similar areas but with @pytest.mark.slow and real engine where possible. It verifies dependencies (rawpy, OpenEXR, etc.) but fails on model loading due to missing config parsing.

**Current Limitations Preventing Full Pipeline Execution:**
- **Synthetic Data Only**: Tests use torch.rand for Bayer/RGB images; no real RAW/EXR/JPG inputs. load_image/save_image are partially tested with temp EXR but synthetic content.
- **Mocked Models/Inference**: Mock(spec=UtNet2/InferenceEngine) bypasses real Denoiser loading/inference. No actual .pt.opt checkpoint usage.
- **Incomplete Dependencies**: External libs checked but not fully used (e.g., rawpy for RAW loading skipped). Internal: No real model_factory/Denoiser instantiation.
- **No Real Data/Models**: Dataset scripts (dl_ds.sh, organize_files.sh) present but not run (Windows bash issues). Models in src/rawnind/models/weights/ unused. Existing test_data/Moor_frog_bl.jpg (RGB JPG) available but not utilized.
- **Pipeline Gaps**: Full flow (load -> demosaic/color-correct -> infer -> process/gain-match -> metrics -> save) not end-to-end with real I/O/models. Error handling tested but not with real failures (e.g., corrupted RAW).

**Full Pipeline Components (from Imports/Calls):**
- **Input**: load_image(fpath, device) → tensor + rgb_xyz_matrix (uses rawpy/OpenEXR/oiio for RAW/EXR; cv2/np for JPG).
- **Preprocess**: bayer_to_prgb(tensor, matrix) if Bayer (demosaic via rawproc, color via matrix).
- **Inference**: InferenceEngine(model, device).infer(input) → Denoiser (UtNet2-based) forward pass on GPU/CPU.
- **Postprocess**: process_image_base(inference_obj, output, gt, matrix) → gain matching, nonlinearities.
- **Metrics**: compute_metrics(input, gt, ["mse", "msssim_loss"]) → pt_losses (MS-SSIM via torch).
- **Output**: save_image(tensor, fpath, metadata) → EXR/JPG via OpenEXR/cv2.
- **E2E Wrapper**: denoise_image_from_fpath_compute_metrics_and_export(fpath_in, inference_obj, fpath_gt, metrics, out_fpath) orchestrates all.

Dependencies:
- **External**: rawpy (RAW), OpenEXR/oiio (EXR), numpy/cv2 (image ops), torch (tensors/models), pytest/yaml (testing/config), unittest.mock (mocks).
- **Internal**: pytorch_helpers (fpath_to_tensor), image_denoiser (core functions), inference_engine/model_factory (engine/Denoiser), raw_denoiser (UtNet2), pt_losses (metrics).

### Specific Modifications Needed

All modifications focus on **test inputs/fixtures/scripts** to enable full execution without altering core pipeline code (rawnind src). This uses existing test_data JPG, downloads one sample RAW for Bayer, and loads real models via factory. Tests remain in pytest style; add markers (@pytest.mark.real) for optional slow runs.

#### 1. **Enhance Test Inputs: Replace Synthetic Data with Real (No Core Changes)**
   - **RGB Tests**: Use existing src/rawnind/tests/test_data/Moor_frog_bl.jpg (real JPG).
     - Update fixtures: Replace sample_rgb_image with load_image("src/rawnind/tests/test_data/Moor_frog_bl.jpg", device="cpu")[0].unsqueeze(0).
     - For GT: Create simple GT by adding noise to loaded image.
   - **Bayer/RAW Tests**: Dataset not downloaded; dl_ds.sh requires Linux/bash (fails on Windows). Create/run Python script to download one small RAW sample (~10MB) from UCLouvain Dataverse (doi:10.14428/DVN/DEQCIM).
     - **New Script**: Create src/rawnind/tests/download_sample_raw.py (below). Run via pytest fixture or manual.
       ```python
       import requests
       import os
       from pathlib import Path

       def download_sample_raw(output_dir="src/rawnind/tests/test_data/raw_samples"):
           Path(output_dir).mkdir(parents=True, exist_ok=True)
           # Sample: Small Bayer RAW from dataset (replace with actual file ID from dataverse API)
           url = "https://dataverse.uclouvain.be/api/access/datafile/123456"  # Replace with real ID, e.g., first Bayer file
           filename = "Bayer_TEST_7D-2_GT_ISO100.cr2"  # Example
           filepath = Path(output_dir) / filename
           if not filepath.exists():
               response = requests.get(url, stream=True)
               response.raise_for_status()
               with open(filepath, 'wb') as f:
                   for chunk in response.iter_content(chunk_size=8192):
                       f.write(chunk)
               print(f"Downloaded {filename} to {output_dir}")
           return str(filepath)

       if __name__ == "__main__":
           download_sample_raw()
       ```
       - Usage: Run `python src/rawnind/tests/download_sample_raw.py` (manual or via fixture). Get file ID via kagi_search if needed, but use example for now.
     - Update fixtures: sample_bayer_image → load_image(downloaded_raw_path, device)[0].unsqueeze(0) (trims to Bayer tensor).
   - **EXR Tests**: Generate real EXR from JPG via save_image in fixture, or download sample EXR if needed (similar script).
   - **Impact**: Enables real I/O (rawpy for RAW, OpenEXR for save/load). No core changes; just fixture updates.

#### 2. **Integrate Real Model Loading (Replace Mocks)**
   - Current mocks skip actual forward pass. Use model_factory to load from weights dir (e.g., DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1/saved_models/iter_3900000.pt.opt).
     - Fix real.py fixture: Parse args.yaml (not config.yaml). Use yaml.safe_load for config, then Denoiser(config).load_state_dict(torch.load(pt_path, map_location=device)).
     - Update mock_denoiser_model/mock_inference_engine fixtures:
       ```python
       @pytest.fixture
       def real_denoiser_model():
           weights_path = "src/rawnind/models/weights/DenoiserTrainingProfiledRGBToProfiledRGB_3ch_2024-10-09-prgb_ms-ssim_mgout_notrans_valeither_-1"
           args_path = Path(weights_path) / "args.yaml"
           pt_path = list(Path(weights_path) / "saved_models").glob("*.pt.opt")[0]
           with open(args_path) as f:
               config = yaml.safe_load(f)
           model = Denoiser(config)  # From model_factory
           model.load_state_dict(torch.load(pt_path, map_location='cpu'))
           model.eval()
           return model

       @pytest.fixture
       def real_inference_engine(real_denoiser_model):
           device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           return InferenceEngine(real_denoiser_model, device)
       ```
     - For Bayer models, select from Bayer* dirs.
     - Add skip if no weights: pytest.skip("No trained models available").
   - **Impact**: Real forward pass/inference. Tests now use full Denoiser/UtNet2. No core changes; fixture-only.

#### 3. **Enable Full E2E Execution with File I/O on Real Data**
   - Update test_full_file_to_file_pipeline and test_image_loading_and_saving_pipeline:
     - Use real input: in_img_fpath = "src/rawnind/tests/test_data/Moor_frog_bl.jpg" (RGB) or downloaded RAW.
     - gt_img_fpath = None or noise-added version saved as temp GT.
     - out_img_fpath = temp EXR/JPG.
     - Use real_base_inference (from above fixture) instead of mock.
     - Verify: os.path.exists(out), load_image(out) shape/range, metrics >0.
   - For Bayer: Add test with RAW → demosaic → infer (RGB model) → save.
   - Add parametrize: @pytest.mark.parametrize("input_type", ["rgb", "bayer"]) to run both.
   - Run full wrapper: denoise_image_from_fpath_compute_metrics_and_export with real paths/engine.
   - **Slow Marker**: Add @pytest.mark.slow to real tests; run via pytest -m slow.
   - **Impact**: Executes load → preprocess → infer → postprocess → metrics → save with real deps (rawpy/cv2/OpenEXR/torch). Temp dirs for I/O.

#### 4. **Dataset Download Script Execution**
   - dl_ds.sh: Downloads all to datasets/RawNIND/src/flat via curl/jq/wget (Linux-only).
   - organize_files.sh: Moves to datasets/RawNIND/src/{Bayer|X-Trans}/{scene}/[gt/].
   - Windows Issue: Bash not native. Propose run in Git Bash/WSL or convert to Python (as above for sample).
   - For full: Create download_full_dataset.py adapting dl_ds.sh (use requests/json to fetch file list, download).
     - But for tests, sample suffices (one RAW + existing JPG).
   - Execute: Manual `python download_sample_raw.py`; integrate as session fixture: def download_data(): ...; autouse=True.

#### 5. **Verification: No Functional Code Changes Required**
   - All proposals modify **tests only**: Fixtures for real paths/models, new download script in tests/, parametrize for types.
   - Core pipeline (image_denoiser.py, inference_engine.py, etc.) unchanged—tests now drive real usage.
   - If download impossible (e.g., API key), use synthetic + real JPG; but sample download possible via public dataverse.
   - Edge: If rawpy fails on Windows RAW, skip Bayer tests or use pre-converted EXR.

#### 6. **Implementation Steps**
   - Add download_sample_raw.py to src/rawnind/tests/.
   - Update both test files with real fixtures (merge real.py into main if duplicate).
   - Add to conftest.py: session fixture for data download/model loading.
   - Run: pytest src/rawnind/tests/test_e2e_image_processing_pipeline.py -m real -v (expect full pipeline with metrics/output files).

These changes ensure tests execute the entire pipeline (load→pre→infer→post→metrics→save) with all deps (external/internal) on real data, using only test enhancements.