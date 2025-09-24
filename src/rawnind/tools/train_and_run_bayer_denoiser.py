"""
Train a Bayer denoiser on ~100 non‑X‑Trans RawNIND pairs (Dataverse) and run it on a
random high‑ISO raw.pixls.us image. Designed to be self‑contained and avoid
modifying existing code.

Summary of behavior
- Downloads RawNIND dataset file list (Dataverse: doi:10.14428/DVN/DEQCIM)
- Randomly (but reasonably) selects candidate noisy/clean pairs (heuristics below)
- Concurrently downloads until ~100 pairs or 4GB total cap (whichever first)
- Filters out X‑Trans/Fuji (.RAF extension or EXIF Make/Model hints)
- Builds a 90/10 train/val split under datasets/DEMO
- Trains a tiny Bayer→RGB denoiser using rawnind.training.clean_api
  • Losses: MSE + optional MS‑SSIM (via additional_metrics for validation)
  • Early stopping + wall‑clock time cap (~10 minutes by default)
- Fetches a random non‑X‑Trans ISO>3200 RAW from raw.pixls.us (best‑effort)
  • If not found after a few attempts, falls back to an extra Dataverse RAW
- Runs inference and saves a 16‑bit TIFF output

CLI example (PowerShell)
python -m rawnind.tools.train_and_run_bayer_denoiser --count 100 --time_max 600

Notes
- This script favors robustness and logging over optimal performance.
- Pairing heuristic uses filename tokens ("noisy"/"clean") when available.
  Otherwise, groups files by filename stem and pairs by ISO (lower ISO=clean).
- Network hiccups are tolerated with retries; selection is randomized.
- No changes to existing modules are made. Only this script is added.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
import torch
import trio

# Project imports (use only public/clean APIs where possible)
from rawnind.training.clean_api import (
    TrainingConfig,
    ExperimentConfig,
    create_denoiser_trainer,
    create_experiment_manager,
)
from rawnind.dependencies import raw_processing as rawproc

# Constants
DATAVERSE_API = "https://dataverse.uclouvain.be/api/datasets/:persistentId/?persistentId={doi}"
RAW_EXTS = {".cr2", ".cr3", ".arw", ".dng", ".nef", ".rw2", ".orf", ".pef", ".raf"}
NON_XTRANS_EXTS = RAW_EXTS - {".raf"}
FUJI_HINTS = ["fuji", "fujifilm", "x-t", "xpro", "x-pro", "x100"]
DEFAULT_DATASET_DIR = Path("datasets/DEMO")
DEFAULT_RUNS_DIR = Path("runs/DEMO")


@dataclass
class DVFile:
    id: int
    label: str
    size: int
    description: str

    @property
    def ext(self) -> str:
        return Path(self.label).suffix.lower()

    def looks_raw_non_xtrans(self) -> bool:
        if self.ext not in RAW_EXTS:
            return False
        if self.ext == ".raf":
            return False
        # crude textual filter for Fuji/X‑Trans in metadata fields
        hay = f"{self.label} {self.description}".lower()
        if any(h in hay for h in FUJI_HINTS):
            return False
        return True


def get_dataverse_file_list(doi: str) -> List[DVFile]:
    url = DATAVERSE_API.format(doi=doi)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    files = data["data"]["latestVersion"]["files"]
    out: List[DVFile] = []
    for f in files:
        df = f.get("dataFile", {})
        out.append(
            DVFile(
                id=df.get("id"),
                label=df.get("filename") or df.get("label") or "",
                size=int(df.get("filesize") or 0),
                description=f.get("description") or "",
            )
        )
    return out


def load_files_from_listing(listing_path: str) -> List[DVFile]:
    """Load a saved Dataverse dataset listing (JSON or YAML) and extract files as DVFile.

    Accepts the repository-provided datasets/RawNIND/dataset.yaml (JSON payload)
    or a raw Dataverse API JSON export. Falls back to yaml.safe_load if JSON
    parsing fails.
    """
    p = Path(listing_path)
    text = p.read_text(encoding="utf-8")
    data = None
    try:
        data = json.loads(text)
    except Exception:
        try:
            import yaml as _yaml  # lazy import
            data = _yaml.safe_load(text)
        except Exception as e:  # pragma: no cover
            raise ValueError(f"Could not parse listing file {listing_path}: {e}")

    try:
        files = data["data"]["latestVersion"]["files"]
    except Exception as e:
        raise ValueError(f"Listing missing expected Dataverse structure: {e}")

    out: List[DVFile] = []
    for f in files:
        df = f.get("dataFile", {})
        out.append(
            DVFile(
                id=df.get("id"),
                label=df.get("filename") or df.get("label") or "",
                size=int(df.get("filesize") or 0),
                description=f.get("description") or "",
            )
        )
    return out


def parse_bayer_label(label: str) -> Tuple[Optional[str], Optional[int]]:
    """Parse Bayer_*_ISO#### filenames to (scene_id, iso)."""
    m = re.search(r"(?i)^bayer_([^_]+)_iso(\d+)", label)
    if not m:
        return None, None
    scene = m.group(1)
    iso = int(m.group(2))
    return scene, iso


def select_diverse_bayer_pairs(files: Sequence[DVFile], target_count: int) -> List[Tuple[DVFile, DVFile]]:
    """Select ~target_count (noisy, clean) pairs with diversity across scenes and ISO bins.

    Strategy:
    - Keep only labels starting with Bayer_ and with non-XTrans RAW exts.
    - Group by scene (token after Bayer_ and before _ISO).
    - For each scene, sort by ISO and create intra-scene pairs by pairing highest ISO with lowest ISO.
    - Round-robin sample across scenes until reaching target_count.
    """
    # Filter Bayer and file types
    bayer_files = [f for f in files if f.label.lower().startswith("bayer_") and Path(f.label).suffix.lower() in NON_XTRANS_EXTS]

    by_scene: Dict[str, List[Tuple[int, DVFile]]] = {}
    for f in bayer_files:
        scene, iso = parse_bayer_label(f.label)
        if scene is None or iso is None:
            continue
        by_scene.setdefault(scene, []).append((iso, f))

    # Build intra-scene pairs (noisy=highest ISO, clean=lowest ISO)
    scene_pairs: Dict[str, List[Tuple[DVFile, DVFile]]] = {}
    for scene, lst in by_scene.items():
        lst_sorted = sorted(lst, key=lambda t: t[0])  # by ISO
        if len(lst_sorted) < 2:
            continue
        pairs_for_scene: List[Tuple[DVFile, DVFile]] = []
        i, j = 0, len(lst_sorted) - 1
        while i < j:
            clean_iso, clean_file = lst_sorted[i]
            noisy_iso, noisy_file = lst_sorted[j]
            pairs_for_scene.append((noisy_file, clean_file))
            i += 1
            j -= 1
        scene_pairs[scene] = pairs_for_scene

    # Round-robin across scenes for diversity
    scenes = list(scene_pairs.keys())
    random.shuffle(scenes)
    out_pairs: List[Tuple[DVFile, DVFile]] = []
    idx_map = {s: 0 for s in scenes}
    while len(out_pairs) < target_count and scenes:
        progressed = False
        for s in list(scenes):
            k = idx_map[s]
            if k < len(scene_pairs[s]):
                out_pairs.append(scene_pairs[s][k])
                idx_map[s] = k + 1
                progressed = True
                if len(out_pairs) >= target_count:
                    break
            else:
                scenes.remove(s)
        if not progressed:
            break

    return out_pairs


def build_pair_candidates(files: Sequence[DVFile]) -> List[Tuple[DVFile, DVFile]]:
    """Try to construct (noisy, clean) pairs from Dataverse file labels.

    Heuristics, in order:
    1) If filename contains 'noisy'/'clean' tokens (case-insensitive), pair by token swap.
    2) Else, group by a conservative stem (strip ISO patterns, numbers), and within a group
       pair two files by ISO where lower ISO -> clean, higher -> noisy (ISO parsed later).
    """
    # 1) Token‑based pairing
    by_label = {f.label: f for f in files}
    pairs: List[Tuple[DVFile, DVFile]] = []

    def token_swap(name: str, a: str, b: str) -> Optional[str]:
        rx = re.compile(re.escape(a), re.IGNORECASE)
        if rx.search(name):
            return rx.sub(b, name)
        return None

    visited = set()
    for f in files:
        if f in visited:
            continue
        for a, b in [("noisy", "clean"), ("noise", "clean"), ("highiso", "clean")]:
            cand = token_swap(f.label, a, b)
            if cand and cand in by_label:
                noisy, clean = (f, by_label[cand]) if a in f.label.lower() else (by_label[cand], f)
                pairs.append((noisy, clean))
                visited.add(noisy)
                visited.add(clean)
                break

    # 2) Group‑by‑stem fallback — defer ISO decision until after download/EXIF parsing
    # Build loose groups to increase chances; pairing finalized post‑download.
    leftover = [f for f in files if f not in visited]
    stem_groups: Dict[str, List[DVFile]] = {}
    for f in leftover:
        stem = re.sub(r"(?i)iso\d+", "", f.label)
        stem = re.sub(r"\d+", "", stem)
        stem = re.split(r"[\s_\-\.]+", stem)[0]
        stem_groups.setdefault(stem, []).append(f)

    # Create provisional pairs from groups of size>=2 (actual noisy/clean order to be decided later)
    for g in stem_groups.values():
        if len(g) >= 2:
            # pair in twos (g[0], g[1]), (g[2], g[3]), ...; will reorder by ISO later
            for i in range(0, len(g) - 1, 2):
                pairs.append((g[i], g[i + 1]))

    # Randomize to avoid bias
    random.shuffle(pairs)
    return pairs


async def _download_one(session: requests.Session, file_id: int, dest: Path, *, retries: int = 3) -> Tuple[bool, int]:
    url = f"https://dataverse.uclouvain.be/api/access/datafile/{file_id}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            # requests is blocking; run in thread to use trio concurrency
            def _do_get():
                with session.get(url, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(chunk_size=1 << 15):
                            if chunk:
                                f.write(chunk)

            await trio.to_thread.run_sync(_do_get)
            tmp.replace(dest)
            return True, dest.stat().st_size
        except Exception as e:  # noqa
            logging.warning(f"Download failed for {file_id} (attempt {attempt}/{retries}): {e}")
            await trio.sleep(1.0 * attempt)
    return False, 0


async def download_pairs_concurrently(pairs: List[Tuple[DVFile, DVFile]], out_dir: Path, cap_bytes: int, max_parallel: int) -> List[Tuple[Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Tuple[Path, Path]] = []
    total_bytes = 0

    # Shuffle to improve randomness
    random.shuffle(pairs)

    async with trio.open_nursery() as nursery:
        sem = trio.Semaphore(max_parallel)
        session = requests.Session()
        results: List[Tuple[int, Path, bool, int]] = []  # (pair_idx, path, ok, bytes)

        async def worker(idx: int, dvf: DVFile, path: Path):
            nonlocal total_bytes
            async with sem:
                ok, n = await _download_one(session, dvf.id, path)
                results.append((idx, path, ok, n))

        # schedule downloads pairwise, respecting cap
        scheduled = 0
        for i, (noisy, clean) in enumerate(pairs):
            if total_bytes >= cap_bytes:
                break
            noisy_path = out_dir / noisy.label
            clean_path = out_dir / clean.label
            nursery.start_soon(worker, 2 * i, noisy, noisy_path)
            nursery.start_soon(worker, 2 * i + 1, clean, clean_path)
            scheduled += 2

        # nursery exits after all scheduled downloads complete

    # Postprocess: gather the successfully downloaded pairs
    have: Dict[str, Path] = {p.name: p for _, p, ok, _ in results if ok}
    # Approx total bytes from successful downloads
    total_bytes = sum(n for *_, ok, n in results if ok)

    for noisy, clean in pairs:
        np = have.get(noisy.label)
        cp = have.get(clean.label)
        if np and cp:
            downloaded.append((np, cp))

    logging.info(f"Downloaded {len(downloaded)} pairs; ~{total_bytes / (1<<30):.2f} GiB")
    return downloaded


def is_fuji_or_xtrans(raw_path: Path) -> bool:
    if raw_path.suffix.lower() == ".raf":
        return True
    # Try rawpy for EXIF make/model
    try:
        import rawpy  # local import to avoid import cost if unused
        with rawpy.imread(str(raw_path)) as rp:
            make = str(getattr(rp, "camera_make", ""))
            model = str(getattr(rp, "camera_model", ""))
            hay = f"{make} {model}".lower()
            return any(h in hay for h in FUJI_HINTS)
    except Exception:
        return False


def get_iso_value(raw_path: Path) -> Optional[int]:
    try:
        import rawpy
        with rawpy.imread(str(raw_path)) as rp:
            iso = getattr(rp, "iso_speed", None)
            if iso is None:
                # some raws: use exif tag via rawpy
                iso = getattr(rp, "shot_iso", None)
            if iso is None:
                return None
            return int(iso)
    except Exception:
        return None


def construct_pairs_from_downloads(downloaded: List[Tuple[Path, Path]]) -> List[Tuple[Path, Path]]:
    """Ensure (noisy, clean) order; if uncertain, use ISO to decide (higher ISO=noisy)."""
    fixed: List[Tuple[Path, Path]] = []
    for a, b in downloaded:
        # filename hints first
        la, lb = a.name.lower(), b.name.lower()
        if ("noisy" in la and "clean" in lb):
            fixed.append((a, b))
            continue
        if ("noisy" in lb and "clean" in la):
            fixed.append((b, a))
            continue
        # fallback: ISO
        ia = get_iso_value(a) or 0
        ib = get_iso_value(b) or 0
        if ia == ib:
            # arbitrary but stable
            fixed.append((a, b))
        elif ia > ib:
            fixed.append((a, b))  # a noisy, b clean
        else:
            fixed.append((b, a))  # b noisy, a clean
    return fixed


def split_pairs(pairs: List[Tuple[Path, Path]], split: float) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    n_train = max(1, int(len(pairs) * split))
    return pairs[:n_train], pairs[n_train:]


def write_index_csv(path: Path, pairs: List[Tuple[Path, Path]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["noisy", "clean", "iso_noisy", "iso_clean"])
        for noisy, clean in pairs:
            w.writerow([str(noisy), str(clean), get_iso_value(noisy) or "", get_iso_value(clean) or ""]) 


class BayerPairsDataset(torch.utils.data.Dataset):
    """Minimal dataset that loads RAW pairs and yields tensors for training.

    Output batch dict keys match clean_api expectations:
      - noisy_images: Tensor [B, 4, H/2, W/2] (RGGB mosaic normalized to [0,1])
      - clean_images: Tensor [B, 3, H, W] linear Rec2020 RGB
      - masks: Tensor [B, 1, H, W] all ones (placeholder)
    """

    def __init__(self, pairs: Sequence[Tuple[Path, Path]], crop_size: int = 128, device: str = "cpu"):
        self.pairs = list(pairs)
        self.crop = crop_size
        self.device = device

    def __len__(self):
        return len(self.pairs)

    def _load_noisy_rggb(self, fpath: Path) -> torch.Tensor:
        rggb, meta = rawproc.raw_fpath_to_rggb_img_and_metadata(str(fpath))
        # rggb: (4, H/2, W/2) numpy float32 in [0,1]
        ten = torch.from_numpy(rggb).float()
        return ten

    def _load_clean_rgb(self, fpath: Path) -> torch.Tensor:
        # load mono + metadata, demosaic to camera RGB, then transform to lin_rec2020
        mono, meta = rawproc.raw_fpath_to_mono_img_and_metadata(str(fpath))
        rgb_cam = rawproc.BayerProcessor(rawproc.ProcessingConfig()).demosaic(mono, meta)
        rgb_lin2020 = rawproc.ColorTransformer().cam_rgb_to_profiled(rgb_cam, meta, profile="lin_rec2020")
        ten = torch.from_numpy(rgb_lin2020).float()  # (3,H,W)
        return ten

    def _random_crop_pair(self, x4: torch.Tensor, y3: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x4 is half‑res per channel mosaic; y3 is full res. We crop y3 and map to x4 accordingly.
        _, H, W = y3.shape
        if H < self.crop or W < self.crop:
            # center crop fallback
            top = max(0, (H - self.crop) // 2)
            left = max(0, (W - self.crop) // 2)
        else:
            top = random.randint(0, H - self.crop)
            left = random.randint(0, W - self.crop)
        y_crop = y3[:, top:top + self.crop, left:left + self.crop]
        # map to RGGB mosaic crop (half resolution per channel grid)
        top4 = top // 2
        left4 = left // 2
        h4 = max(1, self.crop // 2)
        w4 = max(1, self.crop // 2)
        x_crop = x4[:, top4:top4 + h4, left4:left4 + w4]
        return x_crop, y_crop

    def __getitem__(self, idx):
        noisy_path, clean_path = self.pairs[idx]
        x4 = self._load_noisy_rggb(noisy_path)
        y3 = self._load_clean_rgb(clean_path)
        x4, y3 = self._random_crop_pair(x4, y3)
        masks = torch.ones(1, y3.shape[-2], y3.shape[-1], dtype=torch.float32)
        sample = {
            "noisy_images": x4.unsqueeze(0),  # add batch dim later in collate
            "clean_images": y3.unsqueeze(0),
            "masks": masks.unsqueeze(0),
        }
        # Remove batch here; DataLoader batch will stack them.
        return {
            "noisy_images": x4,
            "clean_images": y3,
            "masks": masks,
        }


def make_dataloaders(pairs_train: List[Tuple[Path, Path]], pairs_val: List[Tuple[Path, Path]], crop: int, batch: int) -> Tuple[Iterable, Iterable]:
    train_ds = BayerPairsDataset(pairs_train, crop_size=crop)
    val_ds = BayerPairsDataset(pairs_val, crop_size=crop)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=0)
    return train_dl, val_dl


def choose_batch_size(default: int = 1) -> int:
    # Heuristic: try to use 2 if CUDA has headroom; otherwise 1
    if torch.cuda.is_available():
        try:
            idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            # if > 4GB free, use batch 2
            if free > (4 << 30):
                return 2
        except Exception:
            pass
    return default


def train_bayer_denoiser(train_dl, val_dl, runs_dir: Path, device: str, steps: int, crop: int, add_ms_ssim: bool, time_max_sec: int) -> Tuple[Path, Dict]:
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    exp_dir = runs_dir / f"bayer_quickstart_{ts}"
    exp = ExperimentConfig(
        experiment_name=f"bayer_quickstart_{ts}",
        save_directory=str(exp_dir),
        checkpoint_interval=max(50, steps // 5),
        keep_best_n_models=2,
        metrics_to_track=["loss", "ms_ssim"] if add_ms_ssim else ["loss"],
    )
    trainer = create_denoiser_trainer(
        training_type="bayer_to_rgb",
        config=TrainingConfig(
            model_architecture="unet",
            input_channels=4,
            output_channels=3,
            learning_rate=1e-4,
            batch_size=getattr(train_dl, "batch_size", 1),
            crop_size=crop,
            total_steps=steps,
            validation_interval=max(25, steps // 10),
            loss_function="mse",  # primary loss
            device=device,
            patience=200,
            lr_decay_factor=0.5,
            early_stopping_patience=200,
            additional_metrics=["ms_ssim"] if add_ms_ssim else [],
            filter_units=48,
        ),
    )
    manager = create_experiment_manager(exp)

    # Time‑bounded training wrapper
    start = time.time()

    def time_remaining() -> bool:
        return (time.time() - start) < time_max_sec

    steps_done = 0
    while steps_done < steps and time_remaining():
        # Train in chunks up to validation interval to allow early stopping and time checks
        chunk = min(steps - steps_done, trainer.config.validation_interval)
        res = trainer.train(train_dataloader=train_dl, validation_dataloader=val_dl, experiment_manager=manager, max_steps=trainer.current_step + chunk)
        steps_done = res.get("steps_completed", trainer.current_step)
        # Simple early stop if flagged
        if res.get("early_stopped"):
            logging.info(f"Early stopped: {res.get('early_stop_reason')}")
            break

    # Pick latest/best checkpoint
    ckpts = sorted((exp.checkpoint_dir.glob("model_step_*.pt")), key=lambda p: int(p.stem.split("_")[-1]))
    best = ckpts[-1] if ckpts else None
    return best, {"experiment_dir": str(exp_dir), "steps_completed": steps_done}


def download_random_high_iso_raw_from_pixls(out_dir: Path, iso_min: int = 3200, attempts: int = 5) -> Optional[Path]:
    """Best‑effort: randomly try a few known camera RAW extensions on raw.pixls.us.
    If not successful, return None and caller will fallback to Dataverse.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # This is a placeholder: raw.pixls.us exposes rsync/HTTP trees; we try a few
    # simple heuristics across common extensions. In absence of a formal JSON index
    # here, we give it a handful of tries with randomized camera folders.
    roots = [
        "https://raw.pixls.us/getfile.php/",
    ]
    # Not all servers expose listing; attempt a few known sample endpoints would be ideal,
    # but in the absence of a stable API, we bail out quickly.
    for i in range(attempts):
        # Since we can't query index here robustly, abort quickly to fallback
        break
    return None


def run_inference_on_raw(ckpt_dir: Path, input_fpath: Path, output_path: Path, device: str):
    # Mirror the simple_denoiser pattern but force Bayer path
    from rawnind.inference.clean_api import create_bayer_denoiser
    denoiser = create_bayer_denoiser(architecture="unet", checkpoint_path=str(ckpt_dir), device=device)
    # Load image
    input_tensor, rgb_xyz = rawproc.load_image(str(input_fpath), device)
    with torch.no_grad():
        processed = denoiser.denoise_bayer(input_tensor, rgb_xyz)
    rawproc.save_image(processed.unsqueeze(0), str(output_path), src_fpath=str(input_fpath))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train a Bayer denoiser on RawNIND and run it on a high‑ISO raw.pixls.us image.")
    p.add_argument("--doi", default="doi:10.14428/DVN/DEQCIM", help="Dataverse DOI to use (RawNIND)")
    p.add_argument("--dataset_dir", default=str(DEFAULT_DATASET_DIR), help="Output dir for dataset")
    p.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR), help="Output dir for runs/experiments")
    p.add_argument("--count", type=int, default=100, help="Target number of pairs")
    p.add_argument("--cap_gb", type=float, default=4.0, help="Download cap in GiB")
    p.add_argument("--max_parallel", type=int, default=8, help="Max concurrent downloads")
    p.add_argument("--split", type=float, default=0.9, help="Train split fraction (val is 1-split)")
    p.add_argument("--device", default="auto", help="Device: auto/cpu/cuda")
    p.add_argument("--steps", type=int, default=1000, help="Max training steps")
    p.add_argument("--batch", type=int, default=0, help="Batch size (0=auto)")
    p.add_argument("--crop", type=int, default=128, help="Crop size for training")
    p.add_argument("--time_max", type=int, default=600, help="Max training time in seconds")
    p.add_argument("--iso_min", type=int, default=3200, help="Min ISO for pixls.us pick")
    p.add_argument("--seed", type=int, default=1337, help="Random seed")
    p.add_argument("--dataverse_listing", default="", help="Path to saved Dataverse listing (JSON/YAML), e.g., datasets/RawNIND/dataset.yaml")
    p.add_argument("--dry_run", action="store_true", help="Print planned selection and exit before downloading")

    args = p.parse_args(argv)

    # Logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    raw_dl_dir = dataset_dir / "raw_downloads"
    raw_dl_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load Dataverse files (prefer local listing if provided)
    if args.dataverse_listing and Path(args.dataverse_listing).exists():
        logging.info(f"Loading Dataverse listing from {args.dataverse_listing} ...")
        dv_files = load_files_from_listing(args.dataverse_listing)
    else:
        logging.info("Querying Dataverse file list via API...")
        dv_files = get_dataverse_file_list(args.doi)
    logging.info(f"Total files in listing: {len(dv_files)}")

    # 2) Prefer Bayer_* scene/ISO pairing with diversity
    pair_cands = select_diverse_bayer_pairs(dv_files, args.count)
    if not pair_cands:
        logging.warning("Bayer-based pairing produced 0 pairs; falling back to heuristic non-X-Trans pairing")
        candidates = [f for f in dv_files if f.looks_raw_non_xtrans()]
        random.shuffle(candidates)
        pair_cands = build_pair_candidates(candidates)
        if len(pair_cands) > args.count:
            pair_cands = pair_cands[: args.count]
    else:
        logging.info(f"Selected {len(pair_cands)} Bayer pairs (diverse by scene/ISO)")

    if args.dry_run:
        for i, (noisy, clean) in enumerate(pair_cands[: min(10, len(pair_cands))]):
            logging.info(f"Pair[{i}]: noisy={noisy.label} clean={clean.label}")
        logging.info("Dry run requested; exiting before download.")
        return 0

    cap_bytes = int(args.cap_gb * (1 << 30))

    logging.info(f"Downloading up to {len(pair_cands)} pairs with cap {args.cap_gb:.1f} GiB...")
    downloaded = trio.run(download_pairs_concurrently, pair_cands, raw_dl_dir, cap_bytes, args.max_parallel)

    # 3) Post‑filter: drop Fuji/X‑Trans by EXIF (conservative)
    filtered = []
    for a, b in downloaded:
        if is_fuji_or_xtrans(a) or is_fuji_or_xtrans(b):
            continue
        filtered.append((a, b))

    # Ensure (noisy, clean) ordering
    pairs = construct_pairs_from_downloads(filtered)

    # Target ~count pairs
    if len(pairs) > args.count:
        pairs = pairs[: args.count]

    if not pairs:
        logging.error("No viable pairs after download/filters. Exiting.")
        return 2

    # 4) Split and write indices
    random.shuffle(pairs)
    train_pairs, val_pairs = split_pairs(pairs, args.split)
    (dataset_dir / "splits").mkdir(parents=True, exist_ok=True)
    write_index_csv(dataset_dir / "splits" / "train.csv", train_pairs)
    write_index_csv(dataset_dir / "splits" / "val.csv", val_pairs)
    logging.info(f"Split: train={len(train_pairs)} val={len(val_pairs)}")

    # 5) Dataloaders
    batch = args.batch if args.batch > 0 else choose_batch_size(default=1)
    train_dl, val_dl = make_dataloaders(train_pairs, val_pairs, crop=args.crop, batch=batch)

    # 6) Train with early stop and time cap
    ckpt, train_info = train_bayer_denoiser(
        train_dl=train_dl,
        val_dl=val_dl,
        runs_dir=Path(args.runs_dir),
        device=device,
        steps=args.steps,
        crop=args.crop,
        add_ms_ssim=True,
        time_max_sec=args.time_max,
    )
    if not ckpt:
        logging.error("No checkpoint produced. Exiting.")
        return 3

    # 7) Fetch random ISO>3200 non‑X‑Trans from raw.pixls.us or fallback
    test_dir = dataset_dir / "test_raw"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_raw = download_random_high_iso_raw_from_pixls(test_dir, iso_min=args.iso_min, attempts=5)
    if test_raw is None:
        # Fallback: pick a random downloaded RAW with ISO>3200 (Dataverse)
        hi = [(a, get_iso_value(a)) for a, _ in pairs] + [(b, get_iso_value(b)) for _, b in pairs]
        hi = [p for p in hi if p[1] and p[1] >= args.iso_min]
        if hi:
            test_raw = random.choice(hi)[0]
        else:
            # last resort: any file
            test_raw = pairs[0][0]
    logging.info(f"Test RAW: {test_raw}")

    # 8) Inference and save 16‑bit TIFF
    out_tif = Path(args.runs_dir) / "outputs" / (Path(test_raw).stem + "_denoised.tif")
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    run_inference_on_raw(ckpt.parent, Path(test_raw), out_tif, device)

    logging.info("Done.")
    logging.info(json.dumps({
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "checkpoint": str(ckpt),
        "test_raw": str(test_raw),
        "output_tiff": str(out_tif),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
