"""
Prepare image dataset: generate alignment and loss masks. Output a yaml list of files,alignment,mask_fpath for Bayer->RGB and RGB->RGB

Compute overexposure in Bayer (if available)

Compute alignment and loss in RGB

Problem:
cannot shift 1px in bayer
Solution:
Calculate shift in RGB image; crop a line/column as needed in Bayer->RGB, no worries for RGB->RGB

Loss mask is based on shifted image;
data loader is straightforward with RGB-RGB (pre-shift images, get loss mask)
with Bayer-to-RGB, loss_mask should be adapted ... TODO (by data loader)

metadata needed: f_bayer_fpath, f_linrec2020_fpath, gt_linrec2020_fpath, overexposure_lb, rgb_xyz_matrix
compute shift between every full-size image
compute loss mask between every full-size image
add list of crops (dict of coordinates : path)
"""

import argparse
import logging
import os
import sys
import threading
import time

import yaml

sys.path.append("..")
from rawnind.libs import rawproc
from common.libs import utilities

from rawnind.libs.rawproc import (
    DATASETS_ROOT,
    DS_DN,
    BAYER_DS_DPATH,
    LINREC2020_DS_DPATH,
    RAWNIND_CONTENT_FPATH,
    LOSS_THRESHOLD,
)

NUM_THREADS: int = os.cpu_count() // 4 * 3  #
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")
HDR_EXT = "tif"

"""
#align images needs: bayer_gt_fpath, profiledrgb_gt_fpath, profiledrgb_noisy_fpath
align_images needs: image_set, gt_file_endpath, f_endpath
outputs gt_rgb_fpath, f_bayer_fpath, f_rgb_fpath, best_alignment, mask_fpath
"""


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num_threads", type=int, help="Number of threads.", default=NUM_THREADS
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files"
    )
    parser.add_argument(
        "--dataset",
        default=DS_DN,
        help="Process external dataset (ext_raw_denoise_train, ext_raw_denoise_test, RawNIND, RawNIND_Bostitch)",
    )
    return parser.parse_args()


def find_cached_result(ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results):
    gt_fpath = os.path.join(ds_dpath, image_set, gt_file_endpath)
    f_fpath = os.path.join(ds_dpath, image_set, f_endpath)
    for result in cached_results:
        if result["gt_fpath"] == gt_fpath and result["f_fpath"] == f_fpath:
            return result


def fetch_crops_list(image_set, gt_fpath, f_fpath, is_bayer, ds_base_dpath):
    def get_coordinates(fn: str) -> list[int, int]:
        return [int(c) for c in fn.split(".")[-2].split("_")]

    crops = []
    gt_basename = os.path.basename(gt_fpath)
    f_basename = os.path.basename(f_fpath)
    prgb_image_set_dpath = os.path.join(
        ds_base_dpath, "crops", "proc", "lin_rec2020", image_set
    )
    if is_bayer:
        bayer_image_set_dpath = os.path.join(
            ds_base_dpath, "crops", "src", "Bayer", image_set
        )
    for f_is_gt in (True, False):
        for fn_f in os.listdir(
            os.path.join(prgb_image_set_dpath, "gt" if f_is_gt else "")
        ):
            if fn_f.startswith(f_basename):
                coordinates = get_coordinates(fn_f)
                for fn_gt in os.listdir(os.path.join(prgb_image_set_dpath, "gt")):
                    if fn_gt.startswith(gt_basename):
                        coordinates_gt = get_coordinates(fn_gt)
                        if coordinates == coordinates_gt:
                            crop = {
                                "coordinates": coordinates,
                                "f_linrec2020_fpath": os.path.join(
                                    prgb_image_set_dpath, "gt" if f_is_gt else "", fn_f
                                ),
                                "gt_linrec2020_fpath": os.path.join(
                                    prgb_image_set_dpath, "gt", fn_gt
                                ),
                            }
                            if is_bayer:
                                crop["f_bayer_fpath"] = os.path.join(
                                    bayer_image_set_dpath,
                                    "gt" if f_is_gt else "",
                                    fn_f.replace("." + HDR_EXT, ".npy"),
                                )
                                crop["gt_bayer_fpath"] = os.path.join(
                                    bayer_image_set_dpath,
                                    "gt",
                                    fn_gt.replace("." + HDR_EXT, ".npy"),
                                )
                                if not os.path.exists(
                                    crop["f_bayer_fpath"]
                                ) or not os.path.exists(crop["gt_bayer_fpath"]):
                                    logging.error(
                                        f"Missing crop: {crop['f_bayer_fpath']} and/or {crop['gt_bayer_fpath']}"
                                    )
                                    breakpoint()
                                assert os.path.exists(crop["f_bayer_fpath"])
                                assert os.path.exists(crop["gt_bayer_fpath"])
                            crops.append(crop)
    return crops


if __name__ == "__main__":
    logging.basicConfig(
        filename=LOG_FPATH,
        format="%(message)s",
        level=logging.INFO,
        filemode="w",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    args = get_args()
    logging.info(f"# python {' '.join(sys.argv)}")
    logging.info(f"# {args=}")
    if args.dataset == DS_DN:
        content_fpath = RAWNIND_CONTENT_FPATH
        bayer_ds_dpath = BAYER_DS_DPATH
        linrec_ds_dpath = LINREC2020_DS_DPATH
    else:
        content_fpath = os.path.join(
            DATASETS_ROOT, args.dataset, f"{args.dataset}_masks_and_alignments.yaml"
        )
        bayer_ds_dpath = os.path.join(DATASETS_ROOT, args.dataset, "src", "Bayer")
        linrec_ds_dpath = os.path.join(
            DATASETS_ROOT, args.dataset, "proc", "lin_rec2020"
        )

    args_in = []
    if args.overwrite or not os.path.exists(content_fpath):
        cached_results = []
    else:
        cached_results = utilities.load_yaml(content_fpath, error_on_404=True)
    for ds_dpath in (bayer_ds_dpath, linrec_ds_dpath):
        if not os.path.isdir(ds_dpath):
            continue
        for image_set in os.listdir(ds_dpath):
            if ds_dpath == linrec_ds_dpath and image_set in os.listdir(bayer_ds_dpath):
                continue  # avoid duplicate, use bayer if available
            in_image_set_dpath: str = os.path.join(ds_dpath, image_set)
            gt_files_endpaths: list[str] = [
                os.path.join("gt", fn)
                for fn in os.listdir(os.path.join(in_image_set_dpath, "gt"))
            ]
            noisy_files_endpaths: list[str] = os.listdir(in_image_set_dpath)
            noisy_files_endpaths.remove("gt")

            for gt_file_endpath in gt_files_endpaths:
                if gt_file_endpath.endswith(".xmp") or gt_file_endpath.endswith(
                    "darktable_exported"
                ):
                    continue
                for f_endpath in gt_files_endpaths + noisy_files_endpaths:
                    if f_endpath.endswith(".xmp") or f_endpath.endswith(
                        "darktable_exported"
                    ):
                        continue
                    if find_cached_result(
                        ds_dpath, image_set, gt_file_endpath, f_endpath, cached_results
                    ):
                        continue
                    args_in.append(
                        {
                            "ds_dpath": ds_dpath,
                            "image_set": image_set,
                            "gt_file_endpath": gt_file_endpath,
                            "f_endpath": f_endpath,
                            "masks_dpath": os.path.join(
                                DATASETS_ROOT, args.dataset, f"masks_{LOSS_THRESHOLD}"
                            ),
                        }
                    )
                # INPUT: gt_file_endpath, f_endpath
                # OUTPUT: gt_file_endpath, f_endpath, best_alignment, mask_fpath, mask_name

                if args.verbose:
                    logging.info(
                        f"Image set '{image_set}': {total_gt_files} GT files, {matched_pairs} valid pairs"
                    )

                    # Log detailed pairing information for debugging
                    if matched_pairs > 0:
                        logging.debug(f"Detailed pairs for '{image_set}':")
                        pair_count = 0
                        for arg in args_in[
                            -matched_pairs:
                        ]:  # Get the pairs we just added
                            pair_count += 1
                            gt_name = os.path.basename(arg["gt_file_endpath"])
                            f_name = os.path.basename(arg["f_endpath"])
                            logging.debug(
                                f"  Pair {pair_count}: GT={gt_name} <-> Noisy={f_name}"
                            )

    if not args.verbose:
        logging.info(
            f"Found {total_image_sets} image sets with {total_gt_files_count} GT files and {total_matched_pairs} valid pairs"
        )

    # Run benchmark if requested
    if args.benchmark and len(args_in) > 0:
        run_alignment_benchmark(args_in)
        logging.info("Benchmark completed. Exiting.")
        sys.exit(0)

    logging.info(f"Processing {len(args_in)} image pairs...")
    processing_start = time.time()

    results = []
    cache_lock = threading.Lock()
    write_interval = 10  # Write cache every N results


    def save_result(result):
        """
        Thread-safe callback to save results incrementally.
        Writes to cache file every N results using atomic temp file operations.
        """
        with cache_lock:
            results.append(result)
            # Write to cache every write_interval results for efficiency
            if len(results) % write_interval == 0:
                try:
                    # Use atomic write: write to temp file, then rename
                    temp_fpath = content_fpath.with_suffix('.tmp')
                    with temp_fpath.open("w", encoding="utf-8") as f:
                        yaml.dump(results + (cached_results or []), f, allow_unicode=True)
                    temp_fpath.replace(content_fpath)  # Atomic on POSIX systems
                    logging.debug(f"Cache updated: {len(results)} results written")
                except Exception as e:
                    logging.warning(f"Failed to write cache file: {e}")

    try:
        utilities.mt_runner(
            rawproc.get_best_alignment_compute_gain_and_make_loss_mask,
            args_in,
            num_threads=args.num_threads,
            progress_desc="Processing",
            on_result=save_result,
        )

    except KeyboardInterrupt:
        logging.error(f"prep_image_dataset.py interrupted. Saving results.")

    # Final consolidation: write any remaining results not yet written due to batching
    if results:
        try:
            temp_fpath = content_fpath.with_suffix('.tmp')
            with temp_fpath.open("w", encoding="utf-8") as f:
                yaml.dump(results + (cached_results or []), f, allow_unicode=True)
            temp_fpath.replace(content_fpath)  # Atomic on POSIX systems
            logging.info(f"Final cache consolidation: {len(results)} results written")
        except Exception as e:
            logging.warning(f"Failed to write final cache consolidation: {e}")

    processing_time = time.time() - processing_start
    logging.info(
        f"Alignment and mask generation completed in {processing_time:.2f} seconds"
    )

    if cached_results:
        results = results + cached_results

    for result in results:  # FIXME
        result["crops"] = fetch_crops_list(
            result["image_set"],
            result["gt_fpath"],
            result["f_fpath"],
            result["is_bayer"],
            ds_base_dpath=os.path.join(DATASETS_ROOT, args.dataset),
        )
    utilities.dict_to_yaml(
        results,
        content_fpath,
    )
