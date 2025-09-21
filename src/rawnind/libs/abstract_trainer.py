"""Neural network training and evaluation framework for image-to-image models.

This module provides a comprehensive framework for training, evaluating, and using 
neural networks for image-to-image tasks, with a focus on raw image processing tasks
like denoising and compression. It implements a flexible class hierarchy that supports
various input types (Bayer patterns and RGB) and task types (denoising, compression, 
and combined approaches).

Key features:
- Modular design with a clear class hierarchy and inheritance patterns
- Support for different image formats (Bayer pattern and RGB)
- Configurable training, validation, and testing pipelines
- Automatic experiment management with checkpointing and result tracking
- Command line argument parsing with configuration file support
- Resource management (GPU/CPU, memory usage monitoring)
- Flexible loss functions and metrics for image quality assessment

Class hierarchy:
- ImageToImageNN: Base class for all image-to-image models
  - ImageToImageNNTraining: Extends base with training functionality
    - PRGBImageToImageNNTraining: Specialized for RGB image training
    - BayerImageToImageNNTraining: Specialized for Bayer pattern training
    - DenoiseCompressTraining: Combined denoising and compression training
    - DenoiserTraining: Pure denoising model training
  - BayerImageToImageNN: Specialized for Bayer pattern handling
  - DenoiseCompress: Combined denoising and compression model
  - Denoiser: Pure denoising model
  - BayerDenoiseCompress: Bayer-specific denoising+compression
  - BayerDenoiser: Bayer-specific denoising

Configuration files:
- config/denoise_bayer2prgb.yaml: Configuration for Bayer-to-RGB denoising
- config/denoise_prgb2prgb.yaml: Configuration for RGB-to-RGB denoising
- config/train_dc.yaml: Base configuration for denoising+compression
- config/train_dc_bayer2prgb.yaml: Configuration for Bayer-to-RGB denoising+compression
- config/train_dc_prgb2prgb.yaml: Configuration for RGB-to-RGB denoising+compression

Usage examples:
1. Training a denoiser:
   ```python
   trainer = DenoiserTraining(launch=True)
   trainer.training_loop()
   ```

2. Evaluating a trained model:
   ```python
   model = get_and_load_model(model_type="denoiser", load_path="path/to/model.pt")
   result = model.infer(input_image)
   ```
"""

import itertools
import logging
import os
import platform
import random
import shutil
import statistics
import sys
import time
from typing import Callable, Iterable, Optional

import configargparse
import psutil
import torch
import tqdm
import yaml

#
from . import json_saver, locking, pt_helpers, pt_losses, utilities
from rawnind.tools import save_src
from . import raw, rawds, rawproc
# from rawnind.extmodels import runet
# from rawnind.extmodels import edsr
from rawnind.models import bm3d_denoiser, compression_autoencoders, denoise_then_compress, manynets_compression, \
    raw_denoiser, standard_compressor
from rawnind.tools import find_best_expname_iteration

# from rawnind.tools.make_openexr_extraraw_files import EXTRARAW_DATA_DPATH
BREAKPOINT_ON_ERROR = False


def error_handler():
    """Handle critical errors during execution.
    
    This function provides a standardized way to handle critical errors in the training
    or evaluation process. Depending on the BREAKPOINT_ON_ERROR setting, it either:
    1. Enters interactive debugging mode with a breakpoint, or
    2. Exits the program with an error code.
    
    The function logs the error occurrence before taking action.
    
    Returns:
        None: Function either triggers a breakpoint or exits the program
        
    Notes:
        - Set BREAKPOINT_ON_ERROR to True for debugging during development
        - In production or automated environments, BREAKPOINT_ON_ERROR should be False
    """
    logging.error("error_handler")
    if BREAKPOINT_ON_ERROR:
        breakpoint()
    else:
        exit(1)



class ImageToImageNNTraining(ImageToImageNN):
    """Extension of ImageToImageNN adding training-specific utilities.

    Provides argument registration for training hyperparameters, training/validation
    loops, dataloader wiring and checkpointing helpers. Subclasses should override
    repack_batch() and step() to implement their data layout and loss computation.
    """
    """Training mixin adding loops, evaluation, and logging.

    Extends ImageToImageNN with training-specific features such as learning rate
    scheduling, validation/test routines, checkpointing, and dataloader wiring.
    """

    def __init__(self, **kwargs):
        """Initialize an image to image neural network trainer.

        Args:
            launch (bool): launch at init (otherwise user must call training_loop())
            **kwargs can be specified to overwrite configargparse args.
        """
        # skip if already initialized, by checking for self.optimizer
        if hasattr(self, "optimizer"):
            return

        super().__init__(**kwargs)
        # reset the logging basicConfig in case it's been called before

        self.init_optimizer()
        if self.load_path and (
                self.init_step > 0 or not self.reset_optimizer_on_fallback_load_path
        ):
            self.load_model(self.optimizer, self.load_path + ".opt", device=self.device)
        if self.reset_lr or (self.fallback_load_path and self.init_step == 0):
            self.reset_learning_rate()
        res_fpath: str = os.path.join(self.save_dpath, "trainres.yaml")
        self.json_saver = json_saver.YAMLSaver(
            res_fpath, warmup_nsteps=self.warmup_nsteps
        )
        logging.info(f"See {res_fpath} for results.")

        # get training data
        self.get_dataloaders()

        self.lr_adjustment_allowed_step: int = self.patience

        self.transfer = self.get_transfer_function(self.transfer_function)
        self.transfer_vt = self.get_transfer_function(self.transfer_function_valtest)

    def autocomplete_args(self, args):
        super().autocomplete_args(args)
        if not args.val_crop_size:
            args.val_crop_size = args.test_crop_size

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.init_lr)

    def adjust_lr(self, validation_losses: dict[str, float], step: int):
        model_improved = False
        for lossn, lossv in validation_losses.items():
            if lossv <= self.best_validation_losses[lossn]:
                logging.info(
                    f"self.best_validation_losses[{lossn}]={self.best_validation_losses[lossn]} <- {lossv=}"
                )
                self.best_validation_losses[lossn] = lossv
                self.lr_adjustment_allowed_step = step + self.patience
                model_improved = True
        if not model_improved and self.lr_adjustment_allowed_step < step:  # adjust lr
            old_lr = self.optimizer.param_groups[0]["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.lr_multiplier
            self.optimizer.param_groups[0]["lr"] *= (
                self.lr_multiplier
            )  # FIXME/BUG rm this duplicate multiplication. Currently lr_multiplier is squared as a result
            # there is an assertion that len=1 in init
            logging.info(
                f"adjust_lr: {old_lr} -> {self.optimizer.param_groups[0]['lr']}"
            )
            self.json_saver.add_res(step, {"lr": self.optimizer.param_groups[0]["lr"]})
            self.lr_adjustment_allowed_step = step + self.patience

    def reset_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.init_lr
        logging.info(f"reset_learning_rate to {self.optimizer.param_groups[0]['lr']}")

    # @classmethod
    def add_arguments(self, parser):
        super().add_arguments(parser)

        parser.add_argument(
            "--init_lr", type=float, help="Initial learning rate.", required=True
        )
        parser.add_argument(
            "--reset_lr",
            help="Reset learning rate of loaded model. (Defaults to true if fallback_load_path is set and init_step is 0)",
            action="store_true",
        )
        parser.add_argument(
            "--tot_steps", type=int, help="Number of training steps", required=True
        )

        parser.add_argument(
            "--val_interval",
            type=int,
            help="Number of steps between validation",
            required=True,
        )
        parser.add_argument(
            "--test_interval",
            type=int,
            help="Number of steps between tests",
            required=True,
        )

        parser.add_argument(
            "--crop_size", type=int, help="Training (batched) crop size", required=True
        )
        parser.add_argument(
            "--test_crop_size",
            type=int,
            help="Test (single-image) crop size",
            required=True,
        )
        parser.add_argument(
            "--val_crop_size",
            type=int,
            help="Validation (single-image) crop size. default uses test_crop_size",
        )
        parser.add_argument(
            "--test_reserve",
            nargs="*",
            help="Name of images which should be reserved for testing.",
            required=True,
        )
        parser.add_argument(
            "--bayer_only",
            help="Only use images which are available in Bayer format.",
            action="store_true",
        )
        parser.add_argument(
            "--transfer_function",
            help="Which transfer function (pq, gamma) is applied before the training loss.",
            required=True,
            choices=["pq", "gamma22", None, "None"],
        )
        parser.add_argument(
            "--transfer_function_valtest",
            help="Which transfer function (pq, gamma) is applied before the training loss in validation and tests.",
            required=True,
            choices=["pq", "gamma22", None, "None"],
        )
        parser.add_argument(
            "--patience", type=int, help="Number of steps to wait before LR updates"
        )
        parser.add_argument("--lr_multiplier", type=float, help="LR update multiplier")
        parser.add_argument(
            "--continue_training_from_last_model_if_exists",
            action="store_true",
            help="Continue the last training whose expname matches",
        )
        parser.add_argument(
            "--fallback_load_path",
            help="Path (or expname) of model to load if continue_training_from_last_model_if_exists is set but no previous models are found. Latest model is auto-detected from base expname",
        )
        parser.add_argument(
            "--reset_optimizer_on_fallback_load_path",
            action="store_true",
            help="Reset the optimizer when loading the fallback_load_path model.",
        )
        parser.add_argument(
            "--comment",
            help="Harmless comment which will appear in the log and be part of the expname",
        )
        parser.add_argument(
            "--num_crops_per_image",
            type=int,
            help="Number of crops per image. (Avoids loading too many large images.)",
            required=True,
        )
        parser.add_argument(
            "--batch_size_clean",
            type=int,
            help="Number of clean images in a batch.",
            required=True,
        )
        parser.add_argument(
            "--batch_size_noisy",
            type=int,
            help="Number of noisy images in a batch.",
            required=True,
        )

        parser.add_argument(
            "--noise_dataset_yamlfpaths",
            nargs="+",
            default=[rawproc.RAWNIND_CONTENT_FPATH],
            help="yaml file describing the paired dataset.",
        )
        parser.add_argument(
            "--clean_dataset_yamlfpaths",
            nargs="+",
            default=rawproc.EXTRARAW_CONTENT_FPATHS,
            help="yaml files describing the unpaired dataset.",
        )
        parser.add_argument(
            "--data_pairing",
            help="How to pair the clean and noisy images (x_y for pair, x_x otherwise).",
            required=True,
            choices=["x_y", "x_x", "y_y"],
        )
        parser.add_argument(
            "--arbitrary_proc_method",
            help="Use arbitrary processing in the input. (values are naive or opencv)",
        )
        parser.add_argument(
            "--warmup_nsteps",
            type=int,
            help="Number of steps to warmup. (Affects saving/loading models which are not considered below this step.)",
        )
        # parser.add_argument(
        #     "--exposure_diff_penalty",
        #     type=float,
        #     help="Penalty between input and reconstructed image (applied to both the network output and its processed version)",
        #     required=True,
        # )

    def validate_or_test(
            self,
            dataloader: Iterable,
            test_name: str,
            sanity_check: bool = False,
            save_individual_results: bool = True,
            save_individual_images: bool = False,
            # TODO merge with output_valtest_images (debug_options) and dataloader.OUTPUTS_IMAGE_FILES
    ):
        """Perform validation or testing on a dataset.
        
        This method runs the model on all samples from the provided dataloader,
        calculates performance metrics, and optionally saves individual results
        and output images. It supports distributed evaluation with a locking
        mechanism to prevent resource conflicts.
        
        The method performs these steps:
        1. Establishes a validation lock if needed to prevent parallel evaluations
        2. Processes each batch from the dataloader with the model
        3. Calculates losses and metrics for each sample
        4. Aggregates results and computes statistics
        5. Saves individual sample results and/or images if requested
        
        Args:
            dataloader: Iterable that yields batches of data, one image at a time
            test_name: Identifier for this validation/test run, used for file naming
            sanity_check: If True, runs a minimal validation for debugging purposes
            save_individual_results: If True, saves per-sample metrics to a YAML file
            save_individual_images: If True, saves model output images for each sample
                
        Returns:
            Dictionary with aggregated metrics and loss values
            
        Notes:
            - This method expects the dataloader to return one sample at a time
            - For progressive validation/testing, results from previous runs may be loaded
            - The validation lock prevents multiple processes from running CPU-intensive
              operations simultaneously on shared resources
        """
        # validation lock (TODO put in a function)
        own_lock = bypass_lock = printed_lock_warning = False
        lock_fpath = f"validation_{platform.uname()[1]}_{os.environ.get('CUDA_VISIBLE_DEVICES', 'unk')}.lock"
        if (platform.node() == "sd" or platform.node() == "bd") and (
                "manproc" not in test_name or self.arbitrary_proc_method == "opencv"
        ):
            bypass_lock = True

        while not own_lock and not bypass_lock:  # and 'manproc' in test_name:
            # the first line of the lock file contains the PID of the process which created it;
            # delete the lock if the process no longer exists
            if os.path.isfile(lock_fpath):
                with open(lock_fpath, "r") as f:
                    try:
                        pid = int(f.readline())
                    except ValueError:
                        pid = 0
                if not psutil.pid_exists(pid):
                    try:
                        os.remove(lock_fpath)
                    except FileNotFoundError:
                        pass
                    logging.warning(
                        f"validate_or_test: {lock_fpath} exists but process {pid} does not exist, deleting lock"
                    )
                elif pid == os.getpid():
                    own_lock = True
                else:
                    if not printed_lock_warning:
                        logging.warning(
                            f"validate_or_test: {lock_fpath} exists (owned by {pid=}), waiting for it to disappear"
                        )
                        printed_lock_warning = True
                    time.sleep(random.random() * 10)
            else:
                # write PID and launch command to lock_fpath
                with open(lock_fpath, "w") as f:
                    f.write(f"{os.getpid()}\n")
                    f.write(" ".join(sys.argv))
                if printed_lock_warning:
                    logging.warning(":)")

        with torch.no_grad():
            losses = {lossn: [] for lossn in (self.loss, *self.metrics)}

            individual_results = {}
            # load individual results if they exist and we care
            if save_individual_results:
                assert test_name is not None
                if "progressive" in test_name:
                    split_str = "_ge" if "ge" in test_name else "_le"
                    common_test_name = test_name.split(split_str)[0]
                elif "manproc_hq" in test_name:
                    common_test_name = test_name.replace("_hq", "")
                elif "manproc_gt" in test_name:
                    common_test_name = test_name.replace("_gt", "")
                elif "manproc_q995" in test_name:
                    common_test_name = test_name.replace("_q995", "")
                elif "manproc_q99" in test_name:
                    common_test_name = test_name.replace("_q99", "")
                else:
                    common_test_name = test_name
                os.makedirs(
                    os.path.join(self.save_dpath, common_test_name), exist_ok=True
                )

                if (
                        "progressive_test" in common_test_name
                        and "manproc" in common_test_name
                ):  # ugly hack to use the same individual_results for both progressive and std tests
                    if "manproc_bostitch" in common_test_name:
                        common_test_name_noprog = "manproc_bostitch"
                    else:
                        common_test_name_noprog = "manproc"
                    individual_results_fpath = os.path.join(
                        self.save_dpath,
                        common_test_name_noprog,
                        f"iter_{self.step_n}.yaml",
                    )
                    os.makedirs(
                        os.path.dirname(individual_results_fpath), exist_ok=True
                    )
                else:
                    individual_results_fpath = os.path.join(
                        self.save_dpath, common_test_name, f"iter_{self.step_n}.yaml"
                    )

                if os.path.isfile(individual_results_fpath):
                    individual_results = yaml.safe_load(open(individual_results_fpath))
                    print(f"Loaded {individual_results_fpath=}")
                else:
                    print(
                        f"No previous individual results {individual_results_fpath=} found"
                    )
            individual_images_dpath = os.path.join(
                self.save_dpath, common_test_name, f"iter_{self.step_n}"
            )
            if (
                    save_individual_images
                    or "output_valtest_images" in self.debug_options
                    or (
                    hasattr(dataloader, "OUTPUTS_IMAGE_FILES")
                    and dataloader.OUTPUTS_IMAGE_FILES
            )
            ):
                os.makedirs(individual_images_dpath, exist_ok=True)
            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # if dataloader.__name__ == 'batched_iterator':
                #     breakpoint()
                # mk image key

                if "y_fpath" in batch:
                    y_fn = (
                        batch["y_fpath"]
                        if isinstance(batch["y_fpath"], str)
                        else batch["y_fpath"][0]
                    )
                    y_fn = os.path.basename(y_fn)
                    image_key = y_fn
                    # if "gt_fpath" in batch:
                    #     gt_fn = batch["gt_fpath"] if isinstance(batch["gt_fpath"], str) else batch["gt_fpath"][0]
                    #     gt_fn = os.path.basename(gt_fn)
                    #     image_key += f"_aligned_to_{gt_fn}"
                    if "image_set" in batch:
                        image_key = f"{batch['image_set']}_{image_key}"
                    if "gt_fpath" in batch and "aligned_to" not in image_key:
                        gt_fn = (
                            batch["gt_fpath"]
                            if isinstance(batch["gt_fpath"], str)
                            else batch["gt_fpath"][0]
                        )
                        if "aligned_to" in gt_fn and batch["image_set"] in gt_fn:
                            gt_fn = gt_fn.split("_aligned_to_")[0].split(
                                f"{batch['image_set']}_"
                            )[-1]
                        image_key += f"_aligned_to_{os.path.basename(gt_fn)}"
                else:
                    image_key = i

                if save_individual_results and image_key in individual_results:
                    for lossn, lossv in individual_results[image_key].items():
                        if lossn not in losses:
                            losses[lossn] = []
                        losses[lossn].append(lossv)
                    # if dataloader.__name__ == 'batched_iterator':
                    #     breakpoint()
                    # print(f"DBG: skipping {image_key} (already known result)")
                    continue
                individual_results[image_key] = {}
                x_crops = batch["x_crops"].to(self.device)
                y_crops = batch[
                    "y_crops"
                ].to(
                    self.device, x_crops.dtype
                )  # 2023-08-30: fixed bug w/ match_gain == output: y_crops was always * batch["gain"]
                mask_crops = batch["mask_crops"].to(self.device)
                # print(batch["y_crops"].shape)
                try:
                    model_output = self.model(y_crops)
                    if isinstance(model_output, dict):
                        reconstructed_image, bpp = (
                            model_output["reconstructed_image"],
                            model_output["bpp"],
                        )
                    else:
                        reconstructed_image = model_output
                        bpp = None
                except RuntimeError as e:
                    try:
                        if not bypass_lock:
                            os.remove(lock_fpath)
                    except FileNotFoundError:
                        pass
                    logging.error(
                        f"Error {e} with {batch['gt_fpath']=}, {batch['y_fpath']=}, {y_crops.shape=}, {x_crops.shape=}, {mask_crops.shape=}"
                    )
                    if BREAKPOINT_ON_ERROR:
                        breakpoint()
                    else:
                        exit(1)
                if self.match_gain == "output":
                    processed_output = rawproc.match_gain(x_crops, reconstructed_image)
                else:
                    processed_output = reconstructed_image
                if hasattr(self, "process_net_output"):  # Bayer color transform
                    processed_output = self.process_net_output(
                        processed_output, batch["rgb_xyz_matrix"], x_crops
                    )
                if (
                        "output_valtest_images" in self.debug_options
                ):  # this is pretty ugly :/
                    self._dbg_output_testval_images(
                        batch=batch,
                        processed_output=processed_output,
                        individual_images_dpath=individual_images_dpath,
                        i=i,
                        x_crops=x_crops,
                        y_crops=y_crops,
                        mask_crops=mask_crops,
                    )
                if "net_output_processor_fun" in batch:
                    processed_output_fpath = os.path.join(
                        individual_images_dpath,
                        image_key,
                    )
                    processed_output = batch["net_output_processor_fun"](
                        processed_output, output_fpath=processed_output_fpath
                    )
                else:
                    processed_output = self.transfer_vt(processed_output)
                    x_crops = self.transfer_vt(x_crops)
                # for lossn, lossf in ({self.loss: self.lossf} | self.metrics).items():  # python 38 310 compat

                loss_functions = self.metrics.copy()  # compat
                loss_functions[self.loss] = self.lossf  # compat
                for lossn, lossf in loss_functions.items():  # compat
                    try:
                        lossv = lossf(
                            processed_output * mask_crops, x_crops * mask_crops
                        ).item()
                    except RuntimeError as e:
                        try:
                            if not bypass_lock:
                                os.remove(lock_fpath)
                        except FileNotFoundError:
                            pass
                        logging.error(
                            f"Error {e} with {batch['gt_fpath']=}, {batch['y_fpath']=}, {y_crops.shape=}, {x_crops.shape=}, {processed_output.shape=}, {reconstructed_image.shape=}, {mask_crops.shape=}, {image_key=}"
                        )
                        breakpoint()
                    losses[lossn].append(lossv)
                    print(f"DBG: {lossn=}, {lossv=}")
                    individual_results[image_key][lossn] = lossv

                if bpp is not None:
                    print(f"DBG: {bpp=}")
                    if "bpp" not in losses:
                        losses["bpp"] = []
                        losses["combined"] = []
                    losses["bpp"].append(float(bpp))
                    individual_results[image_key]["bpp"] = float(bpp)
                    combined_loss = float(
                        bpp
                        + self.lossf(
                            processed_output * mask_crops, x_crops * mask_crops
                        ).item()
                        * self.train_lambda
                    )
                    losses["combined"].append(combined_loss)
                    individual_results[image_key]["combined"] = combined_loss

                if sanity_check and i >= 1:
                    break

            if save_individual_results:
                # print(individual_results)
                utilities.dict_to_yaml(individual_results, individual_results_fpath)
                # yaml.safe_dump(individual_results, open(individual_results_fpath, "w"))
        torch.cuda.empty_cache()
        try:
            if not bypass_lock:
                os.remove(lock_fpath)
        except FileNotFoundError:
            pass
        try:
            return {lossn: statistics.mean(lossv) for lossn, lossv in losses.items()}
        except statistics.StatisticsError as e:
            logging.error(f"Error {e} with {losses=}")
            breakpoint()

    def training_loop(self):
        last_test_step = last_val_step = self.step_n = self.init_step
        # Run an initial validation and test to ensure everything works well

        validation_losses = self.validate_or_test(
            dataloader=self.cleannoisy_val_dataloader,
            sanity_check="skip_initial_validation" in self.debug_options,
            test_name="val",
        )
        torch.cuda.empty_cache()
        if "skip_initial_validation" in self.debug_options:
            self.best_validation_losses: dict[str, float] = {
                ln: 9001 for ln in validation_losses
            }
        else:
            logging.info(f"training_loop: {self.step_n=}, {validation_losses=}")
            self.json_saver.add_res(
                self.step_n,
                {
                    f"val_{lossn + self._get_lossn_extension()}": lossv
                    for lossn, lossv in validation_losses.items()
                },
            )
            self.best_validation_losses = (
                validation_losses  # validation_losses[self.loss]
            )
        self.validate_or_test(
            dataloader=self.cleannoisy_test_dataloader,
            sanity_check=True,
            test_name="sanitytest",
        )
        torch.cuda.empty_cache()
        # training loop
        while self.step_n <= self.tot_steps:
            num_training_steps = min(
                self.val_interval + last_val_step - self.step_n,
                self.test_interval + last_test_step - self.step_n,
            )
            if "spam" in self.debug_options:
                logging.debug(f"{num_training_steps=} to do")
            training_loss = self.train(
                optimizer=self.optimizer,
                num_steps=num_training_steps,
                dataloader_cc=self.cleanclean_dataloader,
                dataloader_cn=self.cleannoisy_dataloader,
            )
            self.step_n += num_training_steps
            logging.info(
                f"training_loop: {self.step_n=}, {training_loss=} (over {num_training_steps=})"
            )
            self.json_saver.add_res(
                self.step_n,
                {f"train_{self.loss}": training_loss},
            )
            if self.step_n >= last_val_step + self.val_interval:
                validation_losses = self.validate_or_test(
                    dataloader=self.cleannoisy_val_dataloader, test_name="val"
                )
                torch.cuda.empty_cache()
                logging.info(f"training_loop: {self.step_n=}, {validation_losses=}")
                self.json_saver.add_res(
                    self.step_n,
                    {
                        f"val_{lossn + self._get_lossn_extension()}": lossv
                        for lossn, lossv in validation_losses.items()
                    },
                )
                last_val_step = self.step_n
                self.adjust_lr(validation_losses=validation_losses, step=self.step_n)
            if self.step_n >= last_test_step + self.test_interval:
                test_losses = self.validate_or_test(
                    dataloader=self.cleannoisy_test_dataloader, test_name="test"
                )
                torch.cuda.empty_cache()
                logging.info(f"training_loop: {self.step_n=}, {test_losses=}")
                self.json_saver.add_res(
                    self.step_n,
                    {
                        f"test_{lossn + self._get_lossn_extension()}": lossv
                        for lossn, lossv in test_losses.items()
                    },
                )
                last_test_step = self.step_n

            self.cleanup_models()
            self.save_model(self.step_n)
            # TODO: if validation_loss[self.loss] > self.failure_threshold:  reset

    def get_dataloaders(self) -> None:
        """Instantiate the train/val/test data-loaders into self."""
        class_specific_arguments = {}
        if self.arbitrary_proc_method:
            class_specific_arguments["arbitrary_proc_method"] = (
                self.arbitrary_proc_method
            )
        if self.in_channels == 3:
            cleanclean_dataset_class = (
                rawds.CleanProfiledRGBCleanProfiledRGBImageCropsDataset
            )
            cleannoisy_dataset_class = (
                rawds.CleanProfiledRGBNoisyProfiledRGBImageCropsDataset
            )
            val_dataset_class = (
                rawds.CleanProfiledRGBNoisyProfiledRGBImageCropsValidationDataset
            )
            test_dataloader_class = (
                rawds.CleanProfiledRGBNoisyProfiledRGBImageCropsTestDataloader  # (
                # content_fpath, crop_size, test_reserve
                # )
            )
        elif self.in_channels == 4:
            cleanclean_dataset_class = rawds.CleanProfiledRGBCleanBayerImageCropsDataset
            cleannoisy_dataset_class = rawds.CleanProfiledRGBNoisyBayerImageCropsDataset
            val_dataset_class = (
                rawds.CleanProfiledRGBNoisyBayerImageCropsValidationDataset
            )
            test_dataloader_class = (
                rawds.CleanProfiledRGBNoisyBayerImageCropsTestDataloader
            )  # (
            # content_fpath, crop_size, test_reserve
            # )
            self.color_converter = cleannoisy_dataset_class.camRGB_to_profiledRGB_img
        else:
            raise ValueError(f"{self.in_channels=}")

        if not self.test_only:
            cleanclean_dataset = cleanclean_dataset_class(
                content_fpaths=self.clean_dataset_yamlfpaths,
                num_crops=self.num_crops_per_image,
                crop_size=self.crop_size,
                toy_dataset="toy_dataset" in self.debug_options,
                **class_specific_arguments,
                # test_reserve=self.test_reserve,
            )
            cleannoisy_dataset = cleannoisy_dataset_class(
                content_fpaths=self.noise_dataset_yamlfpaths,
                num_crops=self.num_crops_per_image,
                crop_size=self.crop_size,
                test_reserve=self.test_reserve,
                test="learn_validation" in self.debug_options,
                bayer_only=self.bayer_only,
                toy_dataset="toy_dataset" in self.debug_options,
                data_pairing=self.data_pairing,
                match_gain=self.match_gain == "input",
                **class_specific_arguments,
            )
            # ugly hack to avoid loading a dataset with 0-batch-size and ensuring two datasets for compat
            assert self.batch_size_clean > 0 or self.batch_size_noisy > 0
            if self.batch_size_clean == 0:
                cleanclean_dataset = cleannoisy_dataset
                self.batch_size_clean = 1
                self.batch_size_noisy = self.batch_size_noisy - 1
            elif self.batch_size_noisy == 0:
                cleannoisy_dataset = cleanclean_dataset
                self.batch_size_noisy = 1
                self.batch_size_clean = self.batch_size_clean - 1

            if "1thread" in self.debug_options:
                num_threads_cc = 0
                num_threads_cn = 0
            elif "minimize_threads" in self.debug_options:
                num_threads_cc = self.batch_size_clean
                num_threads_cn = self.batch_size_noisy
            else:
                num_threads_cc = max(
                    self.batch_size_clean + 1, self.batch_size_clean * 2, 3
                )
                num_threads_cn = max(
                    self.batch_size_noisy + 1, int(self.batch_size_noisy * 1.5)
                )
            self.cleanclean_dataloader = torch.utils.data.DataLoader(
                dataset=cleanclean_dataset,
                batch_size=self.batch_size_clean,
                shuffle=True,
                pin_memory=True,
                num_workers=num_threads_cc,
            )
            self.cleannoisy_dataloader = torch.utils.data.DataLoader(
                dataset=cleannoisy_dataset,
                batch_size=self.batch_size_noisy,
                shuffle=True,
                pin_memory=True,
                num_workers=num_threads_cn,
            )
        # FIXME put back tab
        start_time = time.time()
        cleannoisy_val_dataset = val_dataset_class(
            content_fpaths=self.noise_dataset_yamlfpaths,
            crop_size=self.val_crop_size,
            test_reserve=self.test_reserve,
            bayer_only=self.bayer_only,
            toy_dataset="toy_dataset" in self.debug_options,
            match_gain=self.match_gain == "input",
            data_pairing=self.data_pairing,
            **class_specific_arguments,
        )
        self.cleannoisy_val_dataloader = torch.utils.data.DataLoader(
            dataset=cleannoisy_val_dataset,
            batch_size=1,
            shuffle=False,
        )
        logging.info(f"val_dataloader loading time: {time.time() - start_time}")
        # print('DBG FIXME: test_dataloader is hard-disabled')
        try:
            self.cleannoisy_test_dataloader = test_dataloader_class(
                content_fpaths=self.noise_dataset_yamlfpaths,
                crop_size=self.test_crop_size,
                test_reserve=self.test_reserve,
                bayer_only=self.bayer_only,
                toy_dataset="toy_dataset" in self.debug_options,
                match_gain=self.match_gain == "input",
                **class_specific_arguments,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'{e}. Try running with "--noise_dataset_yamlfpaths {rawproc.RAWNIND_CONTENT_FPATH}"?'
            )

    def save_model(self, step: int) -> None:
        fpath = os.path.join(self.save_dpath, "saved_models", f"iter_{step}.pt")
        torch.save(self.model.state_dict(), fpath)
        torch.save(self.optimizer.state_dict(), fpath + ".opt")

    def cleanup_models(self):
        keepers: list[str] = [
            f"iter_{step}" for step in self.json_saver.get_best_steps()
        ]
        for fn in os.listdir(os.path.join(self.save_dpath, "saved_models")):
            if fn.partition(".")[0] not in keepers:
                logging.info(
                    f"cleanup_models: rm {os.path.join(self.save_dpath, 'saved_models', fn)}"
                )
                os.remove(os.path.join(self.save_dpath, "saved_models", fn))
        if "output_valtest_images" in self.debug_options:
            if os.path.isdir(os.path.join(self.save_dpath, "visu")):
                for dn in os.listdir(os.path.join(self.save_dpath, "visu")):
                    if dn not in keepers:
                        logging.info(
                            f"cleanup_models: rm -r {os.path.join(self.save_dpath, 'visu', dn)}"
                        )
                        shutil.rmtree(
                            os.path.join(self.save_dpath, "visu", dn),
                            ignore_errors=True,
                        )

    def train(
            self,
            optimizer: torch.optim.Optimizer,
            num_steps: int,
            dataloader_cc: Iterable,
            dataloader_cn: Iterable,
    ) -> float:
        last_time = time.time()
        # for i, batch in enumerate(
        step_losses: list[float] = []
        first_step: bool = True
        i: int = 0
        for batch in itertools.islice(zip(dataloader_cc, dataloader_cn), 0, num_steps):
            if "timing" in self.debug_options or "spam" in self.debug_options:
                logging.debug(f"data {i} loading time: {time.time() - last_time}")
                last_time: float = time.time()
            locking.check_pause()
            step_losses.append(
                self.step(
                    batch,
                    optimizer=optimizer,
                    output_train_images=(
                            first_step and "output_train_images" in self.debug_options
                    ),
                )
            )
            if "timing" in self.debug_options or "spam" in self.debug_options:
                logging.debug(f"total step {i} time: {time.time() - last_time}")
                last_time: float = time.time()
                i += 1
            first_step = False
        return statistics.mean(step_losses)

    def offline_validation(self):
        """Only validate model (same as in training_loop but called externally / without train)"""
        if "step_n" not in vars(self):
            self.step_n = self.init_step
        logging.info(f"test_and_validate_model: {self.step_n=}")
        if self.step_n not in self.json_saver.results:
            self.json_saver.results[
                self.step_n
            ] = {}  # this shouldn't happen but sometimes the results file is not properly synchronized and we are stuck with an old version I guess
        if (
                "val_" + self.loss + self._get_lossn_extension()
                not in self.json_saver.results[self.step_n]
        ):
            val_losses = self.validate_or_test(
                dataloader=self.cleannoisy_val_dataloader, test_name="val"
            )
            logging.info(f"validation: {self.step_n=}, {val_losses=}")
            self.json_saver.add_res(
                self.step_n,
                {
                    f"val_{lossn + self._get_lossn_extension()}": lossv
                    for lossn, lossv in val_losses.items()
                },
            )

    def offline_std_test(self):
        """Std test (same as in training but run externally)."""
        if "step_n" not in vars(self):
            self.step_n = self.init_step
        print(f"test_and_validate_model: {self.step_n=}")
        if (
                "test_" + self.loss + self._get_lossn_extension()
                in self.json_saver.results[self.step_n]
        ):
            return
        test_losses = self.validate_or_test(
            dataloader=self.cleannoisy_test_dataloader, test_name="test"
        )
        logging.info(f"test: {self.step_n=}, {test_losses=}")
        self.json_saver.add_res(
            self.step_n,
            {
                f"test_{lossn + self._get_lossn_extension()}": lossv
                for lossn, lossv in test_losses.items()
            },
        )

    def offline_custom_test(
            self, dataloader, test_name: str, save_individual_images=False
    ):
        print(f"custom_test: {test_name=}")
        if not hasattr(self, "step_n"):
            self.step_n = self.init_step
        assert self.step_n != 0, "likely failed to get the right model"
        if f"{test_name}_{self.loss}" in self.json_saver.results[self.step_n]:
            print(
                f"custom_test: {test_name=} already done: {self.json_saver.results[self.step_n]}"
            )
            return
        test_losses = self.validate_or_test(
            dataloader,
            test_name=test_name,
            save_individual_results=True,
            save_individual_images=save_individual_images,
        )
        logging.info(f"test {test_name}: {self.step_n=}, {test_losses=}")
        self.json_saver.add_res(
            self.step_n,
            {
                f"{test_name}_{lossn + self._get_lossn_extension()}": lossv
                for lossn, lossv in test_losses.items()
            },
        )

    def compute_train_loss(
            self,
            mask,
            processed_output,
            processed_gt,
            bpp,
            # approx_exposure_diff: torch.Tensor,
    ) -> torch.Tensor:
        # compute loss
        masked_proc_output = processed_output * mask
        masked_proc_gt = processed_gt * mask
        loss = self.lossf(masked_proc_output, masked_proc_gt) * vars(self).get(
            "train_lambda", 1.0
        )
        # penalize exposure difference
        # approx_exposure_diff = (
        #     #    masked_proc_output.mean() - masked_proc_gt.mean()
        #     # ).abs() + (
        #     (reconstructed_image * batch["mask_crops"]).mean()
        #     - (batch["x_crops"] * batch["mask_crops"]).mean()
        # ).abs()
        # approx_exposure_diff = (masked_proc_output.mean() - masked_proc_gt.mean()).abs()
        # if approx_exposure_diff > 0.1:
        #     loss += approx_exposure_diff**2 * self.exposure_diff_penalty

        loss += bpp
        return loss

    # def match_gain_prior_to_rebatch(self, batch):
    ## no longer needed now that it's done in the dataset
    #     for b in batch:
    #         if "y_crops" in b:
    #             b["y_crops"] *= b["gain"].view(len(b["y_crops"]), 1, 1, 1, 1)
    #     return batch

    def _get_lossn_extension(self):
        lossn_extension = ""
        if self.arbitrary_proc_method:
            lossn_extension += ".arbitraryproc"
        elif self.transfer_function_valtest != "pq":
            lossn_extension += f".{self.transfer_function_valtest}"
        return lossn_extension


class PRGBImageToImageNNTraining(ImageToImageNNTraining):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # def repack_batch(batch: tuple[dict], device: torch.device) -> dict:  # python 38 310 compat
    @staticmethod
    def repack_batch(batch: tuple, device: torch.device) -> dict:
        """
        input:
        tuple of dict(x_crops, y_crops, mask_crops, gain)
        where x_crops, y_crops have dimensions batch_size, num_crops_per_image, ch, h, w,
        mask_crops have dimensions batch_size, num_crops_per_image, h, w
        gain is a float

        output:
        dict(x_crops, y_crops, mask_crops)
        where x_crops, y_crops, mask_crops have dimensions batch_size * num_crops_per_image, ch, h, w

        all repacked data is moved to device
        """
        repacked_batch = dict()
        if "y_crops" not in batch[0]:
            batch[0]["y_crops"] = batch[0]["x_crops"]
        for akey in ("x_crops", "y_crops", "mask_crops"):
            repacked_batch[akey] = torch.cat(
                (
                    batch[0][akey].view(-1, *(batch[0][akey].shape)[2:]),
                    batch[1][akey].view(-1, *(batch[1][akey].shape)[2:]),
                )
            ).to(device)
        return repacked_batch

    def step(
            self,
            batch,
            optimizer: torch.optim.Optimizer,
            output_train_images: bool = False,
            **kwargs,
    ):  # WIP
        # unpack data, flatten intra/inter images, and transfer to device
        # last_time = time.time()
        # if self.match_gain == "input":
        #     batch = self.match_gain_prior_to_rebatch(batch)
        try:
            batch = self.repack_batch(batch, self.device)
        except KeyError as e:
            logging.error(e)
        # print(f"repacking time: {time.time()-last_time}")
        # last_time = time.time()
        model_output = self.model(batch["y_crops"])

        if isinstance(self, DenoiseCompressTraining):
            if "spam" in self.debug_options and random.random() < 0.01:
                logging.debug(
                    f"DBG: {model_output['used_dists']=}, {model_output['num_forced_dists']=}"
                )
            reconstructed_image, bpp = (
                model_output["reconstructed_image"],
                model_output["bpp"],
            )
        else:
            reconstructed_image = model_output
            bpp = 0
        # if self.exposure_diff_penalty > 0:
        #     approx_exposure_diff = self.compute_approx_exposure_diff(
        #         batch["x_crops"],
        #         batch["y_crops"],
        #         reconstructed_image,
        #         batch["mask_crops"],
        #     )
        # else:
        #     approx_exposure_diff = 0
        # print(f"model_output time: {time.time()-last_time}")
        # last_time = time.time()
        # match exposure, apply color profile, apply gamma
        # if 'no_match_gains' in self.debug_options:
        #     processed_output = self.transfer(reconstructed_image)
        #     processed_gt = self.transfer(batch["x_crops"])
        # else:

        if self.match_gain == "output":
            reconstructed_image = rawproc.match_gain(
                batch["x_crops"], reconstructed_image
            )
        else:
            reconstructed_image = reconstructed_image

        if output_train_images:  # FIXME (current copy of bayer version. ideally should be a function but oh well)
            # should output reconstructed_image, batch["y_crops"], batch["x_crops"]
            # print(
            #    f"training {batch['y_crops'].mean((0,2,3))=}, {model_output.mean((0,2,3))=}"
            # )
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{self.step_n}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                raw.hdr_nparray_to_file(
                    reconstructed_image[i].detach().cpu().numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_reconstructed.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (reconstructed_image[i].detach() * batch["mask_crops"][i])
                    .cpu()
                    .numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_reconstructed_masked.exr",
                    ),
                    color_profile="lin_rec2020",
                )

                raw.hdr_nparray_to_file(
                    (batch["y_crops"][i]).cpu().numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_input.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    batch["x_crops"][i].cpu().numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_gt.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    self.transfer(batch["x_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt_transfered.exr"),
                    color_profile="lin_rec2020",
                )

        reconstructed_image = self.transfer(reconstructed_image)
        # print(f"processed_output time: {time.time()-last_time}")
        # last_time = time.time()
        gt = self.transfer(batch["x_crops"])
        # if "no_match_gains" not in self.debug_options:
        #     processed_output = rawproc.match_gain(processed_gt, processed_output)
        # print(f"processed_input time: {time.time()-last_time}")
        # last_time = time.time()
        # apply mask, compute loss
        loss = self.compute_train_loss(
            batch["mask_crops"],
            reconstructed_image,
            gt,
            bpp,  # , approx_exposure_diff
        )
        # loss = lossf(
        #     processed_output * batch["mask_crops"],
        #     processed_gt * batch["mask_crops"],
        # )
        # if bpp is not None:
        #     loss = loss * self.train_lambda + bpp
        # print(f"loss time: {time.time()-last_time}")
        # last_time = time.time()
        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        # print(f"backward time: {time.time()-last_time}")
        # last_time = time.time()
        if isinstance(self, DenoiseCompressTraining):
            DenoiseCompressTraining.clip_gradient(optimizer, 5)
        optimizer.step()
        # print(f"optimizer time: {time.time()-last_time}")
        # last_time = time.time()
        return loss.item()

    def _dbg_output_testval_images(
            self,
            batch,
            processed_output,
            individual_images_dpath,
            i,
            x_crops,
            y_crops,
            mask_crops,
    ):
        if isinstance(batch["y_fpath"], list) and len(batch["y_fpath"]) == 1:
            batch["y_fpath"] = batch["y_fpath"][0]
            batch["gt_fpath"] = batch["gt_fpath"][0]
        raw.hdr_nparray_to_file(
            (processed_output * mask_crops)[0].cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i}_{'' if 'y_fpath' not in batch else batch['y_fpath'].split('/')[-1]}_output_masked.exr",
            ),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            processed_output[0].cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i}_{'' if 'y_fpath' not in batch else batch['y_fpath'].split('/')[-1]}_output.exr",
            ),
            color_profile="lin_rec2020",
        )
        gt_fpath = os.path.join(
            individual_images_dpath,
            f"{i}_{'' if 'gt_fpath' not in batch else batch['gt_fpath'].split('/')[-1]}_gt.exr",
        )
        if not os.path.isfile(gt_fpath):
            raw.hdr_nparray_to_file(
                (x_crops * mask_crops)[0].cpu().numpy(),
                gt_fpath,
                color_profile="lin_rec2020",
            )
        raw.hdr_nparray_to_file(
            (y_crops * mask_crops)[0].cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i}_{'' if 'y_fpath' not in batch else batch['y_fpath'].split('/')[-1]}_input.exr",
            ),
            color_profile="lin_rec2020",
        )




class BayerImageToImageNNTraining(ImageToImageNNTraining, BayerImageToImageNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    # def repack_batch(batch: tuple[dict], device: torch.device) -> dict:  # python 38 310 compat
    def repack_batch(batch: tuple, device: torch.device) -> dict:
        """
        input:
        tuple of dict(x_crops, y_crops, mask_crops, rgb_xyz_matrix)
        where x_crops, y_crops have dimensions batch_size, num_crops_per_image, ch, h, w,
        mask_crops have dimensions batch_size, num_crops_per_image, h, w
        and rgb_xyz_matrix have dimensions batch_size, 4, 3

        output:
        dict(x_crops, y_crops, mask_crops, rgb_xyz_matrix)
        where x_crops, y_crops, mask_crops have dimensions batch_size * num_crops_per_image, ch, h, w
        and rgb_xyz_matrix have dimensions batch_size * num_crops_per_image, 4, 3

        all repacked data is moved to device
        """
        repacked_batch: dict = dict()
        for akey in ("x_crops", "y_crops", "mask_crops"):
            repacked_batch[akey] = torch.cat(
                (
                    batch[0][akey].view(-1, *(batch[0][akey].shape)[2:]),
                    batch[1][akey].view(-1, *(batch[1][akey].shape)[2:]),
                )
            ).to(device)

        num_crops_per_image = batch[0]["x_crops"].shape[1]
        repacked_batch["rgb_xyz_matrix"] = torch.cat(
            (
                batch[0]["rgb_xyz_matrix"].repeat_interleave(
                    num_crops_per_image, dim=0
                ),
                batch[1]["rgb_xyz_matrix"].repeat_interleave(
                    num_crops_per_image, dim=0
                ),
            )
        )  # .to(device) # workaround for https://github.com/pytorch/pytorch/issues/86465
        assert (
                repacked_batch["rgb_xyz_matrix"].shape[0]
                == repacked_batch["x_crops"].shape[0]
        )
        return repacked_batch

    def step(
            self,
            batch,
            optimizer: torch.optim.Optimizer,
            output_train_images: bool = False,
    ):  # WIP
        # unpack data, flatten intra/inter images, and transfer to device
        # last_time = time.time()
        # if self.match_gain == "input":
        #     batch = self.match_gain_prior_to_rebatch(batch)
        batch = self.repack_batch(batch, self.device)
        # print(f"repacking time: {time.time()-last_time}")
        # last_time = time.time()
        if "timing" in self.debug_options or "spam" in self.debug_options:
            last_time = time.time()

        model_output = self.model(batch["y_crops"])
        if isinstance(self, DenoiseCompressTraining):
            reconstructed_image, bpp = (
                model_output["reconstructed_image"],
                model_output["bpp"],
            )
        else:
            reconstructed_image = model_output
            bpp = 0
        if "timing" in self.debug_options or "spam" in self.debug_options:
            logging.debug(f"model time: {time.time() - last_time}")
            last_time = time.time()
        # print(f"model_output time: {time.time()-last_time}")
        # last_time = time.time()
        # match exposure, apply color profile, apply gamma
        # if self.exposure_diff_penalty > 0:
        #     approx_exposure_diff = self.compute_approx_exposure_diff(
        #         batch["x_crops"],
        #         batch["y_crops"],
        #         reconstructed_image,
        #         batch["mask_crops"],
        #     )
        # else:
        #     approx_exposure_diff = 0
        processed_output = self.process_net_output(
            reconstructed_image, batch["rgb_xyz_matrix"], batch["x_crops"]
        )
        if output_train_images:
            # print(
            #    f"training {batch['y_crops'].mean((0,2,3))=}, {model_output.mean((0,2,3))=}"
            # )
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{self.step_n}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                with open(
                        os.path.join(visu_save_dir, f"train_{i}_xyzm.txt"), "w"
                ) as fp:
                    fp.write(f"{batch['rgb_xyz_matrix'][i]}")
                y_processed = (
                    self.process_net_output(
                        rawproc.demosaic(batch["y_crops"][i: i + 1].cpu()),
                        batch["rgb_xyz_matrix"][i: i + 1].cpu(),
                        batch["x_crops"][i: i + 1].cpu(),
                    )
                    .squeeze(0)
                    .numpy()
                )
                raw.hdr_nparray_to_file(
                    y_processed,
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_debayered_ct_y.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (processed_output[i].detach() * batch["mask_crops"][i])
                    .cpu()
                    .numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_processed_output_masked.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    processed_output[i].detach().cpu().numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_processed_output.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (reconstructed_image[i].detach() * batch["mask_crops"][i])
                    .cpu()
                    .numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_output.exr",
                    ),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    batch["x_crops"][i].cpu().numpy(),
                    os.path.join(
                        visu_save_dir,
                        f"train_{i}_gt.exr",
                    ),
                    color_profile="lin_rec2020",
                )
        processed_output = self.transfer(processed_output)
        # print(f"processed_output time: {time.time()-last_time}")
        # last_time = time.time()
        gt = self.transfer(batch["x_crops"])

        # print(f"processed_input time: {time.time()-last_time}")
        # last_time = time.time()
        # apply mask, compute loss
        if "timing" in self.debug_options or "spam" in self.debug_options:
            logging.debug(f"processing time: {time.time() - last_time}")
            last_time = time.time()
        loss = self.compute_train_loss(
            batch["mask_crops"],
            processed_output,
            gt,
            bpp,  # , approx_exposure_diff
        )

        # print(f"loss time: {time.time()-last_time}")
        # last_time = time.time()
        # backpropagate and optimize
        if "timing" in self.debug_options or "spam" in self.debug_options:
            logging.debug(f"loss time: {time.time() - last_time}")
            last_time = time.time()
        optimizer.zero_grad()
        loss.backward()
        if isinstance(self, DenoiseCompressTraining):
            DenoiseCompressTraining.clip_gradient(optimizer, 5)
        # print(f"backward time: {time.time()-last_time}")
        # last_time = time.time()

        optimizer.step()
        if "timing" in self.debug_options or "spam" in self.debug_options:
            logging.debug(f"bw+optim: {time.time() - last_time}")
            last_time = time.time()
        # print(f"optimizer time: {time.time()-last_time}")
        # last_time = time.time()
        return loss.item()

    def _dbg_output_testval_images(
            self,
            batch,
            processed_output,
            individual_images_dpath,
            i,
            x_crops,
            y_crops,
            mask_crops,
    ):
        # print(
        #    f"valtest {y_crops.mean((0,2,3))=}, {model_output.mean((0,2,3))=}"
        # )
        with open(
                os.path.join(
                    individual_images_dpath,
                    f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_xyzm.txt",
                ),
                "w",
        ) as fp:
            fp.write(f"{batch['rgb_xyz_matrix']}")
        # raw.hdr_nparray_to_file(
        #     reconstructed_image.squeeze(0).cpu().numpy(),
        #     os.path.join(
        #         individual_images_dpath,
        #         f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_output.exr",
        #     ),
        #     color_profile="lin_rec2020",
        # )
        raw.hdr_nparray_to_file(
            self.process_net_output(
                rawproc.demosaic(batch["y_crops"]),
                batch["rgb_xyz_matrix"],
                batch["x_crops"],
            )
            .squeeze(0)
            .cpu()
            .numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_debayered_ct_y.exr",
            ),
            color_profile="lin_rec2020",
        )
        # breakpoint()
        raw.hdr_nparray_to_file(
            (processed_output * mask_crops).squeeze(0).cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_processed_output_masked.exr",
            ),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            processed_output.squeeze(0).cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_processed_output.exr",
            ),
            color_profile="lin_rec2020",
        )

        raw.hdr_nparray_to_file(
            x_crops[0].cpu().numpy(),
            os.path.join(
                individual_images_dpath,
                f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_gt.exr",
            ),
            color_profile="lin_rec2020",
        )
        # raw.hdr_nparray_to_file(
        #     (
        #         self.process_camRGB_batch(y_crops, rgb_xyz_matrix, x_crops)
        #         * mask_crops
        #     )[0]
        #     .cpu()
        #     .numpy(),
        #     os.path.join(visu_save_dir, f"{i}_processed_input.exr"),
        #     color_profile="lin_rec2020",
        # )
        # pt_helpers.sdr_pttensor_to_file(
        #     y_crops, os.path.join(visu_save_dir, f"{i}_input.png")
        # )


