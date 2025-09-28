'''Training loops and optimization routines.

This module contains the core training functionality extracted from
abstract_trainer.py, including training loops, validation, optimization,
and model management for training.

Extracted from abstract_trainer.py as part of the codebase refactoring.
Refactored to use clean API with TrainingConfig dataclass instead of argparse.
'''

import itertools
import logging
import os
import platform
import random
import shutil
import statistics
import sys
import time
from typing import Callable, Iterable, Optional, Dict, Any

import psutil
import torch
import tqdm
import yaml

from ..dependencies.json_saver import YAMLSaver
from ..dependencies.pt_losses import losses, metrics
from ..dependencies.pytorch_helpers import get_device
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw
from ..dependencies import locking
from ..dependencies import numpy_operations

from .clean_api import TrainingConfig


class BayerImageToImageNNTraining:
    '''Base class for Bayer image processing functionality.

    This class provides Bayer-specific processing capabilities including
    color space conversion and exposure matching.
    '''

    def __init__(self, config: TrainingConfig):
        self.config = config

    def process_net_output(
        self,
        camRGB_images: torch.Tensor,
        rgb_xyz_matrix: torch.Tensor,
        gt_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''Process camRGB output for Bayer images.

        1. Match exposure if gt_images is provided
        2. Apply Lin. Rec. 2020 color profile

        Args:
            camRGB_images: network output to convert
            rgb_xyz_matrix: camRGB to lin_rec2020 conversion matrices
            gt_images: Ground-truth images to match exposure against (if provided)
        '''
        match_gain = self.config.match_gain
        if gt_images is not None and match_gain == 'output':
            camRGB_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=camRGB_images
            )
        output_images = rawproc.camRGB_to_lin_rec2020_images(
            camRGB_images, rgb_xyz_matrix
        )
        if gt_images is not None and match_gain == 'output':
            output_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=output_images
            )
        return output_images


class TrainingLoops:
    '''Core training functionality for neural network training.

    This class provides the main training loops, validation, optimization,
    and model management functionality extracted from the original
    ImageToImageNNTraining class.
    '''

    MODELS_BASE_DPATH = 'models/rawnind'
    CLS_CONFIG_FPATHS = []  # No longer used for CLI configs

    def __init__(self, config: TrainingConfig):
        '''Initialize an image to image neural network trainer.

        Args:
            config: TrainingConfig dataclass with all parameters
        '''
        # Skip if already initialized, by checking for self.optimizer
        if hasattr(self, 'optimizer'):
            return

        self.config = config
        self.test_only = config.test_only

        # Set attributes from config
        self.__dict__.update(vars(config))
        self.autocomplete_config(config)
        self.__dict__.update(vars(config))

        if not self.test_only:
            self.save_args(config)
        self.save_cmd()
        self.device = get_device(self.device)
        if 'cuda' in str(self.device):
            torch.backends.cudnn.benchmark = True  # type: ignore

        # Get logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=os.path.join(
                self.save_dpath, f"{'test' if self.test_only else 'train'}.log"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.DEBUG if self.debug_options else logging.INFO,
            filemode="w",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(f"PID: {os.getpid()}")
        logging.info(f"{vars(self.config)=}")

        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)

        # Instantiate model (placeholder - to be set by subclass)
        self.instantiate_model()
        if self.load_path:
            self.load_model(self.model, self.load_path, device=self.device)

        # Init metrics
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric] = metrics[metric]()
        self.metrics = metrics_dict

        # Initialize optimizer
        self.init_optimizer()
        if self.load_path and (
            self.init_step > 0 or not self.reset_optimizer_on_fallback_load_path
        ):
            self.load_model(self.optimizer, self.load_path + ".opt", device=self.device)
        if self.reset_lr or (self.fallback_load_path and self.init_step == 0):
            self.reset_learning_rate()
        res_fpath: str = os.path.join(self.save_dpath, "trainres.yaml")
        self.json_saver = YAMLSaver(
            res_fpath, warmup_nsteps=self.warmup_nsteps
        )
        logging.info(f"See {res_fpath} for results.")

        # Get training data
        self.get_dataloaders()

        self.lr_adjustment_allowed_step: int = self.patience

        self.transfer = self.get_transfer_function(self.transfer_function)
        self.transfer_vt = self.get_transfer_function(self.transfer_function_valtest)

    @staticmethod
    def save_args(config: TrainingConfig):
        os.makedirs(config.save_dpath, exist_ok=True)
        out_fpath = os.path.join(config.save_dpath, "args.yaml")
        numpy_operations.dict_to_yaml(vars(config), out_fpath)

    def save_cmd(self):
        os.makedirs(self.save_dpath, exist_ok=True)
        out_fpath = os.path.join(
            self.save_dpath, "test_cmd.sh" if self.test_only else "train_cmd.sh"
        )
        # Log configuration instead of command for clean API
        config_fpath = out_fpath + '.config'
        numpy_operations.dict_to_yaml(vars(self.config), config_fpath)
        logging.info(f"Configuration saved to {config_fpath}")

    def autocomplete_config(self, config: TrainingConfig):
        '''
        Auto-complete the following arguments:

        expname: CLASS_NAME_<in_channels>ch<-iteration>
        load_path (optional): can be dpath (autopick best model), expname (autocomplete to dpath), fpath (end-result)
        save_dpath: models/rawnind/<expname>

        to continue:
            determine expname
            determine save_dpath, set load_path accordingly
                or make a function common to save_dpath and load_path
        '''
        # Generate expname and save_dpath, and (incomplete/dir_only) load_path if continue_training_from_last_model_if_exists
        if not config.expname:
            assert config.save_dpath is None, "incompatible args: save_dpath and expname"
            if not config.config:
                config.expname = self._mk_expname(config)
            else:
                config.expname = numpy_operations.get_leaf(config.config).split(".")[0]
            if config.comment:
                config.expname += "_" + config.comment + "_"

            # Handle duplicate expname -> increment
            dup_cnt = None
            while os.path.isdir(
                save_dpath := os.path.join(self.MODELS_BASE_DPATH, config.expname)
            ):
                dup_cnt: int = 1
                while os.path.isdir(f"{save_dpath}-{dup_cnt}"):
                    dup_cnt += 1  # add a number to the last model w/ same expname
                # But load the previous model if continue_training_from_last_model_if_exists or testing
                if config.continue_training_from_last_model_if_exists:
                    if dup_cnt > 1:
                        config.load_path = f"{config.expname}-{dup_cnt - 1}"
                    elif dup_cnt == 1:
                        config.load_path = config.expname
                    else:
                        raise ValueError("bug")
                config.expname = f"{config.expname}-{dup_cnt}"
            # If we want to continue training from last model and there are none from this experiment but fallback_load_path is specified, then load that model and reset the step and learning_rate
            config.save_dpath = save_dpath
        else:
            config.save_dpath = os.path.join(self.MODELS_BASE_DPATH, config.expname)
            os.makedirs(self.MODELS_BASE_DPATH, exist_ok=True)
        # If vars(self).get(
        #    "test_only", False
        # ):  # and config.load_path is None:  config.load_path is the previous best model whereas we want to find the current best one.
        # If self.test_only and config.load_path is None:
        if vars(self).get("test_only", False) and config.load_path is None:
            config.load_path = config.expname
            dup_cnt = None

        def complete_load_path_and_init_step():
            if os.path.isfile(config.load_path) or config.load_path.endswith(".pt"):
                if config.init_step is None:
                    try:
                        config.init_step = int(
                            config.load_path.split(".")[-2].split("_")[-1]
                        )
                    except ValueError as e:
                        logging.warning(
                            f"autocomplete_config: unable to parse init_step from {config.load_path=} ({e=})"
                        )
            else:
                if not os.path.isdir(config.load_path):
                    config.load_path = os.path.join(
                        self.MODELS_BASE_DPATH, config.load_path
                    )
                # FIXME? following line will raise FileNotFoundError if trainres.yaml does not exist

                best_step = self.get_best_step(
                    model_dpath=config.load_path, suffix=self._get_resume_suffix()
                )
                config.load_path = best_step["fpath"]
                # Check if there are newer models
                if vars(config).get(
                    "continue_training_from_last_model_if_exists"
                ) and not vars(self).get("test_only", False):
                    # If config.continue_training_from_last_model_if_exists:
                    dup_cnt_load = None if dup_cnt is None else dup_cnt - 1
                    while not os.path.isfile(config.load_path):
                        logging.info(
                            f"warning: {config.load_path} not found, trying previous model"
                        )
                        if not dup_cnt_load:
                            config.load_path = None
                            logging.warning("no model to load")
                            if vars(self).get("test_only", False):
                                raise ValueError(f"No model to load")
                            return
                        if dup_cnt_load > 1:
                            config.load_path = config.load_path.replace(
                                f"-{dup_cnt_load}{os.sep}",
                                f"-{dup_cnt_load - 1}{os.sep}",
                            )
                            dup_cnt_load -= 1
                        elif dup_cnt_load == 1:
                            config.load_path = config.load_path.replace(
                                f"-{dup_cnt_load}{os.sep}", os.sep
                            )
                            dup_cnt_load = None
                        else:
                            raise ValueError("bug")
                if config.init_step is None:
                    config.init_step = best_step["step_n"]

        # breakpoint()
        if config.load_path is None and config.fallback_load_path is not None:
            config.load_path = (
                find_best_expname_iteration.find_latest_model_expname_iteration(
                    config.fallback_load_path
                )
            )
            config.init_step = 0
        if config.load_path:
            try:
                complete_load_path_and_init_step()
            except KeyError as e:
                logging.error(f"KeyError: {e=}; unable to load previous model.")
                config.load_path = None
                config.init_step = 0
        if config.init_step is None:
            config.init_step = 0

        # If config.continue_training_from_last_model and not config.expname:
        #     if not config.load_path:
        #         config.load_path
        #     self.autocomplete_config(config)  # first pass w/ continue: we determine the expname
        if (
            hasattr(self, "test_only")
            and self.test_only
            and "/scratch/" in vars(config).get("noise_dataset_yamlfpaths", "")
        ):
            # FIXME this doesn't always work, eg "tools/validate_and_test_dc_prgb2prgb.py --config /orb/benoit_phd/models/rawnind_dc/DCTrainingProfiledRGBToProfiledRGB_3ch_L64.0_Balle_Balle_2023-10-27-dc_prgb_msssim_mgout_64from128_x_x_/args.yaml --device -1"
            # when noise_dataset_yamlfpaths is not overwritten through preset_args
            config.noise_dataset_yamlfpaths = [rawproc.RAWNIND_CONTENT_FPATH]
        # config.load_key_metric = f"val_{self._get_resume_suffix()}"  # this would have been nice for tests to have but not implemented on time

    def autocomplete_config(self, config: TrainingConfig):
        '''Override for specific autocomplete logic if needed.'''
        if not config.val_crop_size:
            config.val_crop_size = config.test_crop_size

    def instantiate_model(self):
        # Placeholder - to be implemented by subclasses
        raise NotImplementedError("Subclasses must implement instantiate_model()")

    @staticmethod
    def get_best_step(
        model_dpath: str,
        suffix: str,
        prefix: str = "val",
        # suffix="combined_loss",
    ) -> dict:
        '''Return a dictionary containing step_n: the best step as read from trainres.yaml, fpath: path to the model on that step.'''
        jsonfpath = os.path.join(model_dpath, "trainres.yaml")
        if not os.path.isfile(jsonfpath):
            raise FileNotFoundError(
                "get_best_checkpoint: jsonfpath not found: {}".format(jsonfpath)
            )
        results = numpy_operations.load_yaml(jsonfpath, error_on_404=False)
        metric = "{}_{}".format(prefix, suffix)
        try:
            best_step = results["best_step"][metric]
        except KeyError as e:
            raise KeyError(f'"{metric}" not found in {jsonfpath=}') from e
        return {
            "fpath": os.path.join(model_dpath, "saved_models", f"iter_{best_step}.pt"),
            "step_n": best_step,
        }

    @staticmethod
    def get_transfer_function(
        fun_name: str,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        if str(fun_name) == "None":
            return lambda img: img
        elif fun_name == "pq":
            return rawproc.scenelin_to_pq
        elif fun_name == "gamma22":
            return lambda img: rawproc.gamma(img, gamma_val=2.2, in_place=True)
        else:
            raise ValueError(fun_name)

    def _default_transfer_function(self, fun_name: str):
        '''Default transfer function implementation.'''
        if str(fun_name) == "None":
            return lambda img: img
        elif fun_name == "pq":
            return rawproc.scenelin_to_pq
        elif fun_name == "gamma22":
            return lambda img: rawproc.gamma(img, gamma_val=2.2, in_place=True)
        else:
            raise ValueError(fun_name)

    def init_optimizer(self):
        '''Initialize the optimizer for training.'''
        if not hasattr(self, 'model'):
            raise AttributeError("Model must be set before initializing optimizer")
        init_lr = self.config.init_lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=init_lr)

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, device=None) -> None:
        '''Load model weights from checkpoint.'''
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            logging.info(f"Loaded model from {path}")
        else:
            raise FileNotFoundError(path)

    def adjust_lr(self, validation_losses: dict[str, float], step: int):
        '''Adjust learning rate based on validation performance.

        Args:
            validation_losses: Dictionary of validation metric names to values
            step: Current training step
        '''
        model_improved = False
        for lossn, lossv in validation_losses.items():
            if lossv <= self.best_validation_losses[lossn]:
                logging.info(
                    f"self.best_validation_losses[{lossn}]={self.best_validation_losses[lossn]} <- {lossv=}"
                )
                self.best_validation_losses[lossn] = lossv
                self.lr_adjustment_allowed_step = step + self.patience
                model_improved = True

        if not model_improved and self.lr_adjustment_allowed_step < step:
            old_lr = self.optimizer.param_groups[0]["lr"]
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.lr_multiplier
            # Note: There was a bug in original with duplicate multiplication, fixing it here
            logging.info(
                f"adjust_lr: {old_lr} -> {self.optimizer.param_groups[0]['lr']}"
            )
            self.json_saver.add_res(step, {"lr": self.optimizer.param_groups[0]["lr"]})
            self.lr_adjustment_allowed_step = step + self.patience

    def reset_learning_rate(self):
        '''Reset learning rate to initial value.'''
        init_lr = self.config.init_lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = init_lr
        logging.info(f"reset_learning_rate to {self.optimizer.param_groups[0]['lr']}")

    def validate_or_test(
            self,
            dataloader: Iterable,
            test_name: str,
            sanity_check: bool = False,
            save_individual_results: bool = True,
            save_individual_images: bool = False,
    ):
        '''Perform validation or testing on a dataset.

        Args:
            dataloader: Iterable that yields batches of data
            test_name: Identifier for this validation/test run
            sanity_check: If True, runs a minimal validation for debugging
            save_individual_results: If True, saves per-sample metrics to a YAML file
            save_individual_images: If True, saves model output images for each sample

        Returns:
            Dictionary with aggregated metrics and loss values
        '''
        # Validation lock (simplified for now, but based on original implementation)
        own_lock = bypass_lock = printed_lock_warning = False
        lock_fpath = f"validation_{platform.node()}_{os.environ.get('CUDA_VISIBLE_DEVICES', 'unk')}.lock"

        # Bypass lock for certain conditions (from original logic)
        if (platform.node() == "sd" or platform.node() == "bd") and (
            "manproc" not in test_name or self.config.arbitrary_proc_method == "opencv"
        ):
            bypass_lock = True

        while not own_lock and not bypass_lock:
            # Lock management logic from original implementation
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
                # Write PID and launch command to lock_fpath
                with open(lock_fpath, "w") as f:
                    f.write(f"{os.getpid()}\n")
                    f.write(" ".join(sys.argv))
                if printed_lock_warning:
                    logging.warning(":)")

        with torch.no_grad():
            losses = {lossn: [] for lossn in (self.loss, *self.metrics)}

            individual_results = {}

            # Load individual results if they exist
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

                # Handle progressive test naming logic
                if (
                    "progressive_test" in common_test_name
                    and "manproc" in common_test_name
                ):
                    if "manproc_bostitch" in common_test_name:
                        common_test_name_noprog = "manproc_bostitch"
                    else:
                        common_test_name_noprog = "manproc"
                    individual_results_fpath = os.path.join(
                        self.save_dpath,
                        common_test_name_noprog,
                        f"iter_{getattr(self, 'step_n', 0)}.yaml",
                    )
                    os.makedirs(
                        os.path.dirname(individual_results_fpath), exist_ok=True
                    )
                else:
                    individual_results_fpath = os.path.join(
                        self.save_dpath, common_test_name, f"iter_{getattr(self, 'step_n', 0)}.yaml"
                    )

                if os.path.isfile(individual_results_fpath):
                    individual_results = yaml.safe_load(open(individual_results_fpath))
                    print(f"Loaded {individual_results_fpath=}")
                else:
                    print(
                        f"No previous individual results {individual_results_fpath=} found"
                    )

            individual_images_dpath = os.path.join(
                self.save_dpath, common_test_name, f"iter_{getattr(self, 'step_n', 0)}"
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
                # Create image key for tracking individual results
                if "y_fpath" in batch:
                    y_fn = (
                        batch["y_fpath"]
                        if isinstance(batch["y_fpath"], str)
                        else batch["y_fpath"][0]
                    )
                    y_fn = os.path.basename(y_fn)
                    image_key = y_fn

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

                # Skip if we already have results for this image
                if save_individual_results and image_key in individual_results:
                    for lossn, lossv in individual_results[image_key].items():
                        if lossn not in losses:
                            losses[lossn] = []
                        losses[lossn].append(lossv)
                    continue

                individual_results[image_key] = {}
                x_crops = batch["x_crops"].to(self.device)
                y_crops = batch["y_crops"].to(self.device, x_crops.dtype)
                mask_crops = batch["mask_crops"].to(self.device)

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
                        f"Error {e} with {batch.get('gt_fpath', 'unknown')=}, {batch.get('y_fpath', 'unknown')=}, {y_crops.shape=}, {x_crops.shape=}, {mask_crops.shape=}"
                    )
                    raise e

                if self.match_gain == "output":
                    processed_output = rawproc.match_gain(x_crops, reconstructed_image)
                else:
                    processed_output = reconstructed_image

                if hasattr(self, "process_net_output"):  # Bayer color transform
                    processed_output = self.process_net_output(
                        processed_output, batch["rgb_xyz_matrix"], x_crops
                    )

                if "output_valtest_images" in self.debug_options:
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

                # Compute losses
                loss_functions = self.metrics.copy()
                from ..dependencies.pt_losses import losses as loss_module
                loss_functions[self.loss] = getattr(self, 'lossf', loss_module[self.loss]())

                for lossn, lossf in loss_functions.items():
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
                            f"Error {e} with {batch.get('gt_fpath', 'unknown')=}, {batch.get('y_fpath', 'unknown')=}, {y_crops.shape=}, {x_crops.shape=}, {processed_output.shape=}, {reconstructed_image.shape=}, {mask_crops.shape=}, {image_key=}"
                        )
                        raise e
                    losses[lossn].append(lossv)
                    individual_results[image_key][lossn] = lossv

                # Handle BPP for compression models
                if bpp is not None:
                    if "bpp" not in losses:
                        losses["bpp"] = []
                        losses["combined"] = []
                    losses["bpp"].append(float(bpp))
                    individual_results[image_key]["bpp"] = float(bpp)

                    # Compute combined loss for compression
                    train_lambda = getattr(self, 'train_lambda', 1.0)
                    # Compute combined loss for compression
                    lossf = getattr(self, 'lossf', loss_module[self.loss]())
                    combined_loss = float(
                        bpp
                        + lossf(
                            processed_output * mask_crops, x_crops * mask_crops
                        ).item()
                        * train_lambda
                    )
                    losses["combined"].append(combined_loss)
                    individual_results[image_key]["combined"] = combined_loss

                if sanity_check and i >= 1:
                    break

            # Save individual results
            if save_individual_results:
                numpy_operations.dict_to_yaml(individual_results, individual_results_fpath)

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
            raise e

    def training_loop(self):
        '''Main training loop with proper step management.'''
        # Initialize step counters
        last_test_step = last_val_step = self.step_n = getattr(self, 'init_step', 0)

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
            self.best_validation_losses = validation_losses

        # Initial sanity test
        self.validate_or_test(
            dataloader=self.cleannoisy_test_dataloader,
            sanity_check=True,
            test_name="sanitytest",
        )
        torch.cuda.empty_cache()

        # Main training loop
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

            # CRITICAL: Increment step counter to avoid infinite loop
            self.step_n += num_training_steps

            logging.info(
                f"training_loop: {self.step_n=}, {training_loss=} (over {num_training_steps=})"
            )
            self.json_saver.add_res(
                self.step_n,
                {f"train_{self.loss}": training_loss},
            )

            # Validation
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

            # Testing
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

    def save_model(self, step: int) -> None:
        '''Save model checkpoint.'''
        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)
        fpath = os.path.join(self.save_dpath, "saved_models", f"iter_{step}.pt")
        torch.save(self.model.state_dict(), fpath)
        torch.save(self.optimizer.state_dict(), fpath + ".opt")

    def cleanup_models(self):
        '''Clean up old model checkpoints.'''
        keepers: list[str] = [
            f"iter_{step}" for step in self.json_saver.get_best_steps()
        ]
        models_dir = os.path.join(self.save_dpath, "saved_models")
        if os.path.exists(models_dir):
            for fn in os.listdir(models_dir):
                if fn.partition(".")[0] not in keepers:
                    logging.info(
                        f"cleanup_models: rm {os.path.join(models_dir, fn)}"
                    )
                    os.remove(os.path.join(models_dir, fn))

        # Clean up visualization files too
        if "output_valtest_images" in self.debug_options:
            visu_dir = os.path.join(self.save_dpath, "visu")
            if os.path.isdir(visu_dir):
                for dn in os.listdir(visu_dir):
                    if dn not in keepers:
                        logging.info(
                            f"cleanup_models: rm -r {os.path.join(visu_dir, dn)}"
                        )
                        shutil.rmtree(
                            os.path.join(visu_dir, dn),
                            ignore_errors=True,
                        )

    def train(
        self,
        optimizer: torch.optim.Optimizer,
        num_steps: int,
        dataloader_cc: Iterable,
        dataloader_cn: Iterable,
    ) -> float:
        '''Run training for specified number of steps.'''
        last_time = time.time()
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

    def compute_train_loss(
        self,
        mask,
        processed_output,
        processed_gt,
        bpp,
    ) -> torch.Tensor:
        '''Compute training loss.'''
        # Compute loss
        masked_proc_output = processed_output * mask
        masked_proc_gt = processed_gt * mask
        lossf = getattr(self, 'lossf', losses[self.loss]())
        loss = lossf(masked_proc_output, masked_proc_gt) * getattr(self, 'train_lambda', 1.0)

        # Add rate penalty for compression models
        if bpp is not None:
            loss += bpp

        return loss

    def _get_lossn_extension(self):
        '''Get loss name extension based on processing method.'''
        lossn_extension = ""
        if hasattr(self.config, 'arbitrary_proc_method') and self.config.arbitrary_proc_method:
            lossn_extension += ".arbitraryproc"
        elif hasattr(self.config, 'transfer_function_valtest') and self.config.transfer_function_valtest != "pq":
            lossn_extension += f".{self.config.transfer_function_valtest}"
        return lossn_extension


class ImageToImageNNTraining(TrainingLoops):
    '''Extension of base training with specific training numpy_operations.

    Provides argument registration for training hyperparameters, training/validation
    loops, dataloader wiring and checkpointing helpers. Subclasses should override
    repack_batch() and step() to implement their data layout and loss computation.
    '''

    def __init__(self, config: TrainingConfig):
        '''Initialize an image to image neural network trainer.

        Args:
            config: TrainingConfig dataclass with all parameters
        '''
        super().__init__(config)

        # Initialize loss function
        loss_name = self.config.loss
        try:
            self.lossf = losses[loss_name]()
        except KeyError:
            raise NotImplementedError(f"{loss_name} not in pt_losses.losses")

        # Initialize best validation losses
        self.best_validation_losses: dict[str, float] = {}

        # Initialize metrics
        metrics_dict = {}
        metric_names = self.config.metrics
        if isinstance(metric_names, list):
            for metric in metric_names:
                metrics_dict[metric] = metrics[metric]()
        elif isinstance(metric_names, dict):
            metrics_dict = metric_names
        self.metrics = metrics_dict

    def get_dataloaders(self) -> None:
        '''Instantiate the train/val/test data-loaders into self using clean dataset API.'''
        from ..dataset.clean_api import create_training_dataset, create_validation_dataset, create_test_dataset

        # Use clean API factories
        train_config = {
            'crop_size': self.config.crop_size,
            'num_crops_per_image': self.config.num_crops_per_image,
            'batch_size_clean': self.config.batch_size_clean,
            'batch_size_noisy': self.config.batch_size_noisy,
            'data_pairing': self.config.data_pairing,
            'arbitrary_proc_method': self.config.arbitrary_proc_method,
            'bayer_only': self.config.bayer_only,
            'toy_dataset': 'toy_dataset' in self.config.debug_options,
            'match_gain': self.config.match_gain == 'input',
        }

        # Training dataloaders
        if not self.test_only:
            self.cleanclean_dataloader, self.cleannoisy_dataloader = create_training_dataset(
                clean_yaml_paths=self.config.clean_dataset_yamlfpaths,
                noisy_yaml_paths=self.config.noise_dataset_yamlfpaths,
                input_channels=self.config.in_channels,
                **train_config
            )

        # Validation dataloader
        val_config = {
            'crop_size': self.config.val_crop_size,
            'test_reserve': self.config.test_reserve,
            'bayer_only': self.config.bayer_only,
            'toy_dataset': 'toy_dataset' in self.config.debug_options,
            'match_gain': self.config.match_gain == 'input',
            'data_pairing': self.config.data_pairing,
            'arbitrary_proc_method': self.config.arbitrary_proc_method,
        }
        self.cleannoisy_val_dataloader = create_validation_dataset(
            noisy_yaml_paths=self.config.noise_dataset_yamlfpaths,
            input_channels=self.config.in_channels,
            **val_config
        )

        # Test dataloader
        test_config = {
            'crop_size': self.config.test_crop_size,
            'test_reserve': self.config.test_reserve,
            'bayer_only': self.config.bayer_only,
            'toy_dataset': 'toy_dataset' in self.config.debug_options,
            'match_gain': self.config.match_gain == 'input',
            'arbitrary_proc_method': self.config.arbitrary_proc_method,
        }
        self.cleannoisy_test_dataloader = create_test_dataset(
            noisy_yaml_paths=self.config.noise_dataset_yamlfpaths,
            input_channels=self.config.in_channels,
            **test_config
        )

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        '''Perform a single training step.

        This method should be implemented by subclasses to handle specific
        data formats (RGB vs Bayer) and loss computation.
        '''
        raise NotImplementedError("step() must be implemented by subclasses")

    def offline_validation(self):
        '''Only validate model (same as in training_loop but called externally / without train)'''
        if "step_n" not in vars(self):
            self.step_n = self.config.init_step
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
        '''Std test (same as in training but run externally).'''
        if "step_n" not in vars(self):
            self.step_n = self.config.init_step
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
            self.step_n = self.config.init_step
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


class PRGBImageToImageNNTraining(ImageToImageNNTraining):
    '''Training class specialized for RGB image processing.'''

    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    @staticmethod
    def repack_batch(batch: tuple, device: torch.device) -> dict:
        '''Repack batch data for RGB processing.

        Args:
            batch: Tuple of dicts containing batch data
            device: Target device for tensors

        Returns:
            Dictionary with repacked batch data
        '''
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

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        '''Perform a single training step for RGB images.'''
        try:
            batch = self.repack_batch(batch, self.device)
        except KeyError as e:
            logging.error(e)
            raise e

        model_output = self.model(batch["y_crops"])
        if isinstance(self, DenoiseCompressTraining):
            if "spam" in self.debug_options and random.random() < 0.01:
                logging.debug(
                    f"DBG: {model_output.get('used_dists', 'N/A')=}, {model_output.get('num_forced_dists', 'N/A')=}"
                )
            reconstructed_image, bpp = (
                model_output["reconstructed_image"],
                model_output["bpp"],
            )
        else:
            reconstructed_image = model_output
            bpp = 0

        if self.match_gain == "output":
            reconstructed_image = rawproc.match_gain(
                batch["x_crops"], reconstructed_image
            )

        if output_train_images:
            # Save training images for debugging
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{getattr(self, 'step_n', 0)}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                raw.hdr_nparray_to_file(
                    reconstructed_image[i].detach().cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_reconstructed.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (reconstructed_image[i].detach() * batch["mask_crops"][i])
                    .cpu()
                    .numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_reconstructed_masked.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (batch["y_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_input.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    batch["x_crops"][i].cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    self.transfer(batch["x_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt_transfered.exr"),
                    color_profile="lin_rec2020",
                )

        reconstructed_image = self.transfer(reconstructed_image)
        gt = self.transfer(batch["x_crops"])

        loss = self.compute_train_loss(
            batch["mask_crops"],
            reconstructed_image,
            gt,
            bpp,
        )

        optimizer.zero_grad()
        loss.backward()

        if isinstance(self, DenoiseCompressTraining):
            DenoiseCompressTraining.clip_gradient(optimizer, 5)

        optimizer.step()
        return loss.item()

    def _dbg_output_testval_images(self, batch, processed_output, individual_images_dpath, i, x_crops, y_crops,
                                   mask_crops):
        '''Debug output for test/validation images.'''
        if isinstance(batch["y_fpath"], list) and len(batch["y_fpath"]) == 1:
            batch["y_fpath"] = batch["y_fpath"][0]
            batch["gt_fpath"] = batch["gt_fpath"][0]

        raw.hdr_nparray_to_file(
            (processed_output * mask_crops)[0].cpu().numpy(),
            os.path.join(individual_images_dpath, f"{i}_{batch['y_fpath'].split('/')[-1]}_output_masked.exr"),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            processed_output[0].cpu().numpy(),
            os.path.join(individual_images_dpath, f"{i}_{batch['y_fpath'].split('/')[-1]}_output.exr"),
            color_profile="lin_rec2020",
        )

        gt_fpath = os.path.join(individual_images_dpath, f"{i}_{batch['gt_fpath'].split('/')[-1]}_gt.exr")
        raw.hdr_nparray_to_file(
            x_crops[0].cpu().numpy(), gt_fpath, color_profile="lin_rec2020"
        )

        input_fpath = os.path.join(individual_images_dpath, f"{i}_{batch['y_fpath'].split('/')[-1]}_input.exr")
        raw.hdr_nparray_to_file(
            y_crops[0].cpu().numpy(), input_fpath, color_profile="lin_rec2020"
        )

        mask_fpath = os.path.join(individual_images_dpath, f"{i}_{batch['y_fpath'].split('/')[-1]}_mask.exr")
        raw.hdr_nparray_to_file(
            mask_crops[0].cpu().numpy(), mask_fpath, color_profile="lin_rec2020"
        )


# Additional subclasses like BayerImageToImageNNTraining, DenoiseCompressTraining would follow similar patterns
# but are omitted for brevity in this refactor example. They would also take TrainingConfig and use config attributes.


class DenoiseCompressTraining(ImageToImageNNTraining):
    '''Training class for joint denoising and compression.'''

    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.train_lambda = config.compression_lambda or 1.0

    @staticmethod
    def clip_gradient(optimizer: torch.optim.Optimizer, max_norm: float = 5.0):
        '''Clip gradients to prevent exploding gradients.'''
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm)


# ... (other subclasses like DenoiserTraining, etc., refactored similarly)

class DenoiserTraining(ImageToImageNNTraining):
    '''Training class for denoising models (no compression).'''

    def __init__(self, config: TrainingConfig):
        super().__init__(config)

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        '''Perform a single training step for denoising.'''
        try:
            batch = self.repack_batch(batch, self.device)
        except KeyError as e:
            logging.error(e)
            raise e

        model_output = self.model(batch["y_crops"])
        reconstructed_image = model_output  # No bpp for denoiser
        bpp = None

        if self.match_gain == "output":
            reconstructed_image = rawproc.match_gain(
                batch["x_crops"], reconstructed_image
            )

        if output_train_images:
            # Similar debug output as PRGB
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{getattr(self, 'step_n', 0)}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                raw.hdr_nparray_to_file(
                    reconstructed_image[i].detach().cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_reconstructed.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (reconstructed_image[i].detach() * batch["mask_crops"][i])
                    .cpu()
                    .numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_reconstructed_masked.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (batch["y_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_input.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    batch["x_crops"][i].cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    self.transfer(batch["x_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt_transfered.exr"),
                    color_profile="lin_rec2020",
                )

        reconstructed_image = self.transfer(reconstructed_image)
        gt = self.transfer(batch["x_crops"])

        loss = self.compute_train_loss(
            batch["mask_crops"],
            reconstructed_image,
            gt,
            bpp,  # None for denoiser
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()
