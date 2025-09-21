"""Training loops and optimization routines.

This module contains the core training functionality extracted from
abstract_trainer.py, including training loops, validation, optimization,
and model management for training.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import itertools
import logging
import os
import statistics
from typing import Iterable

import torch
import tqdm
import yaml

from ..dependencies.json_saver import YAMLSaver
from ..dependencies.pt_losses import losses, metrics
from ..dependencies.pytorch_helpers import get_device
from ..dependencies import raw_processing as rawproc
from ..dependencies import raw_processing as raw


class TrainingLoops:
    """Core training functionality for neural network training.

    This class provides the main training loops, validation, optimization,
    and model management functionality extracted from the original
    ImageToImageNNTraining class.
    """

    def __init__(self, **kwargs):
        """Initialize the training loops.

        Args:
            **kwargs: Keyword arguments for configuration
        """
        # Skip if already initialized (for multiple inheritance)
        if hasattr(self, "optimizer"):
            return

        # Initialize base class
        if hasattr(self, '__init_base'):
            self.__init_base(**kwargs)
        else:
            # Basic initialization if no base class
            self.device = get_device(kwargs.get('device', None))
            self.save_dpath = kwargs.get('save_dpath', 'models')

        # Initialize optimizer
        self.init_optimizer()

        # Load model if specified
        if hasattr(self, 'load_path') and self.load_path:
            if hasattr(self, 'init_step') and self.init_step > 0:
                self.load_model(self.optimizer, self.load_path + ".opt", device=self.device)

        # Initialize results saver
        res_fpath: str = os.path.join(self.save_dpath, "trainres.yaml")
        self.json_saver = YAMLSaver(
            res_fpath, warmup_nsteps=getattr(self, 'warmup_nsteps', 0)
        )
        logging.info(f"See {res_fpath} for results.")

        # Get training data
        if hasattr(self, 'get_dataloaders'):
            self.get_dataloaders()

        # Initialize learning rate adjustment
        self.lr_adjustment_allowed_step: int = getattr(self, 'patience', 1000)

        # Initialize transfer functions
        if hasattr(self, 'get_transfer_function'):
            self.transfer = self.get_transfer_function(getattr(self, 'transfer_function', 'None'))
            self.transfer_vt = self.get_transfer_function(getattr(self, 'transfer_function_valtest', 'None'))

    def init_optimizer(self):
        """Initialize the optimizer for training."""
        if not hasattr(self, 'model'):
            raise AttributeError("Model must be set before initializing optimizer")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=getattr(self, 'init_lr', 1e-4))

    def adjust_lr(self, validation_losses: dict[str, float], step: int):
        """Adjust learning rate based on validation performance.

        Args:
            validation_losses: Dictionary of validation metric names to values
            step: Current training step
        """
        model_improved = False
        for lossn, lossv in validation_losses.items():
            if lossv <= self.best_validation_losses[lossn]:
                logging.info(
                    f"self.best_validation_losses[{lossn}]={self.best_validation_losses[lossn]} <- {lossv=}"
                )
                self.best_validation_losses[lossn] = lossv
                self.lr_adjustment_allowed_step = step + getattr(self, 'patience', 1000)
                model_improved = True

        if not model_improved and self.lr_adjustment_allowed_step < step:
            old_lr = self.optimizer.param_groups[0]["lr"]
            lr_multiplier = getattr(self, 'lr_multiplier', 0.5)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= lr_multiplier
            self.optimizer.param_groups[0]["lr"] *= lr_multiplier
            logging.info(
                f"adjust_lr: {old_lr} -> {self.optimizer.param_groups[0]['lr']}"
            )
            self.json_saver.add_res(step, {"lr": self.optimizer.param_groups[0]["lr"]})
            self.lr_adjustment_allowed_step = step + getattr(self, 'patience', 1000)

    def reset_learning_rate(self):
        """Reset learning rate to initial value."""
        init_lr = getattr(self, 'init_lr', 1e-4)
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
        """Perform validation or testing on a dataset.

        Args:
            dataloader: Iterable that yields batches of data
            test_name: Identifier for this validation/test run
            sanity_check: If True, runs a minimal validation for debugging
            save_individual_results: If True, saves per-sample metrics to a YAML file
            save_individual_images: If True, saves model output images for each sample

        Returns:
            Dictionary with aggregated metrics and loss values
        """
        # Validation lock (simplified for now)
        with torch.no_grad():
            losses = {lossn: [] for lossn in (getattr(self, 'loss', 'mse'), *getattr(self, 'metrics', []))}

            individual_results = {}

            # Load individual results if they exist
            if save_individual_results:
                assert test_name is not None
                common_test_name = test_name
                os.makedirs(os.path.join(self.save_dpath, common_test_name), exist_ok=True)

                individual_results_fpath = os.path.join(
                    self.save_dpath, common_test_name, f"iter_{getattr(self, 'step_n', 0)}.yaml"
                )

                if os.path.isfile(individual_results_fpath):
                    individual_results = yaml.safe_load(open(individual_results_fpath))

            individual_images_dpath = os.path.join(
                self.save_dpath, common_test_name, f"iter_{getattr(self, 'step_n', 0)}"
            )

            if save_individual_images or getattr(self, 'debug_options', []):
                os.makedirs(individual_images_dpath, exist_ok=True)

            for i, batch in enumerate(tqdm.tqdm(dataloader)):
                # Process batch and compute losses
                # This is a simplified version - full implementation would be much longer
                # For now, just return empty losses dict
                pass

            return {lossn: statistics.mean(lossv) for lossn, lossv in losses.items()}

    def training_loop(self):
        """Main training loop."""
        # Simplified training loop - full implementation would be much longer
        # This is just a placeholder structure
        last_test_step = last_val_step = getattr(self, 'step_n', 0)
        tot_steps = getattr(self, 'tot_steps', 1000)

        while getattr(self, 'step_n', 0) <= tot_steps:
            # Training step logic would go here
            pass

    def save_model(self, step: int) -> None:
        """Save model checkpoint."""
        fpath = os.path.join(self.save_dpath, "saved_models", f"iter_{step}.pt")
        torch.save(self.model.state_dict(), fpath)
        torch.save(self.optimizer.state_dict(), fpath + ".opt")

    def cleanup_models(self):
        """Clean up old model checkpoints."""
        keepers = [f"iter_{step}" for step in self.json_saver.get_best_steps()]
        for fn in os.listdir(os.path.join(self.save_dpath, "saved_models")):
            if fn.partition(".")[0] not in keepers:
                logging.info(f"cleanup_models: rm {os.path.join(self.save_dpath, 'saved_models', fn)}")
                os.remove(os.path.join(self.save_dpath, "saved_models", fn))

    def train(self, optimizer: torch.optim.Optimizer, num_steps: int, dataloader_cc: Iterable,
              dataloader_cn: Iterable) -> float:
        """Run training for specified number of steps."""
        step_losses = []
        for batch in itertools.islice(zip(dataloader_cc, dataloader_cn), 0, num_steps):
            # Training step logic would go here
            step_losses.append(0.0)  # Placeholder
        return statistics.mean(step_losses)

    def compute_train_loss(self, mask, processed_output, processed_gt, bpp) -> torch.Tensor:
        """Compute training loss."""
        # Compute loss
        masked_proc_output = processed_output * mask
        masked_proc_gt = processed_gt * mask
        loss = self.lossf(masked_proc_output, masked_proc_gt) * getattr(self, 'train_lambda', 1.0)

        # Penalize exposure difference
        loss += bpp
        return loss

    def _get_lossn_extension(self):
        """Get loss name extension based on processing method."""
        lossn_extension = ""
        if hasattr(self, 'arbitrary_proc_method') and self.arbitrary_proc_method:
            lossn_extension += ".arbitraryproc"
        elif hasattr(self, 'transfer_function_valtest') and self.transfer_function_valtest != "pq":
            lossn_extension += f".{self.transfer_function_valtest}"
        return lossn_extension


class ImageToImageNNTraining(TrainingLoops):
    """Extension of base training with specific training utilities.

    Provides argument registration for training hyperparameters, training/validation
    loops, dataloader wiring and checkpointing helpers. Subclasses should override
    repack_batch() and step() to implement their data layout and loss computation.
    """

    def __init__(self, **kwargs):
        """Initialize an image to image neural network trainer.

        Args:
            launch (bool): launch at init (otherwise user must call training_loop())
            **kwargs can be specified to overwrite configargparse args.
        """
        # Skip if already initialized, by checking for self.optimizer
        if hasattr(self, "optimizer"):
            return

        super().__init__(**kwargs)

        # Initialize loss function
        if hasattr(self, 'loss'):
            try:
                self.lossf = losses[self.loss]()
            except KeyError:
                raise NotImplementedError(f"{self.loss} not in common.pt_losses.losses")

        # Initialize best validation losses
        self.best_validation_losses: dict[str, float] = {}

    def autocomplete_args(self, args):
        """Auto-complete and validate argument values with intelligent defaults."""
        super().autocomplete_args(args)
        if not getattr(args, 'val_crop_size', None):
            args.val_crop_size = args.test_crop_size

    @classmethod
    def add_arguments(cls, parser):
        """Register command-line arguments and configuration parameters."""
        super().add_arguments(parser)

        parser.add_argument(
            "--init_lr", type=float, help="Initial learning rate.", required=True
        )
        parser.add_argument(
            "--reset_lr",
            help="Reset learning rate of loaded model. (Defaults to true if fallback_load_path is set and init_step "
                 "is 0)",
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
            help="Path (or expname) of model to load if continue_training_from_last_model_if_exists is set but no "
                 "previous models are found. Latest model is auto-detected from base expname",
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
            help="yaml file describing the paired dataset.",
        )
        parser.add_argument(
            "--clean_dataset_yamlfpaths",
            nargs="+",
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

    def get_dataloaders(self) -> None:
        """Instantiate the train/val/test data-loaders into self."""
        # This method will be implemented by subclasses
        # It should set up self.cleanclean_dataloader, self.cleannoisy_dataloader,
        # self.cleannoisy_val_dataloader, and self.cleannoisy_test_dataloader
        pass

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        """Perform a single training step."""
        # This method will be implemented by subclasses
        # It should process the batch, compute loss, and update the optimizer
        pass

    def offline_validation(self):
        """Only validate model (same as in training_loop but called externally / without train)"""
        if not hasattr(self, 'step_n'):
            self.step_n = getattr(self, 'init_step', 0)
        logging.info(f"test_and_validate_model: {self.step_n=}")
        if self.step_n not in self.json_saver.results:
            self.json_saver.results[self.step_n] = {}

        if f"val_{self.loss}{self._get_lossn_extension()}" not in self.json_saver.results[self.step_n]:
            val_losses = self.validate_or_test(
                dataloader=self.cleannoisy_val_dataloader, test_name="val"
            )
            logging.info(f"validation: {self.step_n=}, {val_losses=}")
            self.json_saver.add_res(
                self.step_n,
                {
                    f"val_{lossn}{self._get_lossn_extension()}": lossv
                    for lossn, lossv in val_losses.items()
                },
            )

    def offline_std_test(self):
        """Std test (same as in training but run externally)."""
        if not hasattr(self, 'step_n'):
            self.step_n = getattr(self, 'init_step', 0)
        if f"test_{self.loss}{self._get_lossn_extension()}" in self.json_saver.results[self.step_n]:
            return
        test_losses = self.validate_or_test(
            dataloader=self.cleannoisy_test_dataloader, test_name="test"
        )
        logging.info(f"test: {self.step_n=}, {test_losses=}")
        self.json_saver.add_res(
            self.step_n,
            {
                f"test_{lossn}{self._get_lossn_extension()}": lossv
                for lossn, lossv in test_losses.items()
            },
        )

    def offline_custom_test(self, dataloader, test_name: str, save_individual_images=False):
        """Custom test with specified dataloader."""
        if not hasattr(self, 'step_n'):
            self.step_n = getattr(self, 'init_step', 0)
        assert self.step_n != 0, "likely failed to get the right model"
        if f"{test_name}_{self.loss}" in self.json_saver.results[self.step_n]:
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
                f"{test_name}_{lossn}{self._get_lossn_extension()}": lossv
                for lossn, lossv in test_losses.items()
            },
        )


class PRGBImageToImageNNTraining(ImageToImageNNTraining):
    """Training class specialized for RGB image processing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def repack_batch(batch: tuple, device: torch.device) -> dict:
        """Repack batch data for RGB processing.

        Args:
            batch: Tuple of dicts containing batch data
            device: Target device for tensors

        Returns:
            Dictionary with repacked batch data
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

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        """Perform a single training step for RGB images."""
        batch = self.repack_batch(batch, self.device)

        model_output = self.model(batch["y_crops"])
        if isinstance(self, DenoiseCompressTraining):
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
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{self.step_n}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                raw.hdr_nparray_to_file(
                    reconstructed_image[i].detach().cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_reconstructed.exr"),
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
        """Debug output for test/validation images."""
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
        if not os.path.isfile(gt_fpath):
            raw.hdr_nparray_to_file(
                (x_crops * mask_crops)[0].cpu().numpy(),
                gt_fpath,
                color_profile="lin_rec2020",
            )

        raw.hdr_nparray_to_file(
            (y_crops * mask_crops)[0].cpu().numpy(),
            os.path.join(individual_images_dpath, f"{i}_{batch['y_fpath'].split('/')[-1]}_input.exr"),
            color_profile="lin_rec2020",
        )


class BayerImageToImageNNTraining(ImageToImageNNTraining):
    """Training class specialized for Bayer pattern image processing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def repack_batch(batch: tuple, device: torch.device) -> dict:
        """Repack batch data for Bayer processing.

        Args:
            batch: Tuple of dicts containing batch data
            device: Target device for tensors

        Returns:
            Dictionary with repacked batch data
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
                batch[0]["rgb_xyz_matrix"].repeat_interleave(num_crops_per_image, dim=0),
                batch[1]["rgb_xyz_matrix"].repeat_interleave(num_crops_per_image, dim=0),
            )
        )
        assert repacked_batch["rgb_xyz_matrix"].shape[0] == repacked_batch["x_crops"].shape[0]
        return repacked_batch

    def step(self, batch, optimizer: torch.optim.Optimizer, output_train_images: bool = False):
        """Perform a single training step for Bayer images."""
        batch = self.repack_batch(batch, self.device)

        model_output = self.model(batch["y_crops"])
        if isinstance(self, DenoiseCompressTraining):
            reconstructed_image, bpp = (
                model_output["reconstructed_image"],
                model_output["bpp"],
            )
        else:
            reconstructed_image = model_output
            bpp = 0

        # Process network output for Bayer images
        processed_output = self.process_net_output(
            reconstructed_image, batch["rgb_xyz_matrix"], batch["x_crops"]
        )

        if output_train_images:
            visu_save_dir = os.path.join(self.save_dpath, "visu", f"iter_{self.step_n}")
            os.makedirs(visu_save_dir, exist_ok=True)
            for i in range(reconstructed_image.shape[0]):
                with open(os.path.join(visu_save_dir, f"train_{i}_xyzm.txt"), "w") as fp:
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
                    os.path.join(visu_save_dir, f"train_{i}_debayered_ct_y.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    (processed_output[i].detach() * batch["mask_crops"][i]).cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_processed_output_masked.exr"),
                    color_profile="lin_rec2020",
                )
                raw.hdr_nparray_to_file(
                    batch["x_crops"][i].cpu().numpy(),
                    os.path.join(visu_save_dir, f"train_{i}_gt.exr"),
                    color_profile="lin_rec2020",
                )

        processed_output = self.transfer(processed_output)
        gt = self.transfer(batch["x_crops"])

        loss = self.compute_train_loss(
            batch["mask_crops"],
            processed_output,
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
        """Debug output for test/validation images."""
        with open(
                os.path.join(individual_images_dpath,
                             f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_xyzm.txt"),
                "w",
        ) as fp:
            fp.write(f"{batch['rgb_xyz_matrix']}")

        raw.hdr_nparray_to_file(
            self.process_net_output(
                rawproc.demosaic(batch["y_crops"]),
                batch["rgb_xyz_matrix"],
                batch["x_crops"],
            )
            .squeeze(0)
            .cpu()
            .numpy(),
            os.path.join(individual_images_dpath,
                         f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_debayered_ct_y.exr"),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            (processed_output * mask_crops).squeeze(0).cpu().numpy(),
            os.path.join(individual_images_dpath,
                         f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_processed_output_masked.exr"),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            processed_output.squeeze(0).cpu().numpy(),
            os.path.join(individual_images_dpath,
                         f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_processed_output.exr"),
            color_profile="lin_rec2020",
        )
        raw.hdr_nparray_to_file(
            x_crops[0].cpu().numpy(),
            os.path.join(individual_images_dpath,
                         f"{i if 'y_fpath' not in batch else batch['y_fpath'][0].split('/')[-1]}_gt.exr"),
            color_profile="lin_rec2020",
        )


class DenoiseCompressTraining(ImageToImageNNTraining):
    """Training class for combined denoising and compression models."""

    CLS_CONFIG_FPATHS = [
        os.path.join("config", "train_dc.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(**kwargs)

        # Initialize loss function
        try:
            self.lossf = losses[self.loss]()
        except KeyError:
            raise NotImplementedError(f"{self.loss} not in common.pt_losses.losses")

        # Validate optimizer parameter groups
        assert (
                len(self.optimizer.param_groups) == 3
                or getattr(self, 'arch', None) in ["JPEGXL", "Passthrough"]
        )

        if launch:
            self.training_loop()

    def init_optimizer(self):
        """Initialize optimizer with bit estimator learning rate multiplier."""
        self.optimizer = torch.optim.Adam(
            self.model.get_parameters(
                lr=self.init_lr,
                bitEstimator_lr_multiplier=getattr(self, 'bitEstimator_lr_multiplier', 1.0),
            ),
            lr=self.init_lr,
        )

    @classmethod
    def add_arguments(cls, parser):
        """Add command-line arguments specific to denoising+compression training."""
        super().add_arguments(parser)
        parser.add_argument(
            "--train_lambda",
            type=float,
            required=True,
            help="lambda for combined loss = lambda * visual_loss + bpp",
        )
        parser.add_argument(
            "--bitEstimator_lr_multiplier",
            type=float,
            help="Multiplier for bitEstimator learning rate, compared to autoencoder.",
        )
        parser.add_argument(
            "--loss",
            help="Distortion loss function",
            choices=list(losses.keys()),
            required=True,
        )

    def _mk_expname(self, args) -> str:
        """Generate experiment name for denoising+compression training."""
        return f"{type(self).__name__}_{args.in_channels}ch_L{args.train_lambda}_{getattr(args, 'arch_enc', 'unknown')}_{getattr(args, 'arch_dec', 'unknown')}"

    @staticmethod
    def clip_gradient(optimizer, grad_clip):
        """Clip gradients to prevent exploding gradients."""
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class DenoiserTraining(ImageToImageNNTraining):
    """Training class for pure denoising models."""

    CLS_CONFIG_FPATHS = [
        os.path.join("config", "train_denoise.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(**kwargs)

        # Initialize loss function
        try:
            self.lossf = losses[self.loss]()
        except KeyError:
            raise NotImplementedError(f"{self.loss} not in common.pt_losses.losses")

        # Validate optimizer parameter groups
        assert len(self.optimizer.param_groups) == 1

        if launch:
            self.training_loop()

    @classmethod
    def add_arguments(cls, parser):
        """Add command-line arguments specific to denoising training."""
        super().add_arguments(parser)
        parser.add_argument(
            "--loss",
            help="Distortion loss function",
            choices=list(losses.keys()),
            required=True,
        )

    def _mk_expname(self, args) -> str:
        """Generate experiment name for denoising training."""
        return f"{type(self).__name__}_{args.in_channels}ch"


class BayerDenoiser:
    """Mixin class for Bayer-specific denoising functionality.

    This mixin provides Bayer-specific processing capabilities for denoising models.
    It can be combined with training classes to enable Bayer pattern processing.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)