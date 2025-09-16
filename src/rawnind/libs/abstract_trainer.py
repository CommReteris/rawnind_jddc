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

# sys.path.append("..")
from common.libs import json_saver, locking, pt_helpers, pt_losses, utilities
from common.tools import save_src
from rawnind.libs import raw, rawds, rawproc
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


class ImageToImageNN:
    """Base class for image-to-image training/inference pipelines.

    This class centralizes argument parsing, logging, device setup, model instantiation
    and common utilities shared by training and evaluation subclasses. Subclasses are
    expected to implement instantiate_model() and may override add_arguments() and
    processing hooks as needed.
    """
    CLS_CONFIG_FPATHS = [
        os.path.join("config", "test_reserve.yaml"),
    ]

    def __init__(self, **kwargs):
        """Initialize the image-to-image neural network framework.
        
        This constructor sets up the entire infrastructure for training and evaluating
        image-to-image neural networks, including argument parsing, logging configuration,
        device setup, model instantiation, and metric initialization.
        
        The initialization process follows these steps:
        1. Skip initialization if already initialized (for multiple inheritance)
        2. Parse and process command-line and config file arguments
        3. Set up logging infrastructure
        4. Create directory structure for experiment outputs
        5. Back up source code for reproducibility
        6. Instantiate the neural network model
        7. Load pre-trained weights if specified
        8. Initialize evaluation metrics
        
        Args:
            **kwargs: Keyword arguments that can override command-line arguments.
                Common parameters include:
                - test_only: Boolean flag for evaluation-only mode
                - preset_args: Dictionary of arguments to override parsed arguments
                - device: String specifying the computation device ("cuda" or "cpu")
                
        Notes:
            - The method modifies the instance state extensively, adding all parsed
              arguments as instance attributes
            - For subclasses, the instantiate_model() method must be implemented
            - The logger is configured to output to both file and console
            - In test_only mode, model saving and certain training preparations are skipped
        """
        # skip if already initialized, by checking for self.device
        if hasattr(self, "device"):
            return

        # initialize subclasses-initialized variables to satisfy the linter
        self.save_dpath: str = None  # type: ignore
        # get args
        if "test_only" in kwargs:
            self.test_only = kwargs["test_only"]
        args = self.get_args(
            ignore_unknown_args=hasattr(self, "test_only") and self.test_only
        )
        if "preset_args" in kwargs:
            vars(args).update(kwargs["preset_args"])
            vars(self).update(kwargs["preset_args"])
        self.__dict__.update(
            vars(args)
        )  # needed here because _get_resume_suffix uses self.loss
        self.autocomplete_args(args)
        self.__dict__.update(vars(args))
        if not hasattr(self, "test_only"):
            self.test_only = kwargs.get("test_only", False)
        if not self.test_only:
            self.save_args(args)
        self.save_cmd()
        self.device = pt_helpers.get_device(args.device)
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True  # type: ignore
        # torch.autograd.set_detect_anomaly(True)
        self.__dict__.update(kwargs)

        # get logger
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
        logging.info(" ".join(sys.argv))
        logging.info(f"PID: {os.getpid()}")
        logging.info(f"{self.__dict__=}")

        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)

        save_src.save_src(
            dest_root_dpath=os.path.join(self.save_dpath, "src"),
            included_dirs=("rawnind", "common"),
        )

        # instantiate model
        self.instantiate_model()
        if self.load_path:
            self.load_model(self.model, self.load_path, device=self.device)

        # init metrics
        metrics = {}
        for metric in self.metrics:
            metrics[metric] = pt_losses.metrics[metric]()
        self.metrics = metrics

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, device=None) -> None:
        """Load pre-trained weights into a model.
        
        This static method loads model weights from a saved checkpoint file into the
        provided model instance, moving the weights to the specified device.
        
        Args:
            model: PyTorch model instance to load weights into
            path: File path to the saved model state dictionary (.pt or .pth file)
            device: Optional device to load the model weights onto (e.g., "cuda:0" or "cpu")
                   If None, the weights are loaded onto the default device
                    
        Returns:
            None
            
        Raises:
            FileNotFoundError: If the specified path does not exist
            RuntimeError: If the state dict structure doesn't match the model architecture
            
        Notes:
            - The method verifies the file exists before attempting to load it
            - A log message is generated upon successful loading
            - In case of a missing file, the debugger is invoked before raising the exception
              (when run in a debugging environment)
        """
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            logging.info(f"Loaded model from {path}")
        else:
            breakpoint()
            raise FileNotFoundError(path)

    def infer(
            self,
            img: torch.Tensor,
            return_dict=False,  # , rgb_xyz_matrix=None, ret_img_only=False, match_gains=True
    ) -> dict:
        """Perform inference with the model on input images.
        
        This method runs the model in evaluation mode on the input images, performing
        the image-to-image transformation (such as denoising or compression/decompression).
        It handles batched or single images and performs appropriate device transfers.
        
        Args:
            img: Input image tensor with shape [C,H,W] or [B,C,H,W], where:
                 B = batch size (optional, will be added if missing)
                 C = number of channels (must match model's expected input channels)
                 H, W = height and width dimensions
            return_dict: If True, returns a dictionary containing model outputs
                        (e.g., {"reconstructed_image": tensor, "bpp": value} for compression models);
                        if False, returns just the reconstructed image tensor
                        
        Returns:
            If return_dict=False: torch.Tensor containing the processed image(s)
            If return_dict=True: dict containing model outputs (always includes "reconstructed_image")
            
        Raises:
            AssertionError: If the input image channels don't match the model's expected input channels
            
        Notes:
            - Input image is automatically converted to a batch if it's a single image
            - Inference is performed with torch.no_grad() for efficiency
            - The model is automatically set to evaluation mode during inference
            - Dictionary output may contain additional metrics like bits-per-pixel (bpp) for compression models
        """
        with torch.no_grad():
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            in_channels = img.shape[1]
            assert in_channels == self.in_channels, (
                f"{in_channels=}, {self.in_channels=}; model configuration does not match input image."
            )
            img = img.to(self.device)
            # img = pt_ops.crop_to_multiple(img, 16)
            # if rgb_xyz_matrix is not None:
            #     rgb_xyz_matrix = rgb_xyz_matrix.to(self.device)
            output = self.model.eval()(img)
            if return_dict:
                if isinstance(output, torch.Tensor):
                    return {"reconstructed_image": output}
                return output
            return output["reconstructed_image"]
        #     output_img = output["reconstructed_image"]
        #     output_img = rawproc.match_gain(img, output_img)
        #     if rgb_xyz_matrix is not None:
        #         output_img = rawproc.camRGB_to_lin_rec2020_images(
        #             output_img, torch.from_numpy(rgb_xyz_matrix).unsqueeze(0)
        #         )
        #         output_img = rawproc.match_gain(img, output_img)
        # if ret_img_only:
        #     return output_img.to(orig_device)
        # output["proc_img"] = output_img.to(orig_device)
        return output

    @staticmethod
    def get_best_step(
            model_dpath: str,
            suffix: str,
            prefix: str = "val",
            # suffix="combined_loss",
    ) -> dict:
        """Find the best-performing model checkpoint based on a specific metric.
        
        This method locates the best iteration of a trained model by examining the
        training results file (trainres.yaml) and identifying which training step
        achieved the best performance according to the specified metric.
        
        Args:
            model_dpath: Path to the model's experiment directory containing 
                        the trainres.yaml file and saved_models subdirectory
            suffix: Metric name suffix to use for finding the best step 
                   (e.g., "msssim", "psnr", "combined_loss")
            prefix: Metric name prefix, typically "val" for validation metrics
                   or "test" for test metrics
                   
        Returns:
            dict: Dictionary containing:
                - "fpath": Full path to the best model checkpoint file
                - "step_n": The iteration/step number of the best checkpoint
                
        Raises:
            FileNotFoundError: If the trainres.yaml file doesn't exist in model_dpath
            KeyError: If the specified metric (prefix_suffix) isn't found in the results file
            
        Notes:
            - The metric is constructed as "{prefix}_{suffix}" (e.g., "val_msssim")
            - The method assumes that model checkpoints follow the naming pattern 
              "iter_{step_number}.pt" and are stored in a "saved_models" subdirectory
            - The trainres.yaml file is expected to have a "best_step" section that 
              maps metric names to their best iteration numbers
        """
        jsonfpath = os.path.join(model_dpath, "trainres.yaml")
        if not os.path.isfile(jsonfpath):
            raise FileNotFoundError(
                "get_best_checkpoint: jsonfpath not found: {}".format(jsonfpath)
            )
        results = utilities.load_yaml(jsonfpath, error_on_404=False)
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
        """Get a transfer function for image pixel value transformation.
        
        This method provides a centralized way to access various transfer functions
        used for image processing. Transfer functions transform pixel values, often
        to convert between different color spaces or to apply non-linear corrections.
        
        Available transfer functions:
        - "None": Identity function (returns input unchanged)
        - "pq": Perceptual Quantizer encoding (converts from scene-linear to PQ encoding)
        - "gamma22": Gamma correction with γ=2.2 (standard sRGB gamma)
        
        Args:
            fun_name: String identifier for the desired transfer function
            
        Returns:
            Callable[[torch.Tensor], torch.Tensor]: A function that applies the 
            requested transfer to tensor inputs
            
        Raises:
            ValueError: If the requested transfer function name is not recognized
            
        Examples:
            >>> transfer_fn = get_transfer_function("gamma22")
            >>> corrected_image = transfer_fn(linear_image)
        """
        if str(fun_name) == "None":
            return lambda img: img
        elif fun_name == "pq":
            return rawproc.scenelin_to_pq
        elif fun_name == "gamma22":
            return lambda img: rawproc.gamma(img, gamma_val=2.2, in_place=True)
        else:
            raise ValueError(fun_name)

    @staticmethod
    def save_args(args):
        """Save command-line arguments to a YAML file for experiment reproducibility.
        
        This method preserves all arguments used to run an experiment by serializing
        them to a YAML file, which can be used later to reproduce the experiment or
        understand its configuration.
        
        Args:
            args: Namespace object containing command-line arguments, typically from
                 argparse.ArgumentParser.parse_args() or configargparse.parse_args()
                
        Notes:
            - Creates the save directory if it doesn't exist
            - Converts the Namespace object to a dictionary using vars()
            - Saves the arguments to args.yaml in the directory specified by args.save_dpath
            - The resulting YAML file can be loaded later to recreate the same arguments
            - This is critical for experiment reproducibility and tracking
        """
        os.makedirs(args.save_dpath, exist_ok=True)
        out_fpath = os.path.join(args.save_dpath, "args.yaml")
        utilities.dict_to_yaml(vars(args), out_fpath)

    def save_cmd(self):
        """Save the command line invocation to a shell script file.
        
        This method preserves the exact command used to run the current experiment
        by saving it to a shell script file. If the file already exists, previous
        commands are preserved as comments, creating a history of experiment runs.
        
        The filename is determined by the mode of operation:
        - test_cmd.sh for evaluation mode (test_only=True)
        - train_cmd.sh for training mode (test_only=False)
        
        This allows for easy reproduction of experiments by simply executing the
        generated script file.
        
        Notes:
            - Creates the save directory if it doesn't exist
            - Preserves previous commands as comments if the file exists
            - Always appends the current command to the end of the file
            - The command is reconstructed from sys.argv to include all arguments
            - The resulting shell script can be executed directly to reproduce 
              the experiment
        """
        os.makedirs(self.save_dpath, exist_ok=True)
        out_fpath = os.path.join(
            self.save_dpath, "test_cmd.sh" if self.test_only else "train_cmd.sh"
        )
        cmd = "python " + " ".join(sys.argv)

        # Read the current cmd.sh file and comment every line
        with open(out_fpath, "w+") as f:
            lines = f.readlines()
            f.seek(0)  # Move the file pointer to the beginning of the file

            # Write the modified lines
            for line in lines:
                f.write("# " + line)

            # Write the current cmd at the end of the file
            f.write(cmd)

    def get_args(self, ignore_unknown_args: bool = False):
        """Parse command-line arguments and configuration files.
        
        This method sets up a ConfigArgParse parser that can handle both command-line
        arguments and YAML configuration files. It registers all model parameters by
        calling add_arguments() and then parses the arguments.
        
        The method supports two parsing modes:
        - Standard mode: All arguments must be recognized; unknown arguments cause errors
        - Ignore mode: Unknown arguments are silently ignored, useful for test-only mode
          where not all training parameters are needed
        
        Args:
            ignore_unknown_args: If True, unknown arguments are ignored rather than
                                raising an error (useful for test-only mode)
                                
        Returns:
            argparse.Namespace: Object containing all parsed arguments as attributes
            
        Notes:
            - Uses ConfigArgParse for combined CLI and config file support
            - Default config files are specified in CLS_CONFIG_FPATHS class attribute
            - YAML is used as the config file format
            - All available parameters are defined in the add_arguments method
            - Configuration priority (highest to lowest):
                1. Command-line arguments
                2. Values from specified config file (--config)
                3. Values from default config files
                4. Default values defined in add_arguments
        """
        parser = configargparse.ArgumentParser(
            description=__doc__,
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            default_config_files=self.CLS_CONFIG_FPATHS,
        )
        self.add_arguments(parser)
        if ignore_unknown_args:
            # if hasattr(self, "test_only") and self.test_only:
            return parser.parse_known_args()[0]
        return parser.parse_args()

    def add_arguments(self, parser):
        """Register command-line arguments and configuration parameters.
        
        This method defines all available parameters for the model, including paths,
        model architecture, training settings, and debug options. These parameters
        can be specified via command-line arguments or configuration files.
        
        The base ImageToImageNN class defines common parameters applicable to all
        image-to-image models. Subclasses typically override this method to add
        their specific parameters while also calling the parent implementation.
        
        Args:
            parser: ConfigArgParse parser to register arguments with
            
        Notes:
            - The --config argument specifies an external YAML configuration file
            - Some arguments like --arch are required while others are optional
            - Some arguments have default values that can be overridden
            - Arguments with choices have their valid options explicitly listed
            - Many arguments support auto-generation/completion via autocomplete_args
            - The following common parameter categories are defined:
                - Configuration options (--config)
                - Model architecture (--arch, --funit, --in_channels)
                - Experiment naming and paths (--expname, --load_path, --save_dpath)
                - Computation options (--device)
                - Debugging flags (--debug_options)
                - Evaluation metrics (--metrics)
                - Image processing options (--match_gain)
        """
        parser.add_argument(
            "--config",
            is_config_file=True,
            dest="config",
            required=False,
            help="config in yaml format",
        )
        parser.add_argument(
            "--arch",
            help="Model architecture",
            required=True,
            choices=self.ARCHS.keys(),
        )
        parser.add_argument("--funit", help="Filters multiplier", type=int)
        parser.add_argument(
            "--init_step",
            type=int,
            help="Initial step (default: continued from load_path or 0)",
        )
        parser.add_argument(
            "--in_channels",
            type=int,
            help="Number of input channels (3 for profiled RGB, 4 for Bayer)",
            choices=[3, 4],
        )
        parser.add_argument(
            "--expname",
            help="Experiment name (if not specified it is auto-generated and auto-incremented)",
        )
        parser.add_argument(
            "--load_path",
            help="Path of model to load (either model directory or checkpoint filepath), or expname",
        )
        parser.add_argument(
            "--save_dpath",
            help=f"Save directory (default is auto-generated as {self.MODELS_BASE_DPATH}/<expname>)",
        )
        # parser.add_argument(
        #     "--test_only",
        #     help="(used internally by test tools, TODO deprecate in favor of ImageToImageNNTesting)",
        #     action="store_true",
        # )
        parser.add_argument(
            "--device", type=int, help="CUDA device number (-1 for CPU)"
        )
        parser.add_argument(  # ideally we'd have two sets of debug options for test/train, but oh well
            "--debug_options",
            nargs="*",
            default=[],
            choices=[
                "1thread",
                "timing",
                "learn_validation",
                "output_valtest_images",
                "output_train_images",
                "skip_initial_validation",
                "spam",
                "minimize_threads",
                "toy_dataset",
            ],
            help=f"Debug options",
        )
        parser.add_argument(
            "--metrics",
            nargs=("*"),
            default=[],
            help=f"Validation and test metrics (not used to update the LR): {pt_losses.metrics}",
        )
        parser.add_argument(
            "--match_gain",
            help="When to match gains wrt ground-truth.",
            required=True,
            choices=["input", "output", "never"],
        )

    def autocomplete_args(self, args):
        """Auto-complete and validate argument values with intelligent defaults.
        
        This method processes the parsed arguments to fill in missing values,
        resolve interdependencies, and ensure consistency. It reduces the burden
        on users by providing sensible defaults and handling complex path relationships.
        
        Key auto-completion actions:
        1. Experiment name (expname):
           - If not provided, generates based on class name, channel count, and parameters
           - May incorporate config filename or additional comments
           
        2. Load path (load_path):
           - Can accept a directory path, experiment name, or specific model file
           - Resolves to the best model in a directory based on validation metrics
           - Sets initial training step based on loaded model if not explicitly provided
           
        3. Save directory (save_dpath):
           - If not provided, generates based on models base path and experiment name
           - Creates directory structure for experiment outputs
           
        Args:
            args: Namespace object containing parsed arguments to be auto-completed
            
        Notes:
            - The method modifies the args object in-place
            - Some arguments may depend on values from other arguments
            - Path resolution handles both relative and absolute paths
            - Experiment naming follows conventions for easier organization
            - For continued training, load_path and save_dpath may be the same
            - The method includes special handling for various model architectures
            - Validation checks ensure arguments are consistent and valid
        """
        # generate expname and save_dpath, and (incomplete/dir_only) load_path if continue_training_from_last_model_if_exists
        if not args.expname:
            assert args.save_dpath is None, "incompatible args: save_dpath and expname"
            if not args.config:
                args.expname = self._mk_expname(args)
            else:
                args.expname = utilities.get_leaf(args.config).split(".")[0]
            if args.comment:
                args.expname += "_" + args.comment + "_"

            # handle duplicate expname -> increment
            dup_cnt = None
            while os.path.isdir(
                    save_dpath := os.path.join(self.MODELS_BASE_DPATH, args.expname)
            ):
                dup_cnt: int = 1
                while os.path.isdir(f"{save_dpath}-{dup_cnt}"):
                    dup_cnt += 1  # add a number to the last model w/ same expname
                # but load the previous model if continue_training_from_last_model_if_exists or testing
                if args.continue_training_from_last_model_if_exists:
                    if dup_cnt > 1:
                        args.load_path = f"{args.expname}-{dup_cnt - 1}"
                    elif dup_cnt == 1:
                        args.load_path = args.expname
                    else:
                        raise ValueError("bug")
                args.expname = f"{args.expname}-{dup_cnt}"
            # if we want to continue training from last model and there are none from this experiment but fallback_load_path is specified, then load that model and reset the step and learning_rate
            args.save_dpath = save_dpath
        else:
            args.save_dpath = os.path.join(self.MODELS_BASE_DPATH, args.expname)
            os.makedirs(self.MODELS_BASE_DPATH, exist_ok=True)
        # if vars(self).get(
        #    "test_only", False
        # ):  # and args.load_path is None:  args.load_path is the previous best model whereas we want to find the current best one.
        # if self.test_only and args.load_path is None:
        if vars(self).get("test_only", False) and args.load_path is None:
            args.load_path = args.expname
            dup_cnt = None

        def complete_load_path_and_init_step():
            if os.path.isfile(args.load_path) or args.load_path.endswith(".pt"):
                if args.init_step is None:
                    try:
                        args.init_step = int(
                            args.load_path.split(".")[-2].split("_")[-1]
                        )
                    except ValueError as e:
                        logging.warning(
                            f"autocomplete_args: unable to parse init_step from {args.load_path=} ({e=})"
                        )
            else:
                if not os.path.isdir(args.load_path):
                    args.load_path = os.path.join(
                        self.MODELS_BASE_DPATH, args.load_path
                    )
                # FIXME? following line will raise FileNotFoundError if trainres.yaml does not exist

                best_step = self.get_best_step(
                    model_dpath=args.load_path, suffix=self._get_resume_suffix()
                )
                args.load_path = best_step["fpath"]
                # check if there are newer models
                if vars(args).get(
                        "continue_training_from_last_model_if_exists"
                ) and not vars(self).get("test_only", False):
                    # if args.continue_training_from_last_model_if_exists:
                    dup_cnt_load = None if dup_cnt is None else dup_cnt - 1
                    while not os.path.isfile(args.load_path):
                        logging.info(
                            f"warning: {args.load_path} not found, trying previous model"
                        )
                        if not dup_cnt_load:
                            args.load_path = None
                            logging.warning("no model to load")
                            if vars(self).get("test_only", False):
                                raise ValueError(f"No model to load")
                            return
                        if dup_cnt_load > 1:
                            args.load_path = args.load_path.replace(
                                f"-{dup_cnt_load}{os.sep}",
                                f"-{dup_cnt_load - 1}{os.sep}",
                            )
                            dup_cnt_load -= 1
                        elif dup_cnt_load == 1:
                            args.load_path = args.load_path.replace(
                                f"-{dup_cnt_load}{os.sep}", os.sep
                            )
                            dup_cnt_load = None
                        else:
                            raise ValueError("bug")
                if args.init_step is None:
                    args.init_step = best_step["step_n"]

        # breakpoint()
        if args.load_path is None and args.fallback_load_path is not None:
            args.load_path = (
                find_best_expname_iteration.find_latest_model_expname_iteration(
                    args.fallback_load_path
                )
            )
            args.init_step = 0
        if args.load_path:
            try:
                complete_load_path_and_init_step()
            except KeyError as e:
                logging.error(f"KeyError: {e=}; unable to load previous model.")
                args.load_path = None
                args.init_step = 0
        if args.init_step is None:
            args.init_step = 0

        # if args.continue_training_from_last_model and not args.expname:
        #     if not args.load_path:
        #         args.load_path
        #     self.autocomplete_args(args)  # first pass w/ continue: we determine the expname
        if (
                hasattr(self, "test_only")
                and self.test_only
                and "/scratch/" in vars(args).get("noise_dataset_yamlfpaths", "")
        ):
            # FIXME this doesn't always work, eg "tools/validate_and_test_dc_prgb2prgb.py --config /orb/benoit_phd/models/rawnind_dc/DCTrainingProfiledRGBToProfiledRGB_3ch_L64.0_Balle_Balle_2023-10-27-dc_prgb_msssim_mgout_64from128_x_x_/args.yaml --device -1
            # when noise_dataset_yamlfpaths is not overwritten through preset_args
            args.noise_dataset_yamlfpaths = [rawproc.RAWNIND_CONTENT_FPATH]
        # args.load_key_metric = f"val_{self._get_resume_suffix()}"  # this would have been nice for tests to have but not implemented on time


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
        lock_fpath = f"validation_{os.uname()[1]}_{os.environ.get('CUDA_VISIBLE_DEVICES', 'unk')}.lock"
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


class BayerImageToImageNN(ImageToImageNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process_net_output(
            self,
            camRGB_images: torch.Tensor,
            rgb_xyz_matrix: torch.Tensor,
            gt_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Process camRGB output s.t. it becomes closer to the final output.
        1. Match exposure if gt_images is provided
        2. Apply Lin. Rec. 2020 color profile
        3. Apply the Rec. 2100 Perceptual Quantizer (actually do this separately elsewhere)

        Args:
            camRGB_images (torch.Tensor): network output to convert
            rgb_xyz_matrix (torch.Tensor): camRGB to lin_rec2020 conversion matrices
            gt_images (Optional[torch.Tensor], optional): Ground-truth images to match exposure against (if provided). Defaults to None.
        """
        if gt_images is not None and self.match_gain == "output":
            camRGB_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=camRGB_images
            )
        output_images = rawproc.camRGB_to_lin_rec2020_images(
            camRGB_images, rgb_xyz_matrix
        )
        if (
                gt_images is not None and self.match_gain == "output"
        ):  # this is probably overkill
            output_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=output_images
            )
        return output_images

    def add_arguments(self, parser):
        super().add_arguments(parser)

        parser.add_argument(
            "--preupsample",
            action="store_true",
            help="Upsample bayer image before processing it.",
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


class DenoiseCompress(ImageToImageNN):
    MODELS_BASE_DPATH = os.path.join("..", "..", "models", "rawnind_dc")
    ARCHS = {
        "ManyPriors" : manynets_compression.ManyPriors_RawImageCompressor,
        "DenoiseThenCompress": denoise_then_compress.DenoiseThenCompress,
        "JPEGXL"     : standard_compressor.JPEGXL_ImageCompressor,
        "JPEG"       : standard_compressor.JPEGXL_ImageCompressor,
        "Passthrough": standard_compressor.Passthrough_ImageCompressor,
    }
    ARCHS_ENC = {
        "Balle": compression_autoencoders.BalleEncoder,
        # "BayerPreUp": compression_autoencoders.BayerPreUpEncoder,
    }
    ARCHS_DEC = {
        "Balle": compression_autoencoders.BalleDecoder,
        "BayerPS": compression_autoencoders.BayerPSDecoder,
        "BayerTC": compression_autoencoders.BayerTCDecoder,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def instantiate_model(self) -> None:
        self.model: torch.nn.Module = self.ARCHS[self.arch](
            in_channels=self.in_channels,
            funit=self.funit,
            device=self.device,
            hidden_out_channels=self.hidden_out_channels,
            bitstream_out_channels=self.bitstream_out_channels,
            encoder_cls=self.ARCHS_ENC[self.arch_enc],
            decoder_cls=self.ARCHS_DEC[self.arch_dec],
            preupsample=vars(self).get("preupsample", False),
        ).to(self.device)

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--arch_enc",
            help="Encoder architecture",
            required=True,
            choices=self.ARCHS_ENC.keys(),
        )
        parser.add_argument(
            "--arch_dec",
            help="Decoder architecture",
            required=True,
            choices=self.ARCHS_DEC.keys(),
        )
        parser.add_argument("--hidden_out_channels", type=int)
        parser.add_argument("--bitstream_out_channels", type=int)

    def _get_resume_suffix(self) -> str:
        return "combined" + self._get_lossn_extension()


class BayerDenoiseCompress(DenoiseCompress, BayerImageToImageNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DenoiseCompressTraining(ImageToImageNNTraining, DenoiseCompress):
    CLS_CONFIG_FPATHS = ImageToImageNN.CLS_CONFIG_FPATHS + [
        os.path.join("config", "train_dc.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(**kwargs)
        try:
            self.lossf = pt_losses.losses[
                self.loss
            ]()  # actually visual loss function, kept name for compatibility.
        except KeyError:
            raise NotImplementedError(f"{self.loss} not in common.pt_losses.losses")
        assert (
                len(self.optimizer.param_groups) == 3
                or self.arch == "JPEGXL"
                or self.arch == "Passthrough"
        )  # match adjust_lr function
        if launch:
            self.training_loop()

    def init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.get_parameters(
                lr=self.init_lr,
                bitEstimator_lr_multiplier=self.bitEstimator_lr_multiplier,
            ),
            lr=self.init_lr,
        )

    def add_arguments(self, parser):
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
            choices=pt_losses.losses.keys(),
            required=True,
        )

    def _mk_expname(self, args: configargparse.Namespace) -> str:
        return f"{type(self).__name__}_{args.in_channels}ch_L{args.train_lambda}_{args.arch_enc}_{args.arch_dec}"

    @staticmethod
    def clip_gradient(optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)


class Denoiser(ImageToImageNN):
    MODELS_BASE_DPATH = os.path.join("..", "..", "models", "rawnind_denoise")
    ARCHS = {
        "unet"  : raw_denoiser.UtNet2,
        "utnet3": raw_denoiser.UtNet3,
        # "runet": runet.Runet,
        "identity": raw_denoiser.Passthrough,
        # "edsr": edsr.EDSR,
        "bm3d"  : bm3d_denoiser.BM3D_Denoiser,
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_resume_suffix(self: "DenoiserTraining") -> str:
        return self.loss + self._get_lossn_extension()

    def instantiate_model(self):
        self.model: torch.nn.Module = self.ARCHS[self.arch](
            in_channels=self.in_channels,
            funit=self.funit,
            preupsample=vars(self).get("preupsample", False),
        ).to(self.device)

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--loss",
            help="Distortion loss function",
            choices=pt_losses.losses.keys(),
            required=True,
        )


class BayerDenoiser(Denoiser, BayerImageToImageNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DenoiserTraining(ImageToImageNNTraining, Denoiser):
    CLS_CONFIG_FPATHS = ImageToImageNN.CLS_CONFIG_FPATHS + [
        os.path.join("config", "train_denoise.yaml")
    ]

    def __init__(self, launch=False, **kwargs):
        super().__init__(**kwargs)

        try:
            self.lossf = pt_losses.losses[self.loss]()
        except KeyError:
            raise NotImplementedError(f"{self.loss} not in common.pt_losses.losses")

        assert len(self.optimizer.param_groups) == 1  # match adjust_lr function
        if launch:
            self.training_loop()

    # def add_arguments(self, parser):
    #     super().add_arguments(parser)
    # parser.add_argument(
    #     "--train_gamma",
    #     help="Which gamma is applied to the training loss.",
    #     required=True,
    # )

    def _mk_expname(self, args: configargparse.Namespace) -> str:
        return f"{type(self).__name__}_{args.in_channels}ch"


# class ImageToImageNNTesting(ImageToImageNN):
#     def __init__(self, **kwargs) -> None:
#         """Initialize an image to image neural network tester."""
#         if hasattr(self, 'json_saver'):
#             return
#         super().__init__(**kwargs)
#         res_fpath: str = os.path.join(self.save_dpath, "testres.json")
#         self.json_saver = json_saver.JSONSaver(res_fpath)
#         logging.info(f"See {res_fpath} for results.")

#     def add_arguments(self, parser):
#         super().add_arguments(parser)


def get_and_load_test_object(
        **kwargs,
) -> ImageToImageNN:  # only used in denoise_image.py
    """Parse config file or arch parameter to get the class name, ie Denoiser or DenoiseCompress."""
    parser = configargparse.ArgumentParser(
        description=__doc__,
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    parser.add_argument(
        "--config",
        is_config_file=True,
        dest="config",
        required=False,
        help="config in yaml format",
    )
    parser.add_argument(
        "--arch",
        help="Model architecture",
        required=True,
        choices=Denoiser.ARCHS.keys() | DenoiseCompress.ARCHS.keys(),
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        help="Number of input channels (3 for profiled RGB, 4 for Bayer)",
        choices=[3, 4],
        required=True,
    )
    args, _ = parser.parse_known_args()
    if args.arch in Denoiser.ARCHS.keys():
        if args.in_channels == 4:
            test_obj = BayerDenoiser(test_only=True, **kwargs)
        else:
            test_obj = Denoiser(test_only=True, **kwargs)
    elif args.arch in DenoiseCompress.ARCHS.keys():
        if args.in_channels == 4:
            test_obj = BayerDenoiseCompress(test_only=True, **kwargs)
        else:
            test_obj = DenoiseCompress(test_only=True, **kwargs)
    else:
        raise NotImplementedError(f"Unknown architecture {args.arch}")
    # if "output_valtest_images" not in test_obj.debug_options:
    #     print("Warning: add --debug_options output_valtest_images to output images.")
    # FIXME actually pretty useless as long as output images overwrite each other, TODO add each name and crop number or create unique names
    test_obj.model = test_obj.model.eval()
    return test_obj


def get_and_load_model(**kwargs):
    """Deprecate if not needed? -- only used in denoise_image.py"""
    return get_and_load_test_object(**kwargs).model
