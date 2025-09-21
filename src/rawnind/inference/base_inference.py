"""Base inference classes for model inference operations.

This module provides the base classes for inference operations, extracted
from the original abstract_trainer.py as part of the codebase refactoring.

Classes:
    ImageToImageNN: Base class for image-to-image inference pipelines
    BayerImageToImageNN: Specialized class for Bayer pattern handling
"""

import logging
import os
import sys
from typing import Callable, Optional

import torch

from ..dependencies.json_saver import YAMLSaver
from ..dependencies.pt_losses import metrics
from ..dependencies.pytorch_helpers import get_device
from ..dependencies.utilities import dict_to_yaml


class ImageToImageNN:
    """Base class for image-to-image inference pipelines.

    This class centralizes argument parsing, logging, device setup, model instantiation
    and common utilities shared by inference subclasses. Subclasses are
    expected to implement instantiate_model() and may override add_arguments() and
    processing hooks as needed.
    """
    CLS_CONFIG_FPATHS = [
        os.path.join("config", "test_reserve.yaml"),
    ]

    def __init__(self, **kwargs):
        """Initialize the image-to-image neural network framework.

        This constructor sets up the entire infrastructure for inference with
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
        self.device = get_device(args.device)
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

        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)

        # instantiate model
        self.instantiate_model()
        if self.load_path:
            self.load_model(self.model, self.load_path, device=self.device)

        # init metrics
        metrics_dict = {}
        for metric in self.metrics:
            metrics_dict[metric] = metrics[metric]()
        self.metrics = metrics_dict

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
        from ..dependencies.utilities import load_yaml
        jsonfpath = os.path.join(model_dpath, "trainres.yaml")
        if not os.path.isfile(jsonfpath):
            raise FileNotFoundError(
                "get_best_checkpoint: jsonfpath not found: {}".format(jsonfpath)
            )
        results = load_yaml(jsonfpath, error_on_404=False)
        metric = "{}_{}".format(prefix, suffix)
        try:
            best_step = results["best_step"][metric]
        except KeyError as e:
            raise KeyError(f'"{metric}" not found in {jsonfpath=}') from e
        return {
            "fpath" : os.path.join(model_dpath, "saved_models", f"iter_{best_step}.pt"),
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
        from ..dependencies import rawproc
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
        dict_to_yaml(vars(args), out_fpath)

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
        import configargparse

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
            help=f"Validation and test metrics (not used to update the LR): {metrics}",
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
                args.expname = dict_to_yaml.get_leaf(args.config).split(".")[0]
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
            from ..dependencies import rawproc
            args.noise_dataset_yamlfpaths = [rawproc.RAWNIND_CONTENT_FPATH]
        # args.load_key_metric = f"val_{self._get_resume_suffix()}"  # this would have been nice for tests to have but not implemented on time


class BayerImageToImageNN(ImageToImageNN):
    """Specialized class for Bayer pattern image-to-image inference.

    This class extends ImageToImageNN with Bayer-specific processing capabilities,
    including color space conversion and demosaicing operations.
    """

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
        from ..dependencies import rawproc
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
