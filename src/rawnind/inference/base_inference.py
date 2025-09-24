'''Base inference classes for model inference operations.

This module provides the base classes for inference operations, extracted
from the original abstract_trainer.py as part of the codebase refactoring.

Classes:
    ImageToImageNN: Base class for image-to-image inference pipelines
    BayerImageToImageNN: Specialized class for Bayer pattern handling

Refactored to use clean API with InferenceConfig dataclass instead of argparse.
'''

import logging
import os
import sys
from typing import Callable, Optional, Dict, Any

import torch

from ..dependencies.json_saver import YAMLSaver
from ..dependencies.pt_losses import metrics
from ..dependencies.pytorch_helpers import get_device
from ..dependencies.json_saver import dict_to_yaml, load_yaml

from .configs import InferenceConfig


class ImageToImageNN:
    '''Base class for image-to-image inference pipelines.

    This class centralizes configuration handling, logging, device setup, model instantiation
    and common utilities shared by inference subclasses. Subclasses are
    expected to implement instantiate_model() and may override add_arguments() and
    processing hooks as needed.
    '''
    CLS_CONFIG_FPATHS = [
        os.path.join("config", "test_reserve.yaml"),
    ]
    
    # Base architectures - subclasses should override
    ARCHS = {
        "identity": None,  # Placeholder - subclasses must override
    }
    
    # Base models directory - subclasses should override  
    MODELS_BASE_DPATH = os.path.join("..", "..", "models", "rawnind_base")

    def __init__(self, config: InferenceConfig):
        '''Initialize the image-to-image neural network framework.

        This constructor sets up the entire infrastructure for inference with
        image-to-image neural networks. It supports direct programmatic initialization
        using the InferenceConfig dataclass.

        Args:
            config: InferenceConfig dataclass with all parameters
        '''
        # Skip if already initialized, by checking for self.device
        if hasattr(self, "device"):
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
        self._setup_logging()
        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)
        
        # Setup device
        self.device = get_device(self.device)
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True  # type: ignore

        # Instantiate model
        self.instantiate_model()
        if getattr(self, 'load_path', None):
            self.load_model(self.model, self.load_path, device=self.device)

        # Init metrics
        metrics_dict = {}
        for metric in getattr(self, 'metrics', []):
            metrics_dict[metric] = metrics[metric]()
        self.metrics = metrics_dict
    
    def instantiate_model(self):
        '''Instantiate the neural network model.
        
        This method should be implemented by subclasses to create the specific
        model architecture. The base implementation raises NotImplementedError.
        '''
        raise NotImplementedError("Subclasses must implement instantiate_model()")
    
    def _get_resume_suffix(self) -> str:
        '''Get the suffix for determining the best model checkpoint.
        
        This method should be implemented by subclasses to specify which metric
        to use when finding the best model checkpoint. For example, 'msssim' for
        denoising models or 'combined' for compression models.
        
        Returns:
            str: The metric suffix for model resume/loading
        '''
        return "default"
    
    def _mk_expname(self, config: InferenceConfig) -> str:
        '''Generate experiment name from configuration.
        
        This method creates a standardized experiment name based on the model
        configuration and inference parameters.
        
        Args:
            config: InferenceConfig containing model configuration
            
        Returns:
            str: Generated experiment name
        '''
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d")
        return f"{self.__class__.__name__}_{config.architecture}_{timestamp}"
    
    def autocomplete_config(self, config: InferenceConfig):
        '''Auto-complete and validate configuration values with intelligent defaults.

        This method processes the configuration to fill in missing values,
        resolve interdependencies, and ensure consistency. It reduces the burden
        on users by providing sensible defaults and handling complex path relationships.

        Key auto-completion actions:
        1. Experiment name (expname):
           - If not provided, generates based on class name, channel count, and parameters
           - May incorporate additional comments

        2. Load path (load_path):
           - Can accept a directory path, experiment name, or specific model file
           - Resolves to the best model in a directory based on validation metrics
           - Sets initial step based on loaded model if not explicitly provided

        3. Save directory (save_dpath):
           - If not provided, generates based on models base path and experiment name
           - Creates directory structure for experiment outputs

        Args:
            config: InferenceConfig object containing parsed configuration to be auto-completed

        Notes:
            - The method modifies the config object in-place
            - Some parameters may depend on values from other parameters
            - Path resolution handles both relative and absolute paths
            - Experiment naming follows conventions for easier organization
            - For continued inference, load_path and save_dpath may be the same
            - The method includes special handling for various model architectures
            - Validation checks ensure parameters are consistent and valid
        '''
        # Generate expname and save_dpath, and (incomplete/dir_only) load_path if continue_training_from_last_model_if_exists
        if not config.expname:
            assert config.save_dpath is None, "incompatible args: save_dpath and expname"
            if not config.config:
                config.expname = self._mk_expname(config)
            else:
                config.expname = dict_to_yaml.get_leaf(config.config).split(".")[0]
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
            # If we want to continue inference from last model and there are none from this experiment but fallback_load_path is specified, then load that model and reset the step and learning_rate
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
            from ..dependencies import raw_processing as rawproc
            config.noise_dataset_yamlfpaths = [rawproc.RAWNIND_CONTENT_FPATH]
        # config.load_key_metric = f"val_{self._get_resume_suffix()}"  # this would have been nice for tests to have but not implemented on time

    @staticmethod
    def save_args(config: InferenceConfig):
        '''Save configuration to a YAML file for experiment reproducibility.

        This method preserves all configuration used for inference by serializing
        them to a YAML file, which can be used later to reproduce the inference or
        understand its configuration.

        Args:
            config: InferenceConfig dataclass containing configuration parameters

        Notes:
            - Creates the save directory if it doesn't exist
            - Converts the dataclass to a dictionary using vars()
            - Saves the configuration to args.yaml in the directory specified by config.save_dpath
            - The resulting YAML file can be loaded later to recreate the same configuration
            - This is critical for experiment reproducibility and tracking
        '''
        os.makedirs(config.save_dpath, exist_ok=True)
        out_fpath = os.path.join(config.save_dpath, "args.yaml")
        dict_to_yaml(vars(config), out_fpath)

    def save_cmd(self):
        '''Save the command line invocation to a shell script file.

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
        '''
        os.makedirs(self.save_dpath, exist_ok=True)
        out_fpath = os.path.join(
            self.save_dpath, "test_cmd.sh" if self.test_only else "train_cmd.sh"
        )
        # Log configuration instead of command for clean API
        config_fpath = out_fpath + '.config'
        utilities.dict_to_yaml(vars(self.config), config_fpath)
        logging.info(f"Configuration saved to {config_fpath}")

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, device=None) -> None:
        if os.path.isfile(path):
            model.load_state_dict(torch.load(path, map_location=device))
            logging.info(f"Loaded model from {path}")
        else:
            breakpoint()
            raise FileNotFoundError(path)

    @staticmethod
    def _get_best_step_from_yaml(
            model_dpath: str,
            suffix: str,
            prefix: str = "val",
    ) -> dict:
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

    def _setup_logging(self):
        '''Setup logging configuration.''' 
        # Get logger
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # Only setup file logging if save_dpath exists and is not a temp directory
        if self.save_dpath and not self.save_dpath.startswith('/tmp'):
            os.makedirs(self.save_dpath, exist_ok=True)
            logging.basicConfig(
                filename=os.path.join(
                    self.save_dpath, f"{'test' if self.test_only else 'train'}.log"
                ),
                datefmt="%Y-%m-%d %H:%M:%S",
                format="%(asctime)s %(levelname)-8s %(message)s",
                level=logging.DEBUG if getattr(self, 'debug_options', []) else logging.INFO,
                filemode="w",
            )
        
        # Always add console logging
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        if hasattr(self, 'save_dpath') and not self.save_dpath.startswith('/tmp'):
            logging.info(" ".join(sys.argv))
            logging.info(f"PID: {os.getpid()}")

    @staticmethod
    def load_model(model: torch.nn.Module, path: str, device=None) -> None:
        '''Load pre-trained weights into a model.

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
        '''
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
        '''Perform inference with the model on input images.

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
        '''
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
        '''Find the best-performing model checkpoint based on a specific metric.

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
        '''
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
        '''Get a transfer function for image pixel value transformation.

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
        '''
        from ..dependencies import raw_processing as rawproc
        if str(fun_name) == "None":
            return lambda img: img
        elif fun_name == "pq":
            return rawproc.scenelin_to_pq
        elif fun_name == "gamma22":
            return lambda img: rawproc.gamma(img, gamma_val=2.2, in_place=True)
        else:
            raise ValueError(fun_name)

    @staticmethod
    def save_args(config: InferenceConfig):
        '''Save configuration to a YAML file for experiment reproducibility.

        This method preserves all configuration used for inference by serializing
        them to a YAML file, which can be used later to reproduce the inference or
        understand its configuration.

        Args:
            config: InferenceConfig dataclass containing configuration parameters

        Notes:
            - Creates the save directory if it doesn't exist
            - Converts the dataclass to a dictionary using vars()
            - Saves the configuration to args.yaml in the directory specified by config.save_dpath
            - The resulting YAML file can be loaded later to recreate the same configuration
            - This is critical for experiment reproducibility and tracking
        '''
        os.makedirs(config.save_dpath, exist_ok=True)
        out_fpath = os.path.join(config.save_dpath, "args.yaml")
        dict_to_yaml(vars(config), out_fpath)

    def save_cmd(self):
        '''Save the command line invocation to a shell script file.

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
        '''
        os.makedirs(self.save_dpath, exist_ok=True)
        out_fpath = os.path.join(
            self.save_dpath, "test_cmd.sh" if self.test_only else "train_cmd.sh"
        )
        # Log configuration instead of command for clean API
        config_fpath = out_fpath + '.config'
        utilities.dict_to_yaml(vars(self.config), config_fpath)
        logging.info(f"Configuration saved to {config_fpath}")


class BayerImageToImageNN(ImageToImageNN):
    '''Specialized class for Bayer pattern image-to-image inference.

    This class extends ImageToImageNN with Bayer-specific processing capabilities,
    including color space conversion and demosaicing operations.
    '''

    def __init__(self, config: InferenceConfig):
        super().__init__(config)

    def process_net_output(
            self,
            camRGB_images: torch.Tensor,
            rgb_xyz_matrix: torch.Tensor,
            gt_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        '''Process camRGB output s.t. it becomes closer to the final output.
        1. Match exposure if gt_images is provided
        2. Apply Lin. Rec. 2020 color profile
        3. Apply the Rec. 2100 Perceptual Quantizer (actually do this separately elsewhere)

        Args:
            camRGB_images (torch.Tensor): network output to convert
            rgb_xyz_matrix (torch.Tensor): camRGB to lin_rec2020 conversion matrices
            gt_images (Optional[torch.Tensor], optional): Ground-truth images to match exposure against (if provided). Defaults to None.
        '''
        from ..dependencies import raw_processing as rawproc
        match_gain = self.config.match_gain
        if gt_images is not None and match_gain == "output":
            camRGB_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=camRGB_images
            )
        output_images = rawproc.camRGB_to_lin_rec2020_images(
            camRGB_images, rgb_xyz_matrix
        )
        if (
                gt_images is not None and match_gain == "output"
        ):  # this is probably overkill
            output_images = rawproc.match_gain(
                anchor_img=gt_images, other_img=output_images
            )
        return output_images

    def autocomplete_config(self, config: InferenceConfig):
        super().autocomplete_config(config)
        config.preupsample = config.enable_preupsampling  # Legacy compatibility
