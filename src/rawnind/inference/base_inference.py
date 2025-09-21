"""Base inference class for model inference operations.

This module provides the base class for inference operations, extracted
from the original ImageToImageNN class in abstract_trainer.py.

Extracted from abstract_trainer.py as part of the codebase refactoring.
"""

import logging
import os
import sys

import torch

from ..dependencies.json_saver import YAMLSaver
from ..dependencies.pt_losses import metrics
# Import from dependencies package (will be moved later)
from ..dependencies.pytorch_helpers import get_device
from ..dependencies.utilities import dict_to_yaml


class BaseInference:
    """Base class for model inference operations.

    This class provides the core functionality for model inference,
    including argument parsing, device setup, and model management.
    It serves as the foundation for specific inference implementations.
    """

    CLS_CONFIG_FPATHS = [
        os.path.join("config", "test_reserve.yaml"),
    ]

    def __init__(self, test_only: bool = True, **kwargs):
        """Initialize the base inference class.

        Args:
            test_only: Boolean flag for evaluation-only mode
            **kwargs: Keyword arguments that can override configuration
        """
        # Skip if already initialized (for multiple inheritance)
        if hasattr(self, "device"):
            return

        # Initialize basic attributes
        self.save_dpath: str = None  # type: ignore
        self.test_only = test_only

        # Get arguments (simplified for inference)
        args = self.get_args()
        if "preset_args" in kwargs:
            vars(args).update(kwargs["preset_args"])
            vars(self).update(kwargs["preset_args"])
        self.__dict__.update(vars(args))

        # Set up device
        self.device = get_device(args.device)
        if "cuda" in str(self.device):
            torch.backends.cudnn.benchmark = True

        # Set up logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            filename=os.path.join(
                self.save_dpath, f"{'test' if self.test_only else 'train'}.log"
            ),
            datefmt="%Y-%m-%d %H:%M:%S",
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.DEBUG if getattr(self, 'debug_options', False) else logging.INFO,
            filemode="w",
        )
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(" ".join(sys.argv))
        logging.info(f"PID: {os.getpid()}")

        # Create save directory
        os.makedirs(os.path.join(self.save_dpath, "saved_models"), exist_ok=True)

        # Initialize metrics
        metrics_dict = {}
        for metric in getattr(self, 'metrics', []):
            metrics_dict[metric] = metrics[metric]()
        self.metrics = metrics_dict

    def get_args(self, ignore_unknown_args: bool = False):
        """Parse command-line arguments for inference.

        This method sets up argument parsing for inference operations.
        Simplified version of the original get_args method.

        Args:
            ignore_unknown_args: If True, unknown arguments are ignored

        Returns:
            argparse.Namespace: Object containing parsed arguments
        """
        # Import configargparse for argument parsing
        import configargparse

        parser = configargparse.ArgumentParser(
            description="Base inference argument parser",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            default_config_files=self.CLS_CONFIG_FPATHS,
        )
        self.add_arguments(parser)

        if ignore_unknown_args:
            return parser.parse_known_args()[0]
        return parser.parse_args()

    def add_arguments(self, parser):
        """Register command-line arguments for inference.

        This method defines the basic arguments needed for inference operations.
        Subclasses should override this to add specific arguments.

        Args:
            parser: ConfigArgParse parser to register arguments with
        """
        parser.add_argument(
            "--config",
            is_config_file=True,
            dest="config",
            required=False,
            help="config in yaml format",
        )
        parser.add_argument(
            "--device", type=int, help="CUDA device number (-1 for CPU)"
        )
        parser.add_argument(
            "--save_dpath",
            help="Save directory for results",
        )

    def save_args(self, args):
        """Save command-line arguments to a YAML file for reproducibility.

        Args:
            args: Namespace object containing command-line arguments
        """
        os.makedirs(args.save_dpath, exist_ok=True)
        out_fpath = os.path.join(args.save_dpath, "args.yaml")
        dict_to_yaml(vars(args), out_fpath)

    def save_cmd(self):
        """Save the command line invocation to a shell script file."""
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
