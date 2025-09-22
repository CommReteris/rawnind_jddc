"""Utility classes for saving and loading training and evaluation results.

This module provides classes for persistently storing model training and evaluation 
results in JSON or YAML format. It supports tracking metrics over time, automatically
identifying best-performing models, and retrieving results from specific training steps.

Key features:
- Persistent storage of training metrics
- Automatic tracking of best performance for each metric
- Support for both minimization and maximization metrics
- Warm-up period where results are not considered for "best" status
- Both JSON and YAML format support through separate classes

The module is primarily used for:
1. Tracking model performance during training
2. Finding the best-performing model configurations
3. Analyzing training progress over time
4. Enabling checkpoint selection based on various metrics

Usage example:
    # Create a saver for tracking results
    saver = JSONSaver('results.json')
    
    # Add results for training step 1000
    saver.add_res(
        step=1000,
        res={'loss': 0.245, 'accuracy': 0.912},
        minimize={'loss': True, 'accuracy': False}  # Lower loss is better, higher accuracy is better
    )
    
    # Later, retrieve the best step for a specific metric
    best_step = saver.get_best_step('loss')
    best_results = saver.get_best_step_results('loss')
"""

from typing import Optional, Set, Dict, Any, Union
import sys
import os

from . import utilities


class JSONSaver:
    """Class for saving and loading training/evaluation results in JSON format.
    
    This class provides functionality to:
    - Save training/evaluation metrics at different steps
    - Track the best values for each metric
    - Retrieve the best-performing steps for specific metrics
    - Load previously saved results for continued training
    
    The class automatically tracks "best" values for each metric, considering whether
    lower values are better (minimize=True) or higher values are better (minimize=False).
    It can also ignore initial steps during a warm-up period when tracking best values.
    
    Attributes:
        jsonfpath: Path to the JSON file for storing results
        best_key_str: Key string for storing best step values (e.g., "best_step")
        results: Dictionary containing all saved results
        warmup_nsteps: Number of initial steps to ignore when tracking best values
    """

    def __init__(
            self,
            jsonfpath: str,
            step_type: str = ["step", "epoch"][0],
            default: Optional[Dict[str, Any]] = None,
            warmup_nsteps: int = 0,
    ):
        """Initialize the JSONSaver with file path and configuration.
        
        Args:
            jsonfpath: Path to the JSON file for storing results
            step_type: Type of step counting ('step' or 'epoch')
            default: Default data to use if file doesn't exist or is empty
            warmup_nsteps: Number of initial steps to ignore when tracking best values
            
        Raises:
            AssertionError: If jsonfpath is a directory
        """
        assert not os.path.isdir(jsonfpath), "JSON path must be a file, not a directory"

        # Initialize default data structure if not provided
        if default is None:
            default = {"best_val": dict()}

        # Create key string for best step/epoch values
        self.best_key_str = "best_{}".format(step_type)  # best step/epoch #
        self.jsonfpath = jsonfpath

        # Load existing results or use default
        self.results = self._load(jsonfpath, default=default)

        # Ensure best_key_str exists in results
        if self.best_key_str not in self.results:
            self.results[self.best_key_str] = dict()

        # Set warmup period
        self.warmup_nsteps = warmup_nsteps

    def _load(self, fpath: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Load results from the JSON file.
        
        Args:
            fpath: Path to the JSON file
            default: Default data to use if file doesn't exist or is empty
            
        Returns:
            Dictionary containing loaded results or default values
        """
        return utilities.jsonfpath_load(fpath, default=default)

    def add_res(
            self,
            step: int,
            res: Dict[str, Any],
            minimize: Union[bool, Dict[str, bool]] = True,
            write: bool = True,
            val_type: Optional[type] = float,
            epoch: Optional[int] = None,
            rm_none: bool = False,
            key_prefix: str = "",
    ) -> None:
        """Add results for a specific training step and update best values.
        
        This method adds result metrics for a specific training step and automatically
        updates the best values for each metric. It can handle both minimization metrics
        (where lower is better, like loss) and maximization metrics (where higher is
        better, like accuracy).
        
        Args:
            step: The training step number
            res: Dictionary of metric names and values to record
            minimize: Whether lower values are better (True) or higher values are better (False).
                      Can be a boolean applied to all metrics or a dictionary mapping metric
                      names to minimize flags.
            write: Whether to write results to file after adding them
            val_type: Type to convert values to (default: float)
            epoch: Alternative to step (for backwards compatibility)
            rm_none: Whether to ignore zero values when tracking best values
            key_prefix: Prefix to add to all metric names in res
            
        Raises:
            ValueError: If neither step nor epoch is specified
            
        Notes:
            - If a metric's value is None, it will be skipped with a warning
            - Steps before warmup_nsteps are not considered for best value tracking
            - List values are stored but not tracked for best values
        """
        # Handle step/epoch parameter (epoch is an alias for step)
        if epoch is not None and step is None:
            step = epoch
        elif (epoch is None and step is None) or step is None or epoch is not None:
            raise ValueError("JSONSaver.add_res: Must specify either step or epoch")

        # Initialize results dictionary for this step if it doesn't exist
        if step not in self.results:
            self.results[step] = dict()

        # Add prefix to metric names if specified
        if key_prefix != "":
            res_ = dict()
            for akey, aval in res.items():
                res_[key_prefix + akey] = aval
            res = res_

        # Process each metric in the results
        for akey, aval in res.items():
            # Skip None values with warning
            if aval is None:
                print(f"JSONSaver.add_res warning: missing value for {akey}")
                continue

            # Convert value to specified type if needed
            if val_type is not None:
                aval = val_type(aval)

            # Store the value for this step
            self.results[step][akey] = aval

            # Skip list values for best tracking (can't compare)
            if isinstance(aval, list):
                continue

            # Skip zero values if rm_none is True
            if rm_none and aval == 0:
                continue

            # Skip steps before warmup period for best tracking
            if step < self.warmup_nsteps:
                continue

            # Determine whether to minimize this metric
            minimize_this = minimize
            if isinstance(minimize, dict):
                minimize_this = minimize.get(akey, True)

            # Initialize best value if this is the first occurrence
            if akey not in self.results["best_val"]:
                self.results["best_val"][akey] = aval

            # Initialize best step if this is the first occurrence
            if akey not in self.results[self.best_key_str]:
                self.results[self.best_key_str][akey] = step

            # Update best value and step if this result is better
            is_better = ((self.results["best_val"][akey] > aval) and minimize_this) or (
                    (self.results["best_val"][akey] < aval) and not minimize_this
            )
            if is_better:
                self.results[self.best_key_str][akey] = step
                self.results["best_val"][akey] = aval

        # Write to file if requested
        if write:
            self.write()

    def write(self) -> None:
        """Write the current results to the JSON file.
        
        This method serializes the current results dictionary to the JSON file
        specified during initialization. It uses utilities.dict_to_json to handle
        the serialization.
        """
        utilities.dict_to_json(self.results, self.jsonfpath)

    def get_best_steps(self) -> Set[int]:
        """Get the set of all steps that are best for at least one metric.
        
        Returns:
            A set of step numbers, where each step is the best for at least one metric
        """
        return set(self.results[self.best_key_str].values())

    def get_best_step(self, akey: str) -> int:
        """Get the step number that has the best value for a specific metric.
        
        Args:
            akey: The name of the metric to get the best step for
            
        Returns:
            The step number with the best value for the specified metric
            
        Raises:
            KeyError: If the metric name is not found in the best steps dictionary
        """
        return self.results[self.best_key_str][akey]

    def get_best_step_results(self, akey: str) -> Dict[str, Any]:
        """Get all results from the step that has the best value for a specific metric.
        
        This is useful for retrieving all metrics from the step that performed best
        on a particular metric.
        
        Args:
            akey: The name of the metric to get the best step results for
            
        Returns:
            Dictionary of all metrics from the best step for the specified metric
            
        Raises:
            KeyError: If the metric name is not found or the best step is not in results
        """
        return self.results[self.get_best_step(akey)]

    def is_empty(self) -> bool:
        """Returns True if there are no saved results."""
        return len(self.results["best_val"]) == 0


class YAMLSaver(JSONSaver):
    """Class for saving and loading training/evaluation results in YAML format.
    
    This class extends JSONSaver to use YAML format instead of JSON for storing results.
    YAML provides better human readability and supports more complex data structures,
    making it easier to manually inspect and edit result files when needed.
    
    The class inherits all functionality from JSONSaver but overrides the file loading
    and writing methods to use YAML instead of JSON. All other behavior (tracking best
    values, adding results, retrieving best steps) remains the same.
    
    Usage is identical to JSONSaver, just with YAML file output:
    
        saver = YAMLSaver('results.yaml')
        saver.add_res(step=1000, res={'loss': 0.245})
    """

    def __init__(
            self,
            jsonfpath: str,
            step_type: str = ["step", "epoch"][0],
            default: Optional[Dict[str, Any]] = None,
            warmup_nsteps: int = 0,
    ):
        """Initialize the YAMLSaver with file path and configuration.
        
        Args:
            jsonfpath: Path to the YAML file for storing results
            step_type: Type of step counting ('step' or 'epoch')
            default: Default data to use if file doesn't exist or is empty
            warmup_nsteps: Number of initial steps to ignore when tracking best values
            
        Note:
            The parameter is still named 'jsonfpath' for compatibility with JSONSaver,
            but it refers to a YAML file path in this class.
        """
        super().__init__(
            jsonfpath, step_type=step_type, default=default, warmup_nsteps=warmup_nsteps
        )

    def _load(self, fpath: str, default: Dict[str, Any]) -> Dict[str, Any]:
        """Load results from the YAML file.
        
        This method overrides JSONSaver._load to use YAML instead of JSON.
        
        Args:
            fpath: Path to the YAML file
            default: Default data to use if file doesn't exist or is empty
            
        Returns:
            Dictionary containing loaded results or default values
            
        Note:
            Uses error_on_404=False to silently use default when file is not found,
            rather than raising an error.
        """
        result = utilities.load_yaml(fpath, error_on_404=False)
        return result if result is not None else default

    def write(self) -> None:
        """Write the current results to the YAML file.
        
        This method overrides JSONSaver.write to use YAML instead of JSON.
        It serializes the current results dictionary to the YAML file
        specified during initialization.
        """
        utilities.dict_to_yaml(self.results, self.jsonfpath)
