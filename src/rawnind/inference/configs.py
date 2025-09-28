from dataclasses import dataclass, field
from typing import Optional, Union, List

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Clean API fields
    architecture: str = "unet"
    input_channels: int = 3
    output_channels: Optional[int] = None
    match_gain: str = "none"
    enable_preupsampling: bool = False
    device: Optional[Union[str, int]] = None
    tile_size: Optional[int] = None
    tile_overlap: Optional[int] = 32
    loss_function: Optional[str] = None
    filter_units: Optional[int] = None
    # Legacy/alternate fields
    arch: Optional[str] = None
    arch_enc: Optional[str] = None
    arch_dec: Optional[str] = None
    in_channels: Optional[int] = None
    funit: Optional[int] = None
    hidden_out_channels: Optional[int] = None
    bitstream_out_channels: Optional[int] = None
    preupsample: Optional[bool] = None
    loss: Optional[str] = None
    metrics: Optional[List[str]] = field(default_factory=list)
    test_only: Optional[bool] = False
    save_dpath: Optional[str] = None
    load_path: Optional[str] = None
    # Training-specific fields
    batch_size: Optional[int] = None
    crop_size: Optional[int] = None
    tot_steps: Optional[int] = None
    val_interval: Optional[int] = None
    patience: Optional[int] = None
    lr_multiplier: Optional[float] = None
    learning_rate: Optional[float] = None

    # Additional attributes for compatibility with base_inference.py
    expname: Optional[str] = None
    init_step: Optional[int] = None
    comment: Optional[str] = None
    fallback_load_path: Optional[str] = None
    continue_training_from_last_model_if_exists: Optional[bool] = False
    def __post_init__(self):
        # Alias clean API to legacy fields if not set
        if self.arch is None:
            self.arch = self.architecture
        if self.in_channels is None:
            self.in_channels = self.input_channels
        if self.funit is None and self.filter_units is not None:
            self.funit = self.filter_units
        if self.loss is None and self.loss_function is not None:
            self.loss = self.loss_function
        if self.preupsample is None:
            self.preupsample = self.enable_preupsampling
        # Alias output_channels if needed
        if hasattr(self, 'output_channels') and self.output_channels is not None:
            self.out_channels = self.output_channels
        # Alias crop_size if needed
        if hasattr(self, 'crop_size') and self.crop_size is not None:
            self.crop_size = self.crop_size
        # Alias batch_size if needed
        if hasattr(self, 'batch_size') and self.batch_size is not None:
            self.batch_size = self.batch_size
        # Alias tot_steps if needed
        if hasattr(self, 'tot_steps') and self.tot_steps is not None:
            self.tot_steps = self.tot_steps
        # Alias val_interval if needed
        if hasattr(self, 'val_interval') and self.val_interval is not None:
            self.val_interval = self.val_interval
        # Alias patience if needed
        if hasattr(self, 'patience') and self.patience is not None:
            self.patience = self.patience
        # Alias lr_multiplier if needed
        if hasattr(self, 'lr_multiplier') and self.lr_multiplier is not None:
            self.lr_multiplier = self.lr_multiplier
        # Alias learning_rate if needed
        if hasattr(self, 'learning_rate') and self.learning_rate is not None:
            self.learning_rate = self.learning_rate
