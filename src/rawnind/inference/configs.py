from dataclasses import dataclass, field
from typing import Optional, Union, List

@dataclass
class InferenceConfig:
    """Configuration for inference."""
    architecture: str = "unet"
    input_channels: int = 3
    match_gain: str = "none"
    enable_preupsampling: bool = False
    device: Optional[Union[str, int]] = None
    tile_size: Optional[int] = None
    tile_overlap: Optional[int] = 32