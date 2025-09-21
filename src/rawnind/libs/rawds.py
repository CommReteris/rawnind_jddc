"""
Raw dataset handlers.

Returns x (clean), y (noisy). Bayer (black-white point) or
ProfiledRGB (Lin. Rec2020).

Loaders from whole images are deprecated (too slow). To bring up to date they would need to handle:
- bayer_only
"""

import random
import logging
import os
import sys
import math
import time
import unittest
from typing import Literal, NamedTuple, Optional, Union
from typing import TypedDict
import tqdm

import torch

from . import pt_helpers, utilities
from . import raw
from . import rawproc
from . import arbitrary_proc_fun

BREAKPOINT_ON_ERROR = True

COLOR_PROFILE = "lin_rec2020"
LOG_FPATH = os.path.join("logs", os.path.basename(__file__) + ".log")

MAX_MASKED: float = 0.5  # Must ensure that we don't send a crop with this more than this many masked pixels
MAX_RANDOM_CROP_ATTEMPS = 10

MASK_MEAN_MIN = 0.8  # 14+11+1 = 26 images out of 1145 = 2.3 %
ALIGNMENT_MAX_LOSS = (
    0.035  # eliminates 6+3+2 + 1+4+2+6+3 = 27 images out of 1145 = 2.4 %
)
OVEREXPOSURE_LB = 0.99

TOY_DATASET_LEN = 25  # debug option