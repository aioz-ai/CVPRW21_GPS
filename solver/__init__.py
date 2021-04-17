"""
This code is modified from Hao Luo's repository.
Paper: Bag of Tricks and A Strong Baseline for Deep Person Re-identification
https://github.com/michuanhaohao/reid-strong-baseline
"""

from .build import make_optimizer, make_optimizer_with_center
from .lr_scheduler import WarmupMultiStepLR