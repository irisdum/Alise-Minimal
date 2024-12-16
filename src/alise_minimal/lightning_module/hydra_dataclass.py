from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CAWConfig:
    """ """

    _target_: Any = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    T_0: int = 2
    T_mult: int = 2


@dataclass
class OptimizerAdamConfig:
    """ """

    _target_: Any = torch.optim.Adam
