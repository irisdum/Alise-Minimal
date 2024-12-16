from dataclasses import dataclass

import torch


@dataclass
class Stats:
    """
    Store stats
    """

    median: list
    qmin: list
    qmax: list


@dataclass
class OneTransform:
    """
    Store transform and its associated stats
    """

    transform: torch.nn.Module | None = None
    stats: Stats | None = None
