"""Definition of class associated to some batches"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SegBatch:
    """
    sits: B T C H W
    positions: B T
    pad_mask: B T
    labels: B H W
    """

    sits: Tensor
    positions: Tensor
    pad_mask: Tensor
    labels: Tensor
