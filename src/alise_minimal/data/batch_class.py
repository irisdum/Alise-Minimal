"""Definition of class associated to some batches"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class SegBatch:
    """
    sits: B T C H W
    positions: B T
    pad_mask: B T boolean Tensor. True means the value will be
    *ignored* in the attention
    labels: B H W (long Tensor)
    """

    sits: Tensor
    positions: Tensor
    pad_mask: Tensor
    labels: Tensor
