"""
Contains all class for temporal positional encoding
"""

import torch
from torch import Tensor
from torch import nn as nn


class PositionalEncoder(nn.Module):
    """
    Traditional Positional encoding as defined by Vaswani 2017.
    Implementation inspired by
    https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """

    def __init__(self, d: int, T: int = 1000, offset: int = 0):
        """

        Parameters
        ----------
        d : Number of features
        T : The scaling constant
        offset :
        """
        super().__init__()
        self.d = d
        self.T = T
        self.denom = torch.pow(
            T, 2 * (torch.arange(offset, offset + d).float() // 2) / d
        )

    def forward(self, batch_positions: Tensor) -> Tensor:
        """

        Parameters
        ----------    assert False
        batch_positions : (B,T) 2D tensor containing positions.

        Returns
        -------
        a tensor of size B,T,C with C corresponding to self.d
        """
        self.denom = self.denom.to(batch_positions.device)
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )  # B x T x C
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        return sinusoid_table
