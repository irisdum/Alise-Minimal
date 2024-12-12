"""
Architecture which combines SSE with temporal PE
"""

from einops import rearrange
from torch import Tensor, nn


class PatchEmbedding(nn.Module):
    """
    Integrate position information on feature map extracted
    by the spectro-spatial encoder (SSE)
    """

    def __init__(self, sse: nn.Module, tpe: nn.Module):
        """

        Parameters
        ----------
        sse : SPectro spatial encoder
        tpe : Temporal positional Encoder
        merge_opt : indicate how temporal information is integrated to feature maps
        """
        super().__init__()
        self.tpe = tpe
        self.sse = sse

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """

        Parameters
        ----------
        x : time series of size B,T,C,H,W
        positions : B,T
        Returns
        -------
        Tensor, of size B,T,D,H,W, with D equals to C if merge_opt=add
        """
        B, T, C, H, W = x.shape
        x = self.sse(rearrange(x, "B T C H W -> (B T) C H W "))
        x = rearrange(x, "(B T) C H W -> B T C H W", B=B, T=T, H=H, W=W)
        encoded_positions = self.tpe(positions)
        encoded_positions = rearrange(encoded_positions, "b t c -> b t c 1 1")
        return x + encoded_positions
