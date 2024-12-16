import torch
import torch.nn.functional as F
from torch import Tensor


def apply_padding(
    allow_padd: bool, max_len: int, sits: Tensor, positions: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """

    Parameters
    ----------
    allow_padd : if True returns padded output
    max_len : the maximum size of the input
    t : the input SITS length
    sits : Tensor with Time as first dimension ex: Tensor of size (T,C,H,W)
    positions : Tensor of size (T)

    Returns
    -------
    if allow_padd is True:
    [sits (max_len, C,H,W), positions (max_len), padd_index (max_len)]
    padd_index is set to True when the element corresponds to a padded elements
    if allow_padd is False
    [sits (T,C,H,W), positions (T), None]
    """
    t = sits.shape[0]
    if allow_padd:
        padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
        padd_doy = (0, max_len - t)
        sits = F.pad(sits, padd_tensor)
        positions = F.pad(positions, padd_doy)
        padd_index = torch.zeros(max_len)
        padd_index[t:] = 1
        padd_index = padd_index.bool()
    else:
        padd_index = None

    return sits, positions, padd_index
