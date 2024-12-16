from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from alise_minimal.data.dataset.utils import apply_padding


@dataclass
class MaskMod:
    """
    Class to store masks information.
    CLD and SCL correspond to Sentinel 2 L2A Sen2corr layers.
    """

    mask_cld: Tensor | None = None  # T,C,H,W
    mask_scl: Tensor | None = None  # T,C,H,W
    padd_mask: Tensor | None = None  # T  # 1 if the date has been padded

    def merge_mask(self) -> Tensor:
        cld_mask = self.mask_cld == 1
        # my_logger.debug(f"mask cld in fun {cld_mask[0, :, 0, 0]}")
        nan_mask = self.mask_scl == 0
        cld_mask_scl = torch.logical_and(self.mask_scl > 6, self.mask_scl < 11)
        cld_mask_scl = torch.logical_or(cld_mask_scl, self.mask_scl < 2)
        cld_mask_scl = torch.logical_or(cld_mask_scl, self.mask_scl == 3)
        cld_mask = torch.logical_or(cld_mask, cld_mask_scl)
        return torch.logical_or(cld_mask, nan_mask)


@dataclass
class PaddingMMDC:
    """
    Class which stores for each modality the maximum length of the time series
    """

    max_len_s2: int | None = None
    max_len_s1_asc: int | None = None
    max_len_s1_desc: int | None = None
    max_len_agera5: int | None = None


@dataclass
class OneMod:
    """
    Class associated to a sample of a specific modality
     Parameters
    ----------
    sits: Tensor of size (T C H W)
    positions: Tensor of size (T)
    """

    sits: Tensor
    positions: Tensor
    mask: MaskMod = None

    def apply_padding(self, max_len: int, allow_padd=True):
        t = self.sits.shape[0]
        sits, positions, padd_index = apply_padding(
            allow_padd, max_len, self.sits, self.positions
        )
        if self.mask.mask_cld is not None:
            padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
            mask_cld = F.pad(self.mask.mask_cld, padd_tensor)
        else:
            mask_cld = None
        if self.mask.mask_scl is not None:
            padd_tensor = (0, 0, 0, 0, 0, 0, 0, max_len - t)
            mask_slc = F.pad(self.mask.mask_scl, padd_tensor)
        else:
            mask_slc = None
        return OneMod(
            sits=sits,
            positions=positions,
            mask=MaskMod(padd_mask=padd_index, mask_cld=mask_cld, mask_scl=mask_slc),
        )


@dataclass
class ItemTensorMMDC:
    """
    Class which stores multimodal SITS samples
    """

    s2: OneMod | None = None
    s1_asc: OneMod | None = None
    s1_desc: OneMod | None = None
    dem: OneMod | None = None
    agera5: OneMod | None = None

    def apply_padding(self, paddmmdc: PaddingMMDC):
        if self.s1_asc is not None:
            s1_asc = self.s1_asc.apply_padding(paddmmdc.max_len_s1_asc)
        else:
            s1_asc = None
        if self.s1_desc is not None:
            s1_desc = self.s1_desc.apply_padding(paddmmdc.max_len_s1_desc)
        else:
            s1_desc = None
        if self.s2 is not None:
            s2 = self.s2.apply_padding(paddmmdc.max_len_s2)
        else:
            s2 = None
        if self.agera5 is not None:
            agera5 = self.agera5.apply_padding(paddmmdc.max_len_agera5)
        else:
            agera5 = None
        return ItemTensorMMDC(
            s2=s2, s1_asc=s1_asc, s1_desc=s1_desc, dem=self.dem, agera5=agera5
        )


@dataclass
class CDInput:
    """
    Class which stores two (mono or multimodal SITS), that we aim to compare.
    """

    year1: ItemTensorMMDC
    year2: ItemTensorMMDC
    raster: Tensor  # dim (H,W)
    mask_raster: Tensor | None = None  # dim (H,W)

    def apply_padding(self, paddmmdc: PaddingMMDC):
        return CDInput(
            year1=self.year1.apply_padding(paddmmdc),
            year2=self.year2.apply_padding(paddmmdc),
            raster=self.raster,
            mask_raster=self.mask_raster,
        )
