"""File for custom collate functions.
Those functions allow to construct batch given numerous samples.
Custom functions need to be defined in this project
as we have created custom dataclass"""

import logging
from collections.abc import Iterable

import torch

from alise_minimal.data.batch_class import CDBInput, SITSBatch
from alise_minimal.data.dataset.sample_class import CDInput, ItemTensorMMDC

my_logger = logging.getLogger(__name__)


def custom_collateitem_mmdc(batch: list[ItemTensorMMDC]) -> SITSBatch:
    """
    If extended could work with other modality.
    FUsion which define batch contruction
    for numerous ItemTensorMMDC samples
    Parameters
    ----------
    batch :

    Returns
    -------

    """
    cld_mask = None
    if batch[0].s2 is not None:
        s2 = torch.stack([b.s2.sits for b in batch])
        s2_positions = torch.stack([b.s2.positions for b in batch])
        padd_s2 = torch.stack([b.s2.mask.padd_mask for b in batch]).bool()
        if batch[0].s2.mask.mask_cld is not None:
            cld_mask = torch.stack([b.s2.mask.mask_cld for b in batch])
        if batch[0].s2.mask.mask_scl is not None:
            slc_mask = torch.stack([b.s2.mask.mask_scl for b in batch])
            cld_mask[slc_mask == 0] = 1  # integrate no data in cloud mask
        else:
            my_logger.infor("No CLD MASK found ")
    else:
        s2 = None
        s2_positions = None
        padd_s2 = None

    return SITSBatch(
        sits=s2, positions=s2_positions, pad_mask=padd_s2, cld_mask=cld_mask
    )


def custom_collate_pastis_cd(batch: Iterable[CDInput]):
    """
    Create batch of CDInput samples
    Parameters
    ----------
    batch :

    Returns
    -------

    """
    year1: SITSBatch = custom_collateitem_mmdc([b.year1 for b in batch])
    year2: SITSBatch = custom_collateitem_mmdc([b.year2 for b in batch])
    label = torch.stack([b.raster for b in batch])
    mask_label = torch.stack([b.mask_raster for b in batch])
    return CDBInput(year1=year1, year2=year2, label=label, mask_label=mask_label)
