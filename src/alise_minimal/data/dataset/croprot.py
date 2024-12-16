import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from alise_minimal.data.dataset.sample_class import (
    CDInput,
    ItemTensorMMDC,
    MaskMod,
    OneMod,
    PaddingMMDC,
)


def from_dict2mask(input_dict) -> MaskMod:
    """

    Parameters
    ----------
    dict : with keys mask_cld,mask_nan and mask slc

    Returns
    -------

    """
    return MaskMod(mask_scl=input_dict["mask_slc"], mask_cld=input_dict["mask_cld"])


def from_dict2sits(input_dict: dict) -> ItemTensorMMDC:
    """

    Parameters
    ----------
    input_dict :

    Returns
    -------

    """
    mask = from_dict2mask(input_dict["mask"])
    one_mod = OneMod(sits=input_dict["sits"], positions=input_dict["doy"], mask=mask)
    return ItemTensorMMDC(s2=one_mod)


def from_dict2cdinput(input_dict: dict) -> CDInput:
    """

    Parameters
    ----------
    input_dict : a dictionary which contains all information of CropRot

    Returns
    -------

    """
    year1 = from_dict2sits(input_dict["year1"])
    year2 = from_dict2sits(input_dict["year2"])
    return CDInput(year1=year1, year2=year2, raster=input_dict["raster"])


class PASTISCDDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        dataset_name="dataset",
        max_len_s2: int = 60,
    ):
        """

        Parameters
        ----------
        dataset_path : path to dataset
        dataset_name : name of the csv which stores path to all samples
        max_len_s2 : the output length of each SITS
        """
        super().__init__()
        self.metadata = pd.read_csv(Path(dataset_path).joinpath(f"{dataset_name}.csv"))
        self.metadata.sort_index(inplace=True)
        self.id_patches = self.metadata["path"]
        self.len = len(self.id_patches)
        self.folder = dataset_path
        self.paddmmdc = PaddingMMDC(max_len_s2=max_len_s2)

    def __len__(self):
        return self.len

    def __getitem__(self, item: int) -> CDInput:
        """
        Mandatory method for Pytorch Dataset class
        Parameters
        ----------
        item : The number of the sample to output

        Returns
        -------

        """
        paths = self.id_patches[item]  # under the form of 129_3.pt
        load_sample: dict = torch.load(
            os.path.join(
                self.folder,
                paths,
            )
        )
        cd_sample = from_dict2cdinput(load_sample)
        mask_labels = cd_sample.raster[0, ...] != 0
        padded_sample = cd_sample.apply_padding(self.paddmmdc)
        padded_sample.mask_raster = mask_labels
        return padded_sample
