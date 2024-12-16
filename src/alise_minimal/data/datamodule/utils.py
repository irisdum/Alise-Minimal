"""
FIle for functions relevant to transform applied on batch
"""

from pathlib import Path

import pandas as pd
import torch
from einops import rearrange
from torch import Tensor, nn

from alise_minimal.data.datamodule.transform import Clip, S2Normalize
from alise_minimal.data.datamodule.transform_class import OneTransform, Stats


def read_csv_stat(path_csv) -> Stats:
    """
    Read the stats stored in a casv
    Parameters
    ----------
    path_csv :

    Returns
    -------

    """
    assert path_csv.exists(), f"No file found at {path_csv}"
    df_stats = pd.read_csv(path_csv, sep=",", index_col=0)
    return Stats(
        median=df_stats.loc["med"].tolist(),
        qmin=df_stats.loc["qmin"].tolist(),
        qmax=df_stats.loc["qmax"].tolist(),
    )


def load_transform_one_mod(path_dir_csv: str, mod: str = "s2") -> OneTransform:
    """

    Parameters
    ----------
    path_dir_csv : path where "dataset_{mod}.csv" is stored,
     which contains modality stats
    mod : modality, implemented only of S2

    Returns
    -------

    """
    if mod != "s2":
        raise NotImplementedError
    path_csv = Path(path_dir_csv).joinpath(f"dataset_{mod}.csv")
    stats = read_csv_stat(path_csv)
    scale = tuple([float(x) - float(y) for x, y in zip(stats.qmax, stats.qmin)])
    return OneTransform(
        torch.nn.Sequential(
            Clip(qmin=stats.qmin, qmax=stats.qmax),
            S2Normalize(med=stats.median, scale=scale),
        ),
        stats,
    )


def apply_transform_basic(batch_sits: Tensor, transform: nn.Module) -> Tensor:
    """
    Reshape before applying transform
    Parameters
    ----------
    batch_sits :
    transform :

    Returns
    -------

    """
    b, *_ = batch_sits.shape
    batch_sits = rearrange(batch_sits, " b t c h w -> c (b t) h w")
    batch_sits = transform(batch_sits)
    batch_sits = rearrange(batch_sits, "c (b t )  h w -> b t c h w", b=b)
    return batch_sits
