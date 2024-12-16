import pytest
import torch
from alise_minimal.data.batch_class import CDBInput
from alise_minimal.data.datamodule.croprot_datamodule import CropRotDataModule


@pytest.mark.trex
def test_croprot_datamodule():
    PATH = "/home/ad/dumeuri/DeepChange/PASTIS_CD/PASTIS_CD/PASTIS_PT_MASK_SHARE2"
    path_dir_csv = "/home/ad/dumeuri/DeepChange/MMDC_OE"

    datamodule = CropRotDataModule(dataset_path=PATH,
                                   path_dir_csv=path_dir_csv,
                                   batch_size=2)
    batch: CDBInput = next(iter(datamodule.train_dataloader()))
    assert batch.year1.sits.shape[0] == 2
    assert batch.year1.sits.shape[1] == batch.year1.positions.shape[1]
    assert torch.min(batch.year1.positions) - torch.max(
        batch.year2.positions) < 2 * 365
