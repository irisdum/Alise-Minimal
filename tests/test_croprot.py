import pytest

from alise_minimal.data.dataset.croprot import CropRotDataset
from alise_minimal.data.dataset.sample_class import CDInput


@pytest.mark.trex
def test_pastiscddataset():
    PATH = "/home/ad/dumeuri/DeepChange/PASTIS_CD/PASTIS_CD/PASTIS_PT_MASK_SHARE2"
    item = CropRotDataset(dataset_path=PATH).__getitem__(0)
    assert isinstance(item, CDInput)
    assert item.year1.s2.sits.shape[0] == item.year1.s2.positions.shape[0]
    assert item.year2.s2.sits.shape[0] == item.year2.s2.positions.shape[0]
