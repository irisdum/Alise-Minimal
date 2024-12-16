import pytest

from alise_minimal.data.dataset.croprot import CropRotDataset
from alise_minimal.data.dataset.sample_class import CDInput


@pytest.mark.trex
def test_pastiscddataset():
    PATH = "/home/ad/dumeuri/DeepChange/PASTIS_CD/PASTIS_CD/PASTIS_PT_MASK_SHARE2"
    ds = CropRotDataset(dataset_path=PATH)
    assert isinstance(ds.__getitem__(0), CDInput)
