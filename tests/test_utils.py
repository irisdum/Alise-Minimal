import torch

from alise_minimal.data.dataset.sample_class import (
    CDInput,
    ItemTensorMMDC,
    MaskMod,
    OneMod,
    PaddingMMDC,
)
from alise_minimal.data.dataset.utils import apply_padding


def test_apply_padding():
    T, C, H, W = 10, 3, 16, 16
    max_len = 20
    sits, positions, padd_mask = apply_padding(
        allow_padd=True,
        max_len=max_len,
        sits=torch.rand(T, C, H, W),
        positions=torch.rand(T),
    )
    assert sits.shape[0] == max_len
    assert positions.shape[0] == max_len
    assert padd_mask.shape[0] == max_len
    assert padd_mask[-max_len + T] is True
    print(padd_mask)


def fake_sample(T, C, H, W):
    sits = torch.rand(T, C, H, W)
    positions = torch.rand(T)
    val_mask = torch.ones(T, C, H, W).bool()
    mask = MaskMod(mask_cld=val_mask, mask_scl=val_mask, padd_mask=None)
    one_mode_sample = OneMod(sits=sits, positions=positions, mask=mask)
    mm_sample = ItemTensorMMDC(s2=one_mode_sample)
    return mm_sample


def test_cd_input_apply_padding():
    T, C, H, W = 10, 3, 32, 32
    max_len = 20
    year_1 = fake_sample(T, C, H, W)
    year_2 = fake_sample(T, C, H, W)
    cd_sample = CDInput(
        year1=year_1,
        year2=year_2,
        labels=torch.randn(H, W),
        mask_labels=torch.ones(H, W),
    )
    padd = PaddingMMDC(max_len_s2=max_len)
    padded_cd_sample = cd_sample.apply_padding(paddmmdc=padd)
    assert padded_cd_sample.year1.s2.sits.shape == (max_len, C, H, W)
    assert padded_cd_sample.year2.s2.positions.shape[0] == max_len
