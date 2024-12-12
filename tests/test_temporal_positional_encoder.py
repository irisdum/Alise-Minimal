import torch

from alise_minimal.torch_model.temporal_positional_encoder import PositionalEncoder


def test_forward():
    B, T, d_model = 2, 30, 32
    input = torch.rand(B, T)
    pe = PositionalEncoder(d=d_model, T=3000)
    output = pe(input)
    assert output.shape == (B, T, d_model)
