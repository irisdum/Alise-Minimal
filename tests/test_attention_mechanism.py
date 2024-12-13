import torch

from alise_minimal.torch_model.attention_mechanism import LearnedQMultiHeadAttention


def test_forward():
    d_model, n_q = 64, 10
    B, T = 8, 30
    ca = LearnedQMultiHeadAttention(n_head=4, d_k=32, d_in=d_model, n_q=n_q)
    X = torch.rand(B, T, d_model)
    pad_mask = torch.ones(B, T).bool()
    out = ca(X, pad_mask=pad_mask)
    assert out.shape == (B, n_q, d_model)
    pad_mask = torch.zeros(B, T).bool()
    out = ca(X, pad_mask=pad_mask)
    assert torch.sum(out) == 0
