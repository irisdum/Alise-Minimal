import torch

from alise_minimal.torch_model.alise import (
    ALISEConfigBuild,
    TransformerConfig,
    TransformerLayerConfig,
    build_alise,
)
from alise_minimal.torch_model.attention_mechanism import ConfigLQMHA
from alise_minimal.torch_model.sse import UnetConfig


def test_forward():
    B, T, C, H, W = 2, 10, 3, 64, 64
    d_model = 64
    pe_T = 3000
    nq = 10
    temp_proj_config = ConfigLQMHA(n_head=4, d_k=32, d_in=d_model, n_q=nq)
    unet_config = UnetConfig(
        inplanes=C,
        planes=d_model,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
    )
    transformer_layer_config = TransformerLayerConfig(
        d_model=d_model, nhead=4, dim_feedforward=128
    )
    transformer_config = TransformerConfig(
        layer_config=transformer_layer_config, num_layers=2
    )
    alise_build_config = ALISEConfigBuild(
        unet_config=unet_config,
        transformer_config=transformer_config,
        temp_proj_config=temp_proj_config,
        pe_T=pe_T,
    )
    alise = build_alise(alise_build_config)
    input = torch.rand(B, T, C, H, W)
    positions = torch.rand(B, T)
    pad_mask = torch.zeros(B, T).bool()
    output = alise(sits=input, positions=positions, pad_mask=pad_mask)
    assert output.shape == (B, nq, d_model, H, W)
