import torch

from alise_minimal.torch_model.patch_embedding import PatchEmbedding
from alise_minimal.torch_model.sse import Unet, UnetConfig
from alise_minimal.torch_model.temporal_positional_encoder import PositionalEncoder


def test_forward():
    B, T, C, H, W = 2, 30, 10, 64, 64
    d_model = 64
    unet_config = UnetConfig(
        inplanes=C,
        planes=d_model,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
    )
    unet = Unet(unet_config)
    tpe = PositionalEncoder(d=d_model)
    input = torch.rand(B, T, C, H, W)
    positions = torch.rand(B, T)
    patch_embed = PatchEmbedding(sse=unet, tpe=tpe)
    output = patch_embed(input, positions)
    assert output.shape == (B, T, d_model, H, W)
