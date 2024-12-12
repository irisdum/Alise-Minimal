import torch

from alise_minimal.torch_model.sse import Unet, UnetConfig


def test_unet_forward():
    B, C, H, W = 2, 10, 64, 64
    unet_config = UnetConfig(
        inplanes=C,
        planes=64,
        encoder_widths=[64, 64, 64, 128],
        decoder_widths=[32, 32, 64, 128],
    )
    unet = Unet(unet_config)
    input = torch.rand(B, C, H, W)
    output = unet(input)
    assert output.shape == (B, 64, 64, 64)
