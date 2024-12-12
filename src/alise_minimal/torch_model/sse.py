"""
File which contains relevant spectral spatial encoder. These architectures process images and do not take into account any temporal dimension.
"""

import torch
from typing_extensions import Literal

from torch import nn as nn, Tensor


class Unet(nn.Module):
    """
    Inspired by https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py
    """

    def __init__(
        self,
        inplanes: int,
        planes: int,
        encoder_widths: list,
        decoder_widths: list,
        encoder_norm: Literal["group", "batch"] = "group",
        padding_mode: str = "reflect",
        decoding_norm: Literal["group", "batch"] = "group",
        return_maps: bool = False,
        str_conv_k: int = 4,
        str_conv_s: int = 2,
        str_conv_p: int = 1,
        skip_conv_norm: Literal["group", "batch"] = "group",
    ):
        """

        Parameters
        ----------
        inplanes : input number of channels
        planes :
        encoder_widths : list of the number of channels in the Unet encoder
        decoder_widths : list of the number of channels in the Unet decoder
        encoder_norm : normalisation type description
        padding_mode : padding type as defined in Conv2d
        decoding_norm : normalisation in the decoder
        return_maps : wether all intermetdiate eafture maps are returned
        str_conv_k : kernel size
        str_conv_s : stride size
        str_conv_p : padding size
        skip_conv_norm : normalisation in the skip connections
        """
        super().__init__()
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths

        self.decoder_widths = decoder_widths
        self.in_conv = ConvBlock(
            nkernels=[inplanes] + [encoder_widths[0], encoder_widths[0]],
            norm=encoder_norm,
            padding_mode=padding_mode,
        )
        self.out_conv = ConvBlock(
            nkernels=[decoder_widths[0], planes],
            norm=decoding_norm,
            padding_mode=padding_mode,
        )
        self.n_stages = len(encoder_widths)
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=encoder_widths[i],
                d_out=encoder_widths[i + 1],
                k=str_conv_k,
                s=str_conv_s,
                p=str_conv_p,
                norm=encoder_norm,
                padding_mode=padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            [
                UpConvBlock(
                    d_in=decoder_widths[i],
                    d_out=decoder_widths[i - 1],
                    d_skip=encoder_widths[i - 1],
                    k=str_conv_k,
                    s=str_conv_s,
                    p=str_conv_p,
                    norm=decoding_norm,
                    padding_mode=padding_mode,
                    skip_conv_norm=skip_conv_norm,
                )
                for i in range(self.n_stages - 1, 0, -1)
            ]
        )

    def forward(self, input: Tensor):
        """

        Parameters
        ----------
        input : Tensor (B,C,H,W)

        Returns
        -------
        either a Tensor of size (B,C,H,W)  or if returns_map is set to True a Tensor of size
        B,C,H,W and well as a list of all intermediate feature maps
        """
        dtype = input.dtype
        out = self.in_conv(input)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i](feature_maps[-1])
            feature_maps.append(out)
            # print(out.shape)
        if self.return_maps:
            maps = [out]
        # print([out.shape for out in feature_maps])
        for i in range(self.n_stages - 1):
            skip = feature_maps[-(i + 2)]
            #  print(skip.shape, out.shape)
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                maps.append(out)
            #            out = rearrange(out, "b c h w -> b h w c")
        out = self.out_conv(out)
        if self.return_maps:
            return out.to(dtype), maps.to(dtype)

        return out.to(dtype)


class UpConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        final_out=None,
        norm="batch",
        d_skip=None,
        padding_mode="reflect",
        skip_conv_norm="batch",
    ):
        super().__init__()
        d = d_out if d_skip is None else d_skip
        if skip_conv_norm == "batch":
            skip_norm_begin = nn.BatchNorm2d(d)
            skip_norm_end = nn.BatchNorm2d(d_out)
        else:
            skip_norm_begin = nn.GroupNorm(num_groups=4, num_channels=d)
            skip_norm_end = nn.GroupNorm(num_groups=4, num_channels=d_out)
        self.skip_conv = nn.Sequential(
            nn.Conv2d(in_channels=d, out_channels=d, kernel_size=1),
            skip_norm_begin,
            nn.ReLU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=d_in,
                out_channels=d_out,
                kernel_size=k,
                stride=s,
                padding=p,
            ),
            skip_norm_end,
            nn.ReLU(),
        )
        self.conv1 = ConvLayer(
            nkernels=[d_out + d, d_out], norm=norm, padding_mode=padding_mode
        )
        if final_out is None:
            final_out = d_out

        self.conv2 = ConvLayer(
            nkernels=[d_out, final_out], norm=norm, padding_mode=padding_mode
        )

    def forward(self, input, skip):
        out = self.up(input)
        out = torch.cat([out, self.skip_conv(skip)], dim=1)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class DownConvBlock(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        k,
        s,
        p,
        norm="batch",
        padding_mode="reflect",
    ):
        super().__init__()
        self.down = ConvLayer(
            nkernels=[d_in, d_in],
            norm=norm,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode,
        )
        self.conv1 = ConvLayer(
            nkernels=[d_in, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )
        self.conv2 = ConvLayer(
            nkernels=[d_out, d_out],
            norm=norm,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        out = self.down(input)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class ConvLayer(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        k=3,
        s=1,
        p=1,
        n_groups=4,
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        layers = []
        if norm == "batch":
            nl = nn.BatchNorm2d
        elif norm == "instance":
            nl = nn.InstanceNorm2d
        elif norm == "group":

            def group_norm(num_feats):
                return nn.GroupNorm(
                    num_channels=num_feats,
                    num_groups=n_groups,
                )

            nl = group_norm
        else:
            nl = None
        for i in range(len(nkernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=nkernels[i],
                    out_channels=nkernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(nkernels[i + 1]))

            if last_relu:
                layers.append(nn.ReLU())
            elif i < len(nkernels) - 2:
                layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        return self.conv(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        nkernels,
        norm="batch",
        last_relu=True,
        padding_mode="reflect",
    ):
        super().__init__()
        self.conv = ConvLayer(
            nkernels=nkernels,
            norm=norm,
            last_relu=last_relu,
            padding_mode=padding_mode,
        )

    def forward(self, input):
        return self.conv(input)
