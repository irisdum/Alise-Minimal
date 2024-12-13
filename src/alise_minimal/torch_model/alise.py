"""
Related to ALISE construction
"""

from dataclasses import dataclass

from einops import rearrange, repeat
from torch import Tensor, nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from alise_minimal.torch_model.attention_mechanism import (
    ConfigLQMHA,
    LearnedQMultiHeadAttention,
)
from alise_minimal.torch_model.patch_embedding import PatchEmbedding
from alise_minimal.torch_model.sse import Unet, UnetConfig
from alise_minimal.torch_model.temporal_positional_encoder import PositionalEncoder


@dataclass
class TransformerLayerConfig:
    d_model: int = 64
    nhead: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    batch_first: bool = True


@dataclass
class TransformerConfig:
    layer_config: TransformerLayerConfig
    num_layers: int = 3


def build_transformer(transformer_config: TransformerConfig) -> TransformerEncoder:
    layer = TransformerEncoderLayer(
        d_model=transformer_config.layer_config.d_model,
        nhead=transformer_config.layer_config.nhead,
        dim_feedforward=transformer_config.layer_config.dim_feedforward,
        dropout=transformer_config.layer_config.dropout,
        batch_first=transformer_config.layer_config.batch_first,
    )
    return TransformerEncoder(
        encoder_layer=layer, num_layers=transformer_config.num_layers
    )


class ALISE(nn.Module):
    """
    ALISE architecture
    """

    def __init__(
        self,
        patch_embedding: PatchEmbedding,
        temporal_encoder: TransformerEncoder,
        temporal_projector: LearnedQMultiHeadAttention,
    ):
        super().__init__()
        self.temporal_projector = temporal_projector
        self.temporal_encoder = temporal_encoder
        self.patch_embedding = patch_embedding

    def forward(self, sits: Tensor, positions: Tensor, pad_mask: Tensor):
        """

        Parameters
        ----------
        sits : (B,T,C,H,W)
        positions : (B,T)
        padd_mask : (B,T), True means the value should be *ignored* in the attention

        Returns
        -------

        """
        B, T, C, H, W = sits.shape
        x = self.patch_embedding(sits, positions=positions)
        x = rearrange(x, "B T C H W -> (B H W ) T C")
        pad_mask = repeat(pad_mask, "B T -> (B H W) T", H=H, W=W)
        x = self.temporal_encoder(x, src_key_padding_mask=pad_mask.bool())
        x = self.temporal_projector(x, pad_mask=~pad_mask)
        x = rearrange(
            x,
            "( B H W ) T C -> B T C H W",
            B=B,
            T=self.temporal_projector.n_q,
            H=H,
            W=W,
        )
        return x


def build_alise(
    unet_config: UnetConfig,
    transformer_config: TransformerConfig,
    temp_proj_config: ConfigLQMHA,
    pe_T: int = 3000,
) -> ALISE:
    sse = Unet(unet_config)
    tpe = PositionalEncoder(d=unet_config.planes, T=pe_T)
    patch_embed = PatchEmbedding(sse=sse, tpe=tpe)
    temporal_encoder = build_transformer(transformer_config)
    temp_proj = LearnedQMultiHeadAttention(config=temp_proj_config)
    return ALISE(
        patch_embedding=patch_embed,
        temporal_encoder=temporal_encoder,
        temporal_projector=temp_proj,
    )
