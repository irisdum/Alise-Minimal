import pytest
import torch.nn
import torchmetrics
from torchmetrics import MetricCollection

from alise_minimal.data.batch_class import CDBInput, SegBatch, SITSBatch
from alise_minimal.lightning_module.fully_supervised_segmentation import (
    FSSegTrainConfig,
    build_alise_fs_seg,
)
from alise_minimal.lightning_module.hydra_dataclass import (
    CAWConfig,
    OptimizerAdamConfig,
)
from alise_minimal.torch_model.alise import (
    ALISEConfigBuild,
    TransformerConfig,
    TransformerLayerConfig,
)
from alise_minimal.torch_model.attention_mechanism import ConfigLQMHA
from alise_minimal.torch_model.decoder import MLPDecoderConfig
from alise_minimal.torch_model.sse import UnetConfig


def create_fake_input(B, T, C, H, W) -> SegBatch:
    input = torch.rand(B, T, C, H, W)
    positions = torch.rand(B, T)
    labels = torch.ones(B, H, W).long()
    return SegBatch(
        sits=input,
        positions=positions,
        labels=labels,
        pad_mask=torch.zeros(B, T).bool(),
    )


def create_fake_cd_input(B, T, C, H, W):
    year1 = SITSBatch(
        sits=torch.rand(B, T, C, H, W),
        positions=torch.rand(B, T),
        pad_mask=torch.zeros(B, T).bool(),
        cld_mask=torch.zeros(B, T, C, H, W),
    )
    year2 = SITSBatch(
        sits=torch.rand(B, T, C, H, W),
        positions=torch.rand(B, T),
        pad_mask=torch.zeros(B, T).bool(),
        cld_mask=torch.zeros(B, T, C, H, W),
    )
    return CDBInput(
        year1,
        year2,
        label=torch.rand(B, 3, H, W).long(),
        mask_label=torch.zeros(B, 3, H, W).long(),
    )


@pytest.mark.local
def test_create_alise_fully_supervised_segmentation():
    B, T, C, H, W = 2, 10, 3, 64, 64
    d_model = 64
    pe_T = 3000
    nq = 10
    n_class = 5
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
    config_decoder = MLPDecoderConfig(d_model * nq, 16, n_class)
    train_config = FSSegTrainConfig(
        batch_size=B,
        optimizer=OptimizerAdamConfig(),
        optimizer_monitor="val_loss",
        scheduler=CAWConfig(),
        lr=0.001,
        loss=torch.nn.CrossEntropyLoss(),
        metrics=MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=n_class
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=n_class
                ),
            }
        ),
    )
    alise_fs_seg_module = build_alise_fs_seg(
        alise_build_config=alise_build_config,
        decoder_config=config_decoder,
        train_config=train_config,
    )
    fake_input = create_fake_cd_input(B, T, C, H, W)
    output, loss = alise_fs_seg_module.shared_step(fake_input)
    assert output.shape == (B, n_class, H, W)
