"""File associated to fullsy supervised training of ALISE"""

from dataclasses import dataclass

from einops import rearrange
from torch import Tensor, nn
from torchmetrics import MetricCollection

from alise_minimal.data.batch_class import SegBatch
from alise_minimal.lightning_module.template_module import TemplateModule, TrainConfig
from alise_minimal.torch_model.alise import ALISE, ALISEConfigBuild, build_alise
from alise_minimal.torch_model.decoder import MLPDecoder, MLPDecoderConfig


@dataclass
class FSSegTrainConfig(TrainConfig):
    metrics: MetricCollection


class AliseFSSeg(TemplateModule):
    """
    Lightning module for fully supervised segmentation tasks
    """

    def __init__(
        self, alise: ALISE, decoder: nn.Module, train_config: FSSegTrainConfig
    ):
        super().__init__(train_config)
        self.model: ALISE = alise
        self.decoder = decoder
        metrics = train_config.metrics  # metrics should be scalar metrics !!!
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, batch: SegBatch) -> Tensor:
        """

        Parameters
        ----------
        batch :

        Returns
        -------
        the output of ALISE
        """
        x = self.model.forward(
            sits=batch.sits, positions=batch.positions, pad_mask=batch.pad_mask
        )
        x = rearrange(x, "B T C H W -> B H W (T C)")
        x = self.decoder(x)
        return rearrange(x, "B H W F -> B F H W")

    def shared_step(self, batch: SegBatch) -> tuple[Tensor, Tensor]:
        out = self.forward(batch)
        loss = self.loss(out, batch.labels)
        return out, loss

    def training_step(self, batch: SegBatch, batch_idx: int) -> Tensor:
        out, loss = self.shared_step(batch)
        self.train_metrics.update(out, batch.labels)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=True,
            batch_size=self.bs,
            prog_bar=True,
        )
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_metrics.compute()
        self.log_dict(
            self.train_metrics,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def validation_step(self, batch: SegBatch, batch_idx: int):
        out, _ = self.shared_step(batch)
        self.val_metrics.update(out, batch.labels)

    def on_validation_epoch_end(self) -> None:
        self.val_metrics.compute()
        self.log_dict(
            self.val_metrics,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )

    def test_step(self, batch: SegBatch, batch_idx: int):
        out, _ = self.shared_step(batch)
        self.test_metrics.update(out, batch.labels)

    def on_test_epoch_end(self) -> None:
        self.test_metrics.compute()
        self.log_dict(
            self.test_metrics,
            on_epoch=True,
            batch_size=self.bs,
            prog_bar=True,
        )


def build_alise_fs_seg(
    alise_build_config: ALISEConfigBuild,
    decoder_config: MLPDecoderConfig,
    train_config: FSSegTrainConfig,
) -> AliseFSSeg:
    alise = build_alise(alise_build_config)
    mlp = MLPDecoder(decoder_config)
    alis_fs_seg = AliseFSSeg(alise=alise, decoder=mlp, train_config=train_config)
    return alis_fs_seg
