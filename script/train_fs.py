"""Script to train ALISE in a fully supervised configuration"""

import logging
import signal
from pathlib import Path

import hydra
import torchmetrics
from hydra.utils import instantiate
from lightning import Trainer
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from torchmetrics import MetricCollection

from alise_minimal.data.datamodule.croprot_datamodule import CropRotDataModule
from alise_minimal.lightning_module.fully_supervised_segmentation import (
    AliseFSSeg,
    FSSegTrainConfig,
    build_alise_fs_seg,
)
from alise_minimal.torch_model.alise import (
    ALISEConfigBuild,
    TransformerConfig,
    TransformerLayerConfig,
)
from alise_minimal.torch_model.attention_mechanism import ConfigLQMHA
from alise_minimal.torch_model.decoder import MLPDecoderConfig
from alise_minimal.torch_model.sse import UnetConfig

my_logger = logging.getLogger(__name__)


def load_train_config(config: DictConfig, num_class: int):
    """

    Parameters
    ----------
    config :
    num_class : nber of class involded in the segmentation task

    Returns
    -------

    """
    return FSSegTrainConfig(
        batch_size=config.datamodule.batch_size,
        optimizer=config.train.optimizer_config,
        optimizer_monitor="val_loss",
        scheduler=config.train.scheduler_config,
        lr=0.001,
        loss=instantiate(config.train.loss),
        metrics=MetricCollection(
            {
                "accuracy": torchmetrics.Accuracy(
                    task="multiclass", num_classes=num_class
                ),
                "precision": torchmetrics.Precision(
                    task="multiclass", num_classes=num_class
                ),
            }
        ),
    )


def load_alisefs_module_from_config(
    config: DictConfig, num_bands: int, num_class: int
) -> AliseFSSeg:
    """
    Parameters
    ----------
    config :
    num_bands : nber of input bands
    num_class : nber of class involded in the segmentation class

    Returns
    -------

    """
    unet_config = UnetConfig(
        inplanes=num_bands, planes=config.module.d_model, **config.module.unet_config
    )
    transformer_layer_config = TransformerLayerConfig(
        d_model=config.module.d_model, **config.module.transformer_layer_config
    )
    transformer_config = TransformerConfig(
        layer_config=transformer_layer_config, **config.module.transformer_config
    )
    temp_proj_config = ConfigLQMHA(
        d_in=config.module.d_model, **config.module.temp_proj_config
    )
    alise_build_config = ALISEConfigBuild(
        unet_config=unet_config,
        transformer_config=transformer_config,
        temp_proj_config=temp_proj_config,
        pe_T=config.pe_encoder.pe_T,
    )
    config_decoder = MLPDecoderConfig(
        inplanes=config.module.d_model * config.module.temp_proj_config.n_q,
        planes=num_class,
        d_hidden=config.module.config_decoder.d_hidden,
    )
    return build_alise_fs_seg(
        alise_build_config=alise_build_config,
        decoder_config=config_decoder,
        train_config=load_train_config(config, num_class=num_class),
    )


@hydra.main(config_path="../config/", config_name="train_fs.yaml")
def main(myconfig: DictConfig):
    if myconfig.verbose == 0:
        my_logger.setLevel(logging.WARN)
    elif myconfig.verbose == 1:
        my_logger.setLevel(logging.INFO)
    elif myconfig.verbose == 2:
        my_logger.setLevel(logging.DEBUG)
    callbacks = [instantiate(cb_conf) for _, cb_conf in myconfig.callbacks.items()]
    logger = [
        instantiate(logg_conf, save_dir=Path.cwd())
        for _, logg_conf in myconfig.logger.items()
    ]
    if myconfig.train.slurm_restart:
        print("Automatic restart")
        plugin = [SLURMEnvironment(requeue_signal=signal.SIGHUP, auto_requeue=True)]
    else:
        plugin = None

    my_trainer: Trainer = instantiate(
        myconfig.train.trainer,
        callbacks=callbacks,
        logger=logger,
        max_epochs=myconfig.train.trainer.max_epochs,
        plugins=plugin,
        _convert_="partial",
    )
    datamodule: CropRotDataModule = instantiate(myconfig.datamodule)
    myconfig.train.batch_size = datamodule.batch_size
    pl_module = load_alisefs_module_from_config(
        config=myconfig,
        num_class=datamodule.num_classes,
        num_bands=len(datamodule.s2_band),
    )

    my_trainer.fit(pl_module, datamodule=datamodule, ckpt_path=myconfig.ckpt_path)
    my_trainer.test(pl_module, datamodule)


if __name__ == "__main__":
    main()
