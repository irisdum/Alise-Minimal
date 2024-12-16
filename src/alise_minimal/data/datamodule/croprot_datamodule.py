""" " File to define CropRot Lightning DataModule"""

from torch.utils.data import DataLoader

from alise_minimal.data.batch_class import CDBInput
from alise_minimal.data.datamodule.collate_fn import custom_collate_pastis_cd
from alise_minimal.data.datamodule.template_datamodule import TemplateDataModule
from alise_minimal.data.datamodule.utils import (
    apply_transform_basic,
    load_transform_one_mod,
)
from alise_minimal.data.dataset.croprot import CropRotDataset


class CropRotDataModule(TemplateDataModule):
    """
    LightningDataModule class dedicated to CropRot data
    """

    def __init__(
        self,
        dataset_path: str,
        path_dir_csv: str,
        dataset_name="dataset",
        max_len_s2: int = 60,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        batch_size: int = 2,
        dict_classes: dict | None = None,
        s2_band: list | None = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            dataset_name=dataset_name,
            path_dir_csv=path_dir_csv,
            max_len_s2=max_len_s2,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            batch_size=batch_size,
            s2_band=s2_band,
        )
        self.s2_transform = load_transform_one_mod(
            path_dir_csv=self.path_dir_csv, mod="s2"
        )
        if dict_classes is None:
            dict_classes = dict(
                zip(
                    [0, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    [
                        "background",
                        "rapeseed",
                        "cereal",
                        "proteaginous",
                        "soybean",
                        "sunflower",
                        "corn",
                        "rice",
                        "tuber",
                        "grassland",
                    ],
                )
            )
        self.dict_classes = dict_classes
        self.labels = list(self.dict_classes.values())
        self.num_classes = max(self.dict_classes.keys())
        print(self.num_classes)
        self.data_train = CropRotDataset(
            dataset_path=self.dataset_path,
            dataset_name=f"{self.dataset_name}_train.csv",
            max_len_s2=60,
        )
        self.data_val = CropRotDataset(
            dataset_path=self.dataset_path,
            dataset_name=f"{self.dataset_name}_val.csv",
            max_len_s2=60,
        )
        self.data_test = CropRotDataset(
            dataset_path=self.dataset_path,
            dataset_name=f"{self.dataset_name}_val.csv",
            max_len_s2=60,
        )

    def train_dataloader(self):
        """
        Mandatory LightningDataModule method
        Returns
        -------

        """
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=custom_collate_pastis_cd,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=custom_collate_pastis_cd,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=custom_collate_pastis_cd,
        )

    def transfer_batch_to_device(self, batch: CDBInput, device, dataloader_idx):
        """
        Mandatory when using personalized data class as output
         of the dataset __getitem__ method
        Parameters
        ----------
        batch :
        device :
        dataloader_idx :

        Returns
        -------

        """
        if isinstance(batch, CDBInput):
            # move all tensors in your custom data structure to the device
            batch.year1.to_device(device)
            batch.year2.to_device(device)
            batch.label = batch.label.to(device)
            batch.mask_label = batch.mask_label.to(device)
        elif dataloader_idx == 0:
            # skip device transfer for the first dataloader or anything you wish
            pass
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        return batch

    def on_after_batch_transfer(self, batch: CDBInput, dataloader_idx: int) -> CDBInput:
        """
        Motivations: apply transform on GPU not on CPU,
         to avoid transfert of float32 from CPU to GPU
        (Tricks from Julien Michel). CPU data are int16.
        Parameters
        ----------
        batch :
        dataloader_idx :

        Returns
        -------

        """

        year_1_s2 = apply_transform_basic(batch.year1.sits, self.s2_transform.transform)
        year_2_s2 = apply_transform_basic(batch.year2.sits, self.s2_transform.transform)
        # batch.year1.positions.to()
        # batch.year2.s2_doy = batch.year2.s2_doy.to(year_2_s2)
        batch.year1.sits = year_1_s2
        batch.year2.sits = year_2_s2

        return batch
