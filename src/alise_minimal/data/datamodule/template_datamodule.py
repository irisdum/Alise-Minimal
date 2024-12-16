"""File for the definition of the lightning datamodule class"""

from lightning import LightningDataModule


class TemplateDataModule(LightningDataModule):
    """
    Basis class for lightning DataModule
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
    ):
        super().__init__()
        self.path_dir_csv = path_dir_csv
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.max_len_s2 = max_len_s2
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
