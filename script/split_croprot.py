"""
Main script to split the CropRot dataset into training testing and val
"""

import random
from pathlib import Path

import hydra
import pandas as pd


def save_as_csv(l_files: list, save_dir: str, suffix: str):
    df = pd.DataFrame(l_files, columns=["path"])
    df.to_csv(Path(save_dir).joinpath(f"dataset_{suffix}.csv"))


@hydra.main(config_path="../config/", config_name="split_croprot.yaml")
def main(config):
    iterate_items = Path(config.dataset_path).rglob("*.pt")
    l_files = []
    for idx, path in enumerate(iterate_items):
        # print(path)
        l_files += [path]
    random.shuffle(l_files)
    train_idx = int(config.train_per * len(l_files))
    val_idx = int(config.val_per * len(l_files))
    train_files = l_files[:train_idx]
    val_files = l_files[train_idx : train_idx + val_idx]
    test_files = l_files[train_idx + val_idx :]
    save_as_csv(train_files, config.dataset_path, suffix="train")
    save_as_csv(val_files, config.dataset_path, suffix="val")
    save_as_csv(test_files, config.dataset_path, suffix="test")
    print(f"Successful creation of csv files at {config.dataset_path}")


if __name__ == "__main__":
    main()
