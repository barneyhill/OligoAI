from torch.utils.data import DataLoader

import lightning.pytorch as pl

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.dataset import ASODataset

class ASODataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        alphabet: Alphabet = Alphabet(),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.alphabet = alphabet

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state

    def setup(self, stage: Optional[str] = None):
        dataset = ASODataset(
            self.data_path,
            alphabet=self.alphabet,
            pad_to_max_len=True
        )

        self.train_dataset, self.val_dataset, self.test_dataset = dataset.train_val_test_split(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            random_state=self.random_state
        )

        print(f"Dataset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        print(f"Max sequence length: {dataset.max_enc_seq_len}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False
        )
