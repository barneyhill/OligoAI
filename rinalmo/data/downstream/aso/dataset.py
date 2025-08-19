import torch
from torch.utils.data import Dataset, Subset

import pandas as pd
import pickle
import numpy as np

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

class ASODataset(Dataset):
    def __init__(
        self,
        data_path: Union[str, Path],
        alphabet: Alphabet,
        pad_to_max_len: bool = True,
    ):
        super().__init__()

        # Load the CSV file
        self.df = pd.read_csv(data_path)

        # Convert DNA to RNA (T -> U)
        self.df['rna_sequence'] = self.df['aso_sequence_5_to_3'].str.replace('T', 'U')

        # Filter out any rows with missing inhibition values
        self.df = self.df.dropna(subset=['inhibition_percent'])
        self.df = self.df.reset_index(drop=True)

        # Calculate median dosage for imputation
        self.median_dosage = self.df['dosage'].median()
        
        self.alphabet = alphabet

        self.max_enc_seq_len = -1
        if pad_to_max_len:
            # Add 2 for special tokens (CLS/EOS)
            self.max_enc_seq_len = self.df['rna_sequence'].str.len().max() + 2

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]

        seq = df_row['rna_sequence']
        seq_encoded = torch.tensor(
            self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len),
            dtype=torch.long
        )

        # Normalize inhibition_percent to 0-1 range
        inhibition = torch.tensor(df_row['inhibition_percent'] / 100.0, dtype=torch.float32)

        # Get dosage, impute with median if missing
        dosage = df_row['dosage'] if pd.notna(df_row['dosage']) else self.median_dosage
        dosage = torch.tensor(dosage, dtype=torch.float32)

        # Also return custom_id for validation grouping
        custom_id = df_row['custom_id']

        return seq_encoded, inhibition, dosage, custom_id


    def train_val_test_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42):
        """
        Split the dataset into train, validation, and test sets at the custom_id level.
        All samples with the same custom_id will be in the same split.

        Args:
            train_ratio: Proportion of custom_ids for training
            val_ratio: Proportion of custom_ids for validation
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)
        
        # Get unique custom_ids and shuffle them
        unique_custom_ids = self.df['custom_id'].unique()
        n_custom_ids = len(unique_custom_ids)
        shuffled_custom_ids = np.random.permutation(unique_custom_ids)
        
        # Calculate split sizes
        train_size = int(train_ratio * n_custom_ids)
        val_size = int(val_ratio * n_custom_ids)
        
        # Split custom_ids
        train_custom_ids = shuffled_custom_ids[:train_size]
        val_custom_ids = shuffled_custom_ids[train_size:train_size + val_size]
        test_custom_ids = shuffled_custom_ids[train_size + val_size:]
        
        # Get indices for each split
        train_indices = self.df[self.df['custom_id'].isin(train_custom_ids)].index.tolist()
        val_indices = self.df[self.df['custom_id'].isin(val_custom_ids)].index.tolist()
        test_indices = self.df[self.df['custom_id'].isin(test_custom_ids)].index.tolist()
        
        print(f"Split by custom_id - Train: {len(train_custom_ids)} custom_ids ({len(train_indices)} samples)")
        print(f"Val: {len(val_custom_ids)} custom_ids ({len(val_indices)} samples)")
        print(f"Test: {len(test_custom_ids)} custom_ids ({len(test_indices)} samples)")
        
        train_ds = Subset(self, indices=train_indices)
        val_ds = Subset(self, indices=val_indices)
        test_ds = Subset(self, indices=test_indices)

        return train_ds, val_ds, test_ds
