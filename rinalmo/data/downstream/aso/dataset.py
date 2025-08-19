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
        
        # Load the pickle file
        self.df = pd.read_csv(data_path)
        
        # Convert DNA to RNA (T -> U)
        self.df['rna_sequence'] = self.df['aso_sequence_5_to_3'].str.replace('T', 'U')
        
        # Filter out any rows with missing inhibition values
        self.df = self.df.dropna(subset=['inhibition_percent'])
        self.df = self.df.reset_index(drop=True)
        
        self.alphabet = alphabet
        
        self.max_enc_seq_len = -1
        if pad_to_max_len:
            # Add 2 for special tokens (CLS/EOS)
            self.max_enc_seq_len = self.df['rna_sequence'].str.len().max() + 2
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]
        
        seq = df_row['rna_sequence']
        seq_encoded = torch.tensor(
            self.alphabet.encode(seq, pad_to_len=self.max_enc_seq_len), 
            dtype=torch.long
        )
        
        # Normalize inhibition_percent to 0-1 range for better training stability
        # We'll denormalize in the model wrapper
        inhibition = torch.tensor(df_row['inhibition_percent'] / 100.0, dtype=torch.float32)
        
        return seq_encoded, inhibition
    
    def train_val_test_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42):
        """
        Split the dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            random_state: Random seed for reproducibility
        """
        np.random.seed(random_state)
        n_samples = len(self.df)
        indices = np.random.permutation(n_samples)
        
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_ds = Subset(self, indices=train_indices.tolist())
        val_ds = Subset(self, indices=val_indices.tolist())
        test_ds = Subset(self, indices=test_indices.tolist())
        
        return train_ds, val_ds, test_ds
