import torch
from torch.utils.data import Dataset, Subset

import pandas as pd
import pickle
import numpy as np
import ast # For safely evaluating string-formatted lists

from typing import Union, Optional
from pathlib import Path

from rinalmo.data.alphabet import Alphabet

class ASODataset(Dataset):
    def __init__(self, data_path, alphabet, pad_to_max_len=True):
        super().__init__()

        # Store the data path for later use
        self.data_path = Path(data_path)

        # Load the CSV file
        self.df = pd.read_csv(self.data_path)

        # Convert DNA to RNA (T -> U)
        self.df['rna_sequence'] = self.df['aso_sequence_5_to_3'].str.replace('T', 'U')

        # Filter out any rows with missing inhibition values
        self.df = self.df.dropna(subset=['inhibition_percent'])

        # Parse the string-formatted lists into actual Python lists
        # Using ast.literal_eval is safe and efficient for this task.
        self.df['sugar_mods'] = self.df['sugar_mods'].apply(ast.literal_eval)
        self.df['backbone_mods'] = self.df['backbone_mods'].apply(ast.literal_eval)

        self.df = self.df.reset_index(drop=True)

        # Calculate median dosage for imputation
        self.median_dosage = self.df['dosage'].median()
        self.alphabet = alphabet

        # ### NEW: Explicitly define the vocabularies for modifications ###
        # This ensures consistent tokenization across all experiments.
        # '<pad>' is assigned to index 0, which is standard practice.
        self.chem_vocab = {
            '<pad>': 0,
            'DNA': 1,
            'MOE': 2,
            'cET': 3,
        }
        self.backbone_vocab = {
            '<pad>': 0,
            'PO': 1,
            'PS': 2,
        }

        print("Using Chemistry Vocabulary:", self.chem_vocab)
        print("Using Backbone Vocabulary:", self.backbone_vocab)

        self.max_enc_seq_len = -1
        if pad_to_max_len:
            # Add 2 for special tokens (CLS/EOS)
            self.max_enc_seq_len = self.df['rna_sequence'].str.len().max() + 2
            # Calculate max length for context sequences too
            self.max_context_len = self.df['rna_context'].str.len().max() + 2

    def _tokenize_and_pad_mods(self, mods_list, vocab, max_len):
        """
        Tokenizes a list of modification strings and pads it to align with the
        nucleotide sequence, which has CLS and EOS tokens.
        """
        pad_idx = vocab['<pad>']

        # 1. Start with a pad token to align with the CLS token
        padded_tokens = [pad_idx]

        # 2. Add the actual modification tokens. Use .get() for safety, defaulting to pad_idx
        #    if an unexpected modification string appears.
        tokens = [vocab.get(mod, pad_idx) for mod in mods_list]
        padded_tokens.extend(tokens)

        # 3. Add padding to match the final length (accounts for EOS and seq padding)
        padding_needed = max_len - len(padded_tokens)
        if padding_needed > 0:
            padded_tokens.extend([pad_idx] * padding_needed)

        # Ensure the final list is not longer than max_len
        return padded_tokens[:max_len]

    def __getitem__(self, idx):
        df_row = self.df.iloc[idx]

        # Encode ASO sequence (unchanged)
        seq_encoded = torch.tensor(
            self.alphabet.encode(df_row['rna_sequence'], pad_to_len=self.max_enc_seq_len),
            dtype=torch.long
        )

        # Tokenize and pad the modification sequences using the hardcoded vocabularies
        chem_encoded = torch.tensor(
            self._tokenize_and_pad_mods(
                df_row['sugar_mods'], self.chem_vocab, self.max_enc_seq_len
            ),
            dtype=torch.long
        )

        backbone_encoded = torch.tensor(
            self._tokenize_and_pad_mods(
                df_row['backbone_mods'], self.backbone_vocab, self.max_enc_seq_len
            ),
            dtype=torch.long
        )

        # Encode masked context (unchanged)
        context_encoded = torch.tensor(
            self.alphabet.encode(df_row['rna_context'], pad_to_len=self.max_context_len),
            dtype=torch.long
        )
        context_encoded[context_encoded == self.alphabet.unk_idx] = self.alphabet.mask_idx

        inhibition = torch.tensor(df_row['inhibition_percent'], dtype=torch.float32)
        dosage = df_row['dosage'] if pd.notna(df_row['dosage']) else self.median_dosage
        dosage = torch.tensor(dosage, dtype=torch.float32)
        custom_id = df_row['custom_id']

        # Update the return statement with the new tensors
        return (
            seq_encoded,
            chem_encoded,
            backbone_encoded,
            context_encoded,
            inhibition,
            dosage,
            custom_id
        )

    def train_val_test_split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, random_state: int = 42):
        np.random.seed(random_state)
        unique_custom_ids = self.df['custom_id'].unique()
        n_custom_ids = len(unique_custom_ids)
        shuffled_custom_ids = np.random.permutation(unique_custom_ids)

        train_size = int(train_ratio * n_custom_ids)
        val_size = int(val_ratio * n_custom_ids)

        train_custom_ids = shuffled_custom_ids[:train_size]
        val_custom_ids = shuffled_custom_ids[train_size:train_size + val_size]
        test_custom_ids = shuffled_custom_ids[train_size + val_size:]

        train_indices = self.df[self.df['custom_id'].isin(train_custom_ids)].index.tolist()
        val_indices = self.df[self.df['custom_id'].isin(val_custom_ids)].index.tolist()
        test_indices = self.df[self.df['custom_id'].isin(test_custom_ids)].index.tolist()

        print(f"Split by custom_id - Train: {len(train_custom_ids)} custom_ids ({len(train_indices)} samples)")
        print(f"Val: {len(val_custom_ids)} custom_ids ({len(val_indices)} samples)")
        print(f"Test: {len(test_custom_ids)} custom_ids ({len(test_indices)} samples)")

        # Create a copy of the dataframe to add the split column
        split_df = self.df.copy()

        # Add the 'split' column and assign values based on the indices
        split_df['split'] = 'unassigned' # Default value
        split_df.loc[train_indices, 'split'] = 'train'
        split_df.loc[val_indices, 'split'] = 'val'
        split_df.loc[test_indices, 'split'] = 'test'

        # Construct the output filename
        output_filename = self.data_path.parent / f"{self.data_path.stem}.withsplit.csv"

        # Save the dataframe with the new 'split' column
        split_df.to_csv(output_filename, index=False)
        print(f"Split assignments saved to: {output_filename}")

        # Create Subset objects for PyTorch
        train_ds = Subset(self, indices=train_indices)
        val_ds = Subset(self, indices=val_indices)
        test_ds = Subset(self, indices=test_indices)

        return train_ds, val_ds, test_ds
