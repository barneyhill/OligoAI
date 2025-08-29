import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

from rinalmo.data.alphabet import Alphabet
from rinalmo.data.downstream.aso.dataset import ASODataset
from train_aso import ASOInhibitionPredictionWrapper


class ASODatasetWithLen(ASODataset):
    """Wrapper around ASODataset that adds __len__ method for DataLoader compatibility"""

    def __len__(self):
        return len(self.df)


def run_inference(
    model_checkpoint_path: str,
    data_path: str,
    output_path: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto"
):
    """
    Run inference on all rows in the dataset and save predictions to CSV.

    Args:
        model_checkpoint_path: Path to the trained model checkpoint
        data_path: Path to the CSV file used for training
        output_path: Path to save the CSV with predictions (optional)
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
        device: Device to run inference on ('auto', 'cpu', 'cuda')
    """

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Initialize alphabet and dataset
    alphabet = Alphabet()
    dataset = ASODatasetWithLen(
        data_path=data_path,
        alphabet=alphabet,
        pad_to_max_len=True
    )

    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Max sequence length: {dataset.max_enc_seq_len}")

    # Create dataloader for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        shuffle=False  # Important: keep original order
    )

    # Load the trained model
    print(f"Loading model from: {model_checkpoint_path}")
    model = ASOInhibitionPredictionWrapper.load_from_checkpoint(model_checkpoint_path)
    model.eval()
    model.to(device)

    print(f"Model loaded successfully")
    print(f"Chemistry vocab size: {len(dataset.chem_vocab)}")
    print(f"Backbone vocab size: {len(dataset.backbone_vocab)}")

    # Run inference
    all_predictions = []
    all_targets = []
    all_custom_ids = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            aso_tokens, chem_tokens, backbone_tokens, context_tokens, inhibition_target, dosage, custom_ids = batch

            # Move tensors to device
            aso_tokens = aso_tokens.to(device)
            chem_tokens = chem_tokens.to(device)
            backbone_tokens = backbone_tokens.to(device)
            context_tokens = context_tokens.to(device)
            dosage = dosage.to(device)

            # Use autocast for mixed precision on GPU
            if device == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)
            else:
                predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)

            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(inhibition_target.numpy())
            all_custom_ids.extend(custom_ids)

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} samples...")

    print(f"Inference completed. Generated {len(all_predictions)} predictions")

    # Load original dataframe and add predictions
    df = pd.read_csv(data_path)

    # Ensure we have the same number of predictions as rows in the dataframe
    assert len(all_predictions) == len(df), f"Mismatch: {len(all_predictions)} predictions vs {len(df)} rows"

    # Add dataset split assignments (same logic as in the original dataset)
    print("Assigning dataset splits...")
    np.random.seed(42)  # Use same random state as default
    unique_custom_ids = df['custom_id'].unique()
    n_custom_ids = len(unique_custom_ids)
    shuffled_custom_ids = np.random.permutation(unique_custom_ids)

    # Use same ratios as default (0.8 train, 0.1 val, 0.1 test)
    train_ratio, val_ratio = 0.8, 0.1
    train_size = int(train_ratio * n_custom_ids)
    val_size = int(val_ratio * n_custom_ids)

    train_custom_ids = shuffled_custom_ids[:train_size]
    val_custom_ids = shuffled_custom_ids[train_size:train_size + val_size]
    test_custom_ids = shuffled_custom_ids[train_size + val_size:]

    # Add split column
    df['split'] = 'unassigned'  # Default value
    df.loc[df['custom_id'].isin(train_custom_ids), 'split'] = 'train'
    df.loc[df['custom_id'].isin(val_custom_ids), 'split'] = 'val'
    df.loc[df['custom_id'].isin(test_custom_ids), 'split'] = 'test'

    print(f"Split assignments - Train: {len(train_custom_ids)} custom_ids, Val: {len(val_custom_ids)} custom_ids, Test: {len(test_custom_ids)} custom_ids")

    # Add predictions to dataframe
    df['predicted_inhibition_percent'] = all_predictions
    df['prediction_error'] = np.abs(np.array(all_predictions) - df['inhibition_percent'].values)
    df['prediction_squared_error'] = (np.array(all_predictions) - df['inhibition_percent'].values) ** 2

    # Calculate some summary statistics
    mae = np.mean(np.abs(np.array(all_predictions) - df['inhibition_percent'].values))
    mse = np.mean((np.array(all_predictions) - df['inhibition_percent'].values) ** 2)
    rmse = np.sqrt(mse)

    print(f"\nPrediction Statistics:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Calculate R² score
    from scipy.stats import pearsonr
    r, p_value = pearsonr(all_predictions, df['inhibition_percent'].values)
    r2 = r ** 2
    print(f"R²: {r2:.4f}")
    print(f"Pearson correlation: {r:.4f} (p-value: {p_value:.2e})")

    # Calculate per-split statistics
    for split in ['train', 'val', 'test']:
        split_mask = df['split'] == split
        if split_mask.sum() > 0:
            split_predictions = np.array(all_predictions)[split_mask]
            split_targets = df.loc[split_mask, 'inhibition_percent'].values
            split_mae = np.mean(np.abs(split_predictions - split_targets))
            split_mse = np.mean((split_predictions - split_targets) ** 2)
            split_rmse = np.sqrt(split_mse)

            # Calculate R² for this split
            from scipy.stats import pearsonr
            if len(split_predictions) > 1:
                r, p_value = pearsonr(split_predictions, split_targets)
                r2 = r ** 2
                print(f"{split.upper()} - MAE: {split_mae:.4f}, RMSE: {split_rmse:.4f}, R²: {r2:.4f}, samples: {split_mask.sum()}")
            else:
                print(f"{split.upper()} - MAE: {split_mae:.4f}, RMSE: {split_rmse:.4f}, samples: {split_mask.sum()}")

    # Calculate per-custom_id statistics if available
    if 'custom_id' in df.columns:
        custom_id_stats = []
        for custom_id in df['custom_id'].unique():
            mask = df['custom_id'] == custom_id
            pred_subset = np.array(all_predictions)[mask]
            target_subset = df.loc[mask, 'inhibition_percent'].values

            if len(pred_subset) > 1:
                from scipy.stats import spearmanr
                corr, _ = spearmanr(pred_subset, target_subset)
                if not np.isnan(corr):
                    custom_id_stats.append(corr)

        if custom_id_stats:
            mean_spearman = np.mean(custom_id_stats)
            print(f"Mean Spearman correlation across custom_ids: {mean_spearman:.4f}")

    # Determine output path
    if output_path is None:
        data_path_obj = Path(data_path)
        output_path = data_path_obj.parent / f"{data_path_obj.stem}.with_predictions.csv"

    # Save results
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run inference on ASO inhibition prediction model")
    parser.add_argument("data_path", type=str, help="Path to the CSV file used for training")
    parser.add_argument("--model_checkpoint", type=str, default="../mae_final=19.03.ckpt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the CSV with predictions (default: input_file.with_predictions.csv)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to run inference on")

    args = parser.parse_args()

    # Check if files exist
    if not Path(args.data_path).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_path}")

    if not Path(args.model_checkpoint).exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_checkpoint}")

    # Run inference
    results_df = run_inference(
        model_checkpoint_path=args.model_checkpoint,
        data_path=args.data_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device
    )

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
