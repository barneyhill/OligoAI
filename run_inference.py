import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

from rinalmo.data.alphabet import Alphabet
# --- FIX 1: Import the StandardScaler ---
from rinalmo.utils.scaler import StandardScaler
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

    # --- FIX 2: Re-create the scaler by fitting on the training data ---
    # First, load the dataframe to determine the training split
    df = pd.read_csv(data_path)
    
    # Use the same logic as the datamodule to split data
    np.random.seed(42) # Must be the same seed used for training
    unique_custom_ids = df['custom_id'].unique()
    shuffled_custom_ids = np.random.permutation(unique_custom_ids)
    train_ratio = 0.8
    train_size = int(train_ratio * len(unique_custom_ids))
    train_custom_ids = set(shuffled_custom_ids[:train_size])
    
    # Filter the dataframe to get only the training data
    train_df = df[df['custom_id'].isin(train_custom_ids)]
    
    # Fit the scaler on the 'inhibition_percent' column of the training data
    scaler = StandardScaler()
    train_targets = torch.tensor(train_df['inhibition_percent'].values, dtype=torch.float32).unsqueeze(-1)
    scaler.partial_fit(train_targets)
    print(f"Scaler re-created from training data with mean: {scaler._mean.item():.4f} and std: {scaler._scale.item():.4f}")


    # Initialize alphabet and dataset
    alphabet = Alphabet()
    dataset = ASODatasetWithLen(
        data_path=data_path,
        alphabet=alphabet,
        pad_to_max_len=True,
        df=df # Pass the pre-loaded dataframe
    )

    print(f"Loaded dataset with {len(dataset)} samples")

    # Create dataloader for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        shuffle=False
    )

    # Load the trained model, passing the re-created scaler
    print(f"Loading model from: {model_checkpoint_path}")
    # --- FIX 3: Pass the scaler object during model loading ---
    model = ASOInhibitionPredictionWrapper.load_from_checkpoint(
        model_checkpoint_path, 
        scaler=scaler
    )
    model.eval()
    model.to(device)

    print(f"Model loaded successfully")

    # Run inference
    all_predictions = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack batch
            aso_tokens, chem_tokens, backbone_tokens, context_tokens, _, dosage, _ = batch

            # Move tensors to device
            aso_tokens = aso_tokens.to(device)
            chem_tokens = chem_tokens.to(device)
            backbone_tokens = backbone_tokens.to(device)
            context_tokens = context_tokens.to(device)
            dosage = dosage.to(device)

            # Use autocast for mixed precision on GPU
            if device == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # Model outputs SCALED predictions
                    scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)
            else:
                scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)

            # --- FIX 4: Inverse transform the predictions to get the real values ---
            unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)

            all_predictions.extend(unscaled_predictions.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} samples...")

    print(f"Inference completed. Generated {len(all_predictions)} predictions")

    # The rest of the script is largely the same, but we re-use the pre-loaded df
    assert len(all_predictions) == len(df), f"Mismatch: {len(all_predictions)} predictions vs {len(df)} rows"

    # Add split column
    print("Assigning dataset splits for analysis...")
    df['split'] = 'test' # Default to test
    val_size = int(0.1 * len(unique_custom_ids))
    val_custom_ids = set(shuffled_custom_ids[train_size : train_size + val_size])
    df.loc[df['custom_id'].isin(train_custom_ids), 'split'] = 'train'
    df.loc[df['custom_id'].isin(val_custom_ids), 'split'] = 'val'
    
    # The rest of the analysis from here will now work correctly
    # Add predictions to dataframe
    df['predicted_inhibition_percent'] = all_predictions
    df['prediction_error'] = np.abs(np.array(all_predictions) - df['inhibition_percent'].values)
    df['prediction_squared_error'] = (np.array(all_predictions) - df['inhibition_percent'].values) ** 2

    # Calculate some summary statistics
    mae = np.mean(df['prediction_error'])
    rmse = np.sqrt(np.mean(df['prediction_squared_error']))

    print(f"\nOverall Prediction Statistics:")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Calculate R² score
    from scipy.stats import pearsonr
    r, p_value = pearsonr(df['predicted_inhibition_percent'], df['inhibition_percent'])
    r2 = r ** 2
    print(f"R²: {r2:.4f}")
    
    # Calculate per-split statistics
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        if not split_df.empty:
            mae = np.mean(split_df['prediction_error'])
            rmse = np.sqrt(np.mean(split_df['prediction_squared_error']))
            r, _ = pearsonr(split_df['predicted_inhibition_percent'], split_df['inhibition_percent'])
            r2 = r ** 2
            print(f"{split.upper()} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, samples: {len(split_df)}")
    
    # ... (rest of your analysis script) ...
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
