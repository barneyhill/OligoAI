import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import argparse

from rinalmo.data.alphabet import Alphabet
from rinalmo.utils.scaler import StandardScaler
from rinalmo.data.downstream.aso.dataset import ASODataset
from train_aso import ASOInhibitionPredictionWrapper


class ASODatasetWithLen(ASODataset):
    """Wrapper around ASODataset that adds __len__ method for DataLoader compatibility"""

    def __len__(self):
        return len(self.df)


def run_inference_on_kcnt2(
    model_checkpoint_path: str,
    data_path: str,
    output_path: str = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "auto"
):
    """
    Run inference on the KCNT2 dataset and save predictions to a new CSV.

    Args:
        model_checkpoint_path: Path to the trained model checkpoint.
        data_path: Path to the KCNT2 CSV file.
        output_path: Path to save the CSV with predictions (optional).
        batch_size: Batch size for inference.
        num_workers: Number of workers for data loading.
        device: Device to run inference on ('auto', 'cpu', 'cuda').
    """

    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the trained model with its saved scaler
    print(f"Loading model from: {model_checkpoint_path}")
    model = ASOInhibitionPredictionWrapper.load_from_checkpoint(
        model_checkpoint_path
        # Remove the scaler parameter - use the one saved with the model
    )
    model.eval()
    model.to(device)
    print(f"Model loaded successfully with saved scaler: {type(model.scaler)}")

    # Initialize alphabet and dataset
    alphabet = Alphabet()
    dataset = ASODatasetWithLen(
        data_path=data_path,
        alphabet=alphabet,
        pad_to_max_len=True,
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

    # Run inference
    all_predictions = []

    print("Starting inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Unpack the batch provided by the ASODataset
            # Match the format from the original inference script
            aso_tokens, chem_tokens, backbone_tokens, context_tokens, _, dosage, transfection_method_tokens, _ = batch

            # Move tensors to device
            aso_tokens = aso_tokens.to(device)
            chem_tokens = chem_tokens.to(device)
            backbone_tokens = backbone_tokens.to(device)
            context_tokens = context_tokens.to(device)
            dosage = dosage.to(device)
            transfection_method_tokens = transfection_method_tokens.to(device)

            # Use autocast for mixed precision on GPU
            if device == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage, transfection_method_tokens)
            else:
                scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage, transfection_method_tokens)

            # Inverse transform the predictions to get the real values
            unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)
            all_predictions.extend(unscaled_predictions.cpu().numpy().flatten())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} samples...")

    print(f"Inference completed. Generated {len(all_predictions)} predictions")

    # Load the original dataframe
    df = pd.read_csv(data_path)
    
    # Verify prediction count matches dataframe
    assert len(all_predictions) == len(df), f"Mismatch: {len(all_predictions)} predictions vs {len(df)} rows"

    # Add predictions to the original dataframe under the requested column name
    df['oligoAI_scores'] = all_predictions

    # Determine output path
    if output_path is None:
        data_path_obj = Path(data_path)
        output_path = data_path_obj.parent / f"{data_path_obj.stem}_w_oligoAI.csv"

    # Save results
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run OligoAI inference on KCNT2 data")
    parser.add_argument("data_path", type=str, help="Path to the KCNT2_expression.csv file")
    parser.add_argument("--model_checkpoint", type=str, default="../mae_final=19.03.ckpt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the CSV with predictions (default: <input_file>_w_oligoAI.csv)")
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

    # Run the adapted inference function
    run_inference_on_kcnt2(
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
