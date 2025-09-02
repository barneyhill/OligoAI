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

    # --- MODIFICATION 1: Create a dummy scaler ---
    # The original model was trained with scaled targets. Since we don't have the
    # original training data to re-create the exact scaler, we use a dummy scaler
    # that does not alter the model's output. The resulting scores will be the
    # model's raw (scaled) predictions.
    scaler = StandardScaler()
    scaler._mean = torch.tensor(0.0)
    scaler._scale = torch.tensor(1.0)
    print("Created a dummy scaler (mean=0, std=1). The output scores will be the model's raw scaled predictions.")

    # Initialize alphabet and dataset
    alphabet = Alphabet()
    dataset = ASODatasetWithLen(
        data_path=data_path,  # data_path is not needed as we provide the dataframe directly
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

    # --- MODIFICATION 3: Load the model with the dummy scaler ---
    print(f"Loading model from: {model_checkpoint_path}")
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
            # Unpack the batch provided by the ASODataset
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
                    scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)
            else:
                scaled_predictions = model(aso_tokens, chem_tokens, backbone_tokens, context_tokens, dosage)

            # With our dummy scaler, inverse_transform returns the same scaled predictions
            unscaled_predictions = model.scaler.inverse_transform(scaled_predictions)
            all_predictions.extend(unscaled_predictions.cpu().numpy().flatten())

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * batch_size} samples...")

    df = pd.read_csv('KCNT2_expression.csv')
    # --- MODIFICATION 4: Add new scores column and save results ---
    # Add predictions to the original dataframe under the requested column name
    df['oligoAI_scores'] = all_predictions

    # Save results
    df.to_csv('KCNT2_expression_w_oligoAI.csv', index=False)

    return df


def main():
    parser = argparse.ArgumentParser(description="Run OligoAI inference on KCNT2 data")
    parser.add_argument("data_path", type=str, help="Path to the KCNT2_expression.csv file")
    parser.add_argument("--model_checkpoint", type=str, default="../mae_final=19.03.ckpt",
                        help="Path to the trained model checkpoint")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the CSV with predictions (default: <input_file>_with_oligoAI_scores.csv)")
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
