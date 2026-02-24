"""
Test / evaluate a trained CNN3D model on brain MRI data.

Usage:
    python test.py --model_path best_cnn3d.pth --data_dir data_split/test

Outputs:
    - Per-sample predictions saved to CSV (predicted age, true age, MAE)
    - Overall MAE, MSE, and R² printed to console
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data.dataset import process_scan
from models.cnn3d import CNN3D


def extract_age_from_filename(filename):
    """Extract integer age from a filename like ``..._<age>.nii(.gz)``."""
    match = re.search(r'_(\d+)\.nii', filename)
    return int(match.group(1)) if match else None


def predict_single(model, device, nii_path):
    """Run inference on a single NIfTI file and return the predicted age."""
    volume = process_scan(nii_path)
    tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(tensor)
    return pred.item()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CNN3D brain age model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained .pth weights")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test NIfTI files")
    parser.add_argument("--output_csv", type=str, default="predictions.csv",
                        help="Path to save prediction results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = CNN3D(num_classes=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # Collect test files
    nii_files = [
        f for f in os.listdir(args.data_dir)
        if f.endswith('.nii') or f.endswith('.nii.gz')
    ]
    if not nii_files:
        print("No NIfTI files found in data_dir.")
        return

    # Inference
    results = []
    for fname in nii_files:
        fpath = os.path.join(args.data_dir, fname)
        true_age = extract_age_from_filename(fname)
        if true_age is None:
            print(f"  Skipping {fname} (cannot parse age)")
            continue

        pred_age = predict_single(model, device, fpath)
        mae = abs(pred_age - true_age)
        results.append({
            "file": fname,
            "true_age": true_age,
            "pred_age": round(pred_age, 2),
            "mae": round(mae, 2),
        })
        print(f"  {fname} | True: {true_age} | Pred: {pred_age:.2f} | MAE: {mae:.2f}")

    if not results:
        print("No valid results.")
        return

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")

    # Overall metrics
    true_ages = df["true_age"].values
    pred_ages = df["pred_age"].values
    overall_mae = mean_absolute_error(true_ages, pred_ages)
    overall_mse = mean_squared_error(true_ages, pred_ages)
    overall_r2 = r2_score(true_ages, pred_ages)

    print(f"\n{'='*40}")
    print(f"  Samples : {len(df)}")
    print(f"  MAE     : {overall_mae:.4f}")
    print(f"  MSE     : {overall_mse:.4f}")
    print(f"  R²      : {overall_r2:.4f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
