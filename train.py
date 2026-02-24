"""
Train the CNN3D model for brain age prediction.

Usage:
    python train.py --data_dir <path_to_data> [--epochs 500] [--lr 1e-3]

The data directory should contain ``train/`` and ``test/`` subdirectories with
NIfTI files named ``<prefix>_<age>.nii``.
"""

import argparse
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data

from data.dataset import CTScanDataset
from models.cnn3d import CNN3D


# ============================================================================
# Training & Validation
# ============================================================================

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    """Train the model for one epoch and return average loss."""
    model.train()
    total_loss = 0
    for batch_data, target in train_loader:
        optimizer.zero_grad()
        batch_data = batch_data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(batch_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader.dataset)


def validate(model, device, val_loader, criterion):
    """Evaluate the model on the validation set and return average loss."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_data, target in val_loader:
            batch_data = batch_data.to(device)
            target = target.to(device)
            output = model(batch_data)
            loss = criterion(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader.dataset)


def plot_loss(train_losses, val_losses, save_path="loss_curve.png"):
    """Plot and save training / validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label='Training Loss')
    plt.plot(val_losses, 'g-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Loss curve saved to {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CNN3D for brain age prediction")
    parser.add_argument("--data_dir", type=str, default="samples",
                        help="Root data directory containing train/ and test/ subdirs")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--pretrain", type=str, default=None,
                        help="Path to pretrained CNN3D weights")
    parser.add_argument("--save_path", type=str, default="best_cnn3d.pth",
                        help="Path to save best model weights")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading threads")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- Data -----
    train_dir = osp.join(args.data_dir, "train")
    val_dir = osp.join(args.data_dir, "test")

    print("Loading datasets ...")
    t0 = time.time()
    train_dataset = CTScanDataset(train_dir, num_workers=args.num_workers)
    val_dataset = CTScanDataset(val_dir, num_workers=args.num_workers)
    print(f"Data loaded in {time.time() - t0:.1f}s "
          f"(train={len(train_dataset)}, val={len(val_dataset)})")

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )
    val_loader = data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    # ----- Model -----
    model = CNN3D(num_classes=1, pretrain_path=args.pretrain).to(device)
    criterion = nn.L1Loss(reduction="sum").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ----- Training loop -----
    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print(f"Starting training for {args.epochs} epochs ...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, device, train_loader, optimizer, criterion)
        val_loss = validate(model, device, val_loader, criterion)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[{epoch:03d}/{args.epochs:03d}] "
              f"train Loss: {train_loss:.6f}  val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_path)
            print(f"  → Saved best model (val_loss={val_loss:.6f})")

    plot_loss(train_losses, val_losses)
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
