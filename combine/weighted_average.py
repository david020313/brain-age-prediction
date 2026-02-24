"""
Weighted Average Fusion of C1 (gray matter) and C2 (white matter) predictions.

Two pre-trained CNN3D models predict independently, then their outputs are
combined via learnable weights: ``pred = w1*pred_c1 + w2*pred_c2``.

Usage:
    python -m combine.weighted_average --data_dir <path> --pretrain_c1 c1.pth --pretrain_c2 c2.pth
"""

import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data as data

from models.cnn3d import CNN3D
from combine._dual_dataset import DualChannelDataset


class WeightedAverageFusion(nn.Module):
    """Fuse two CNN3D models via learnable weighted average."""

    def __init__(self, pretrain_path1=None, pretrain_path2=None):
        super().__init__()
        self.model1 = CNN3D(num_classes=1)
        self.model2 = CNN3D(num_classes=1)

        if pretrain_path1:
            self.model1.load_state_dict(
                torch.load(pretrain_path1, map_location='cpu', weights_only=True), strict=False
            )
        if pretrain_path2:
            self.model2.load_state_dict(
                torch.load(pretrain_path2, map_location='cpu', weights_only=True), strict=False
            )

        # Freeze feature extractors, only train weights
        for p in self.model1.parameters():
            p.requires_grad = False
        for p in self.model2.parameters():
            p.requires_grad = False

        self.w1 = nn.Parameter(torch.tensor([0.5]))
        self.w2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x1, x2):
        pred1 = self.model1(x1)
        pred2 = self.model2(x2)
        w = torch.softmax(torch.stack([self.w1, self.w2]), dim=0)
        return w[0] * pred1 + w[1] * pred2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--pretrain_c1", type=str, required=True)
    parser.add_argument("--pretrain_c2", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="weighted_avg.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DualChannelDataset(osp.join(args.data_dir, "train"))
    val_ds = DualChannelDataset(osp.join(args.data_dir, "test"))
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    model = WeightedAverageFusion(args.pretrain_c1, args.pretrain_c2).to(device)
    criterion = nn.L1Loss(reduction="sum").to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train(); t_loss = 0
        for combined, target in train_loader:
            c1 = combined[:, 0:1].to(device)
            c2 = combined[:, 1:2].to(device)
            target = target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(c1, c2), target)
            loss.backward(); optimizer.step(); t_loss += loss.item()
        t_loss /= len(train_loader.dataset)

        model.eval(); v_loss = 0
        with torch.no_grad():
            for combined, target in val_loader:
                c1 = combined[:, 0:1].to(device)
                c2 = combined[:, 1:2].to(device)
                v_loss += criterion(model(c1, c2), target.to(device)).item()
        v_loss /= len(val_loader.dataset)

        print(f"[{ep:03d}/{args.epochs}] train: {t_loss:.4f} val: {v_loss:.4f} "
              f"w1={model.w1.item():.3f} w2={model.w2.item():.3f}")
        if v_loss < best:
            best = v_loss
            torch.save(model.state_dict(), args.save_path)
