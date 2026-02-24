"""
Dual-Channel CNN3D: Gray matter (C1) and White matter (C2) as two input channels.

Instead of separate branches, both tissue maps are stacked as a 2-channel input
and processed by a single CNN3D (with ``in_channels=2``).

Usage:
    python -m combine.dual_channel --data_dir <path> [--epochs 500]
"""

import argparse
import os.path as osp

import torch
import torch.nn as nn
import torch.utils.data as data

from combine._dual_dataset import DualChannelDataset


class DualChannelCNN3D(nn.Module):
    """CNN3D that takes 2-channel input (C1 + C2)."""

    def __init__(self, num_classes=1, pretrain_path=None):
        super().__init__()
        self.act = nn.LeakyReLU(0.01)

        # Note: first conv takes 2 channels instead of 1
        self.conv1 = nn.Conv3d(2, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.batch_norm1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, 3, padding=1); self.pool2 = nn.MaxPool3d(2); self.batch_norm2 = nn.BatchNorm3d(64)
        self.skip_conn1 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(32, 64, 1), nn.BatchNorm3d(64))

        self.conv3 = nn.Conv3d(64, 128, 3, padding=1); self.pool3 = nn.MaxPool3d(2); self.batch_norm3 = nn.BatchNorm3d(128)
        self.skip_conn2 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(64, 128, 1), nn.BatchNorm3d(128))

        self.conv4 = nn.Conv3d(128, 256, 3, padding=1); self.pool4 = nn.MaxPool3d(2); self.batch_norm4 = nn.BatchNorm3d(256)
        self.skip_conn3 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(128, 256, 1), nn.BatchNorm3d(256))

        self.conv5 = nn.Conv3d(256, 512, 3, padding=1); self.pool5 = nn.MaxPool3d(2); self.batch_norm5 = nn.BatchNorm3d(512)
        self.skip_conn4 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(256, 512, 1), nn.BatchNorm3d(512))

        self.conv6 = nn.Conv3d(512, 1024, 3, padding=1); self.pool6 = nn.MaxPool3d(2); self.batch_norm6 = nn.BatchNorm3d(1024)
        self.skip_conn5 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(512, 1024, 1), nn.BatchNorm3d(1024))

        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

        if pretrain_path:
            state = torch.load(pretrain_path, map_location='cpu', weights_only=True)
            # skip conv1 (different in_channels)
            filtered = {k: v for k, v in state.items() if not k.startswith('conv1.')}
            self.load_state_dict(filtered, strict=False)

    def forward(self, x):
        x = self.batch_norm1(self.pool1(self.act(self.conv1(x))))
        i = x; x = self.batch_norm2(self.pool2(self.act(self.conv2(x)))); x = x + self.skip_conn1(i)
        i = x; x = self.batch_norm3(self.pool3(self.act(self.conv3(x)))); x = x + self.skip_conn2(i)
        i = x; x = self.batch_norm4(self.pool4(self.act(self.conv4(x)))); x = x + self.skip_conn3(i)
        i = x; x = self.batch_norm5(self.pool5(self.act(self.conv5(x)))); x = x + self.skip_conn4(i)
        i = x; x = self.batch_norm6(self.pool6(self.act(self.conv6(x)))); x = x + self.skip_conn5(i)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        return self.fc2(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="dual_channel.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DualChannelDataset(osp.join(args.data_dir, "train"))
    val_ds = DualChannelDataset(osp.join(args.data_dir, "test"))
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    model = DualChannelCNN3D(pretrain_path=args.pretrain).to(device)
    criterion = nn.L1Loss(reduction="sum").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train(); t_loss = 0
        for d, t in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(d.to(device)), t.to(device))
            loss.backward(); optimizer.step(); t_loss += loss.item()
        t_loss /= len(train_loader.dataset)

        model.eval(); v_loss = 0
        with torch.no_grad():
            for d, t in val_loader:
                v_loss += criterion(model(d.to(device)), t.to(device)).item()
        v_loss /= len(val_loader.dataset)

        print(f"[{ep:03d}/{args.epochs}] train: {t_loss:.4f} val: {v_loss:.4f}")
        if v_loss < best:
            best = v_loss; torch.save(model.state_dict(), args.save_path)
