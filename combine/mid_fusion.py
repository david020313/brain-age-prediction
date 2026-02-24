"""
Mid-level Fusion of Gray Matter (C1) and White Matter (C2) branches.

Architecture: Two parallel CNN branches extract features independently, then
fuse at the intermediate feature level with attention mechanisms (SE + Spatial).

Usage:
    python -m combine.mid_fusion --data_dir <path> [--epochs 500]
"""

import argparse
import os
import os.path as osp
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as data

from data.dataset import read_nifti_file
from combine._dual_dataset import DualChannelDataset


# ============================================================================
# Attention Modules
# ============================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (channel attention)."""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, *_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention Module."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attention


# ============================================================================
# Model
# ============================================================================

def _make_branch(in_ch=1):
    """Create one CNN branch (5 conv blocks with skip connections)."""
    return nn.ModuleDict({
        'conv1': nn.Conv3d(in_ch, 32, 3, padding=1), 'pool1': nn.MaxPool3d(2), 'bn1': nn.BatchNorm3d(32),
        'conv2': nn.Conv3d(32, 64, 3, padding=1), 'pool2': nn.MaxPool3d(2), 'bn2': nn.BatchNorm3d(64),
        'skip1': nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(32, 64, 1), nn.BatchNorm3d(64)),
        'conv3': nn.Conv3d(64, 128, 3, padding=1), 'pool3': nn.MaxPool3d(2), 'bn3': nn.BatchNorm3d(128),
        'skip2': nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(64, 128, 1), nn.BatchNorm3d(128)),
        'conv4': nn.Conv3d(128, 256, 3, padding=1), 'pool4': nn.MaxPool3d(2), 'bn4': nn.BatchNorm3d(256),
        'skip3': nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(128, 256, 1), nn.BatchNorm3d(256)),
        'conv5': nn.Conv3d(256, 512, 3, padding=1), 'pool5': nn.MaxPool3d(2), 'bn5': nn.BatchNorm3d(512),
        'skip4': nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(256, 512, 1), nn.BatchNorm3d(512)),
    })


def _forward_branch(branch, x, act):
    """Forward pass through a branch."""
    x = act(branch['conv1'](x)); x = branch['pool1'](x); x = branch['bn1'](x)
    identity = x
    x = act(branch['conv2'](x)); x = branch['pool2'](x); x = branch['bn2'](x); x = x + branch['skip1'](identity)
    identity = x
    x = act(branch['conv3'](x)); x = branch['pool3'](x); x = branch['bn3'](x); x = x + branch['skip2'](identity)
    identity = x
    x = act(branch['conv4'](x)); x = branch['pool4'](x); x = branch['bn4'](x); x = x + branch['skip3'](identity)
    identity = x
    x = act(branch['conv5'](x)); x = branch['pool5'](x); x = branch['bn5'](x); x = x + branch['skip4'](identity)
    return x


class MidFusionCNN3D(nn.Module):
    """Mid-level fusion: two CNN branches + attention fusion head."""

    def __init__(self, pretrain_path_c1=None, pretrain_path_c2=None, num_classes=1):
        super().__init__()
        self.act = nn.LeakyReLU(0.01)
        self.c1_branch = _make_branch()
        self.c2_branch = _make_branch()

        # Fusion layers
        self.fusion_conv1 = nn.Conv3d(512, 256, 3, padding=1)
        self.fusion_bn1 = nn.BatchNorm3d(256)
        self.fusion_conv2 = nn.Conv3d(256, 256, 3, padding=1)
        self.fusion_bn2 = nn.BatchNorm3d(256)
        self.fusion_conv3 = nn.Conv3d(256, 256, 3, padding=1)
        self.fusion_bn3 = nn.BatchNorm3d(256)
        self.fusion_skip = nn.Conv3d(512, 256, 1)

        self.se = SEBlock(256)
        self.sa = SpatialAttention()

        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

        if pretrain_path_c1:
            self._load_branch(pretrain_path_c1, self.c1_branch)
        if pretrain_path_c2:
            self._load_branch(pretrain_path_c2, self.c2_branch)

    def forward(self, x):
        x_c1 = _forward_branch(self.c1_branch, x[:, 0:1], self.act)
        x_c2 = _forward_branch(self.c2_branch, x[:, 1:2], self.act)

        f = x_c1 + x_c2
        identity = self.fusion_skip(f)
        f = self.fusion_bn1(self.act(self.fusion_conv1(f)))
        f = self.fusion_bn2(self.act(self.fusion_conv2(f)))
        f = self.fusion_bn3(self.act(self.fusion_conv3(f)))
        f = f + identity
        f = self.sa(self.se(f))

        x = self.gap(f)
        x = torch.flatten(x, 1)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    @staticmethod
    def _load_branch(path, branch):
        """Load single-branch CNN3D weights into a fusion branch."""
        state = torch.load(path, map_location='cpu', weights_only=True)
        mapping = {}
        for i in range(1, 6):
            for p in ['weight', 'bias']:
                old = f'conv{i}.{p}'; new = f'conv{i}.{p}'
                if old in state: mapping[new] = state[old]
                old = f'batch_norm{i}.{p}'; new = f'bn{i}.{p}'
                if old in state: mapping[new] = state[old]
            for p in ['running_mean', 'running_var']:
                old = f'batch_norm{i}.{p}'; new = f'bn{i}.{p}'
                if old in state: mapping[new] = state[old]
        for i in range(1, 5):
            for j, p in enumerate(['weight', 'bias']):
                for sub in [1, 2]:
                    old = f'skip_conn{i}.{sub}.{p}'; new = f'skip{i}.{sub}.{p}'
                    if old in state: mapping[new] = state[old]
            for p in ['running_mean', 'running_var']:
                old = f'skip_conn{i}.2.{p}'; new = f'skip{i}.2.{p}'
                if old in state: mapping[new] = state[old]
        branch.load_state_dict(mapping, strict=False)
        print(f"Loaded {len(mapping)} params from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="c1_data")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pretrain_c1", type=str, default=None)
    parser.add_argument("--pretrain_c2", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="mid_fusion.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = DualChannelDataset(osp.join(args.data_dir, "train"))
    val_ds = DualChannelDataset(osp.join(args.data_dir, "test"))
    train_loader = data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = data.DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True)

    model = MidFusionCNN3D(args.pretrain_c1, args.pretrain_c2).to(device)
    criterion = nn.L1Loss(reduction="sum").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        t_loss = sum(
            (optimizer.zero_grad(), criterion(model(d.to(device)), t.to(device)).backward() or criterion(model(d.to(device)), t.to(device)).item())[-1]
            for d, t in train_loader
        ) if False else 0
        # Simplified training loop
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
            best = v_loss
            torch.save(model.state_dict(), args.save_path)
