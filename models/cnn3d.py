"""
CNN3D Model for 3D Brain Age Prediction.

A 3D Convolutional Neural Network with skip connections for brain age
regression using NIfTI MRI volumes.

Architecture:
    6 convolutional blocks with skip connections + global average pooling + FC layers.
    Input: (batch, 1, depth, height, width) — single-channel 3D MRI volume.
    Output: (batch, 1) — predicted brain age.
"""

import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """3D CNN with residual skip connections for brain age prediction."""

    def __init__(self, num_classes=1, pretrain_path=None):
        super(CNN3D, self).__init__()
        self.act = nn.LeakyReLU(0.01)

        # Block 1
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.batch_norm1 = nn.BatchNorm3d(32)

        # Block 2
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.batch_norm2 = nn.BatchNorm3d(64)
        self.skip_conn1 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=1, stride=1),
            nn.BatchNorm3d(64)
        )

        # Block 3
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.batch_norm3 = nn.BatchNorm3d(128)
        self.skip_conn2 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm3d(128)
        )

        # Block 4
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        self.batch_norm4 = nn.BatchNorm3d(256)
        self.skip_conn3 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(128, 256, kernel_size=1, stride=1),
            nn.BatchNorm3d(256)
        )

        # Block 5
        self.conv5 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool3d(2)
        self.batch_norm5 = nn.BatchNorm3d(512)
        self.skip_conn4 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(256, 512, kernel_size=1, stride=1),
            nn.BatchNorm3d(512)
        )

        # Block 6
        self.conv6 = nn.Conv3d(512, 1024, kernel_size=3, padding=1)
        self.pool6 = nn.MaxPool3d(2)
        self.batch_norm6 = nn.BatchNorm3d(1024)
        self.skip_conn5 = nn.Sequential(
            nn.MaxPool3d(2),
            nn.Conv3d(512, 1024, kernel_size=1, stride=1),
            nn.BatchNorm3d(1024)
        )

        # Classifier head
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, num_classes)

        # Load pretrained weights if provided
        if pretrain_path:
            self._load_pretrained(pretrain_path)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool1(x)
        x = self.batch_norm1(x)

        # Block 2 + skip
        identity = x
        x = self.conv2(x)
        x = self.act(x)
        x = self.pool2(x)
        x = self.batch_norm2(x)
        x = x + self.skip_conn1(identity)

        # Block 3 + skip
        identity = x
        x = self.conv3(x)
        x = self.act(x)
        x = self.pool3(x)
        x = self.batch_norm3(x)
        x = x + self.skip_conn2(identity)

        # Block 4 + skip
        identity = x
        x = self.conv4(x)
        x = self.act(x)
        x = self.pool4(x)
        x = self.batch_norm4(x)
        x = x + self.skip_conn3(identity)

        # Block 5 + skip
        identity = x
        x = self.conv5(x)
        x = self.act(x)
        x = self.pool5(x)
        x = self.batch_norm5(x)
        x = x + self.skip_conn4(identity)

        # Block 6 + skip
        identity = x
        x = self.conv6(x)
        x = self.act(x)
        x = self.pool6(x)
        x = self.batch_norm6(x)
        x = x + self.skip_conn5(identity)

        # Head
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def _load_pretrained(self, pretrain_path):
        """Load pretrained weights from a checkpoint file."""
        checkpoint = torch.load(pretrain_path, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint, strict=False)
        print(f"Loaded pretrained weights from {pretrain_path}")
