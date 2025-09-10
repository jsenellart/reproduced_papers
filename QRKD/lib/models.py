"""Teacher/Student CNNs for MNIST used in the paper (parameter counts approximated)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class SmallCNN(nn.Module):
    """~1.7k-7k parameter range variants depending on widths."""

    def __init__(self, channels: int = 8, fc: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(channels * 7 * 7, fc)
        self.head = nn.Linear(fc, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(f.relu(self.conv2(x)))
        # 28x28 -> pool -> 14x14 -> conv no size change -> pool -> 7x7 (if we had two pools)
        # here only one pool; ensure shape to 7x7 by additional pool
        x = self.pool(x)
        feat = torch.flatten(x, 1)
        feat = f.relu(self.fc1(feat))
        logits = self.head(feat)
        return logits, feat


class TeacherCNN(SmallCNN):
    def __init__(self):
        super().__init__(channels=16, fc=64)  # higher capacity


class StudentCNN(SmallCNN):
    def __init__(self):
        super().__init__(channels=8, fc=32)  # smaller capacity
