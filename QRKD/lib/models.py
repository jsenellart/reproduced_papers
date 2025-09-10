"""Teacher/Student CNNs for MNIST with exact parameter targets.

Both models output a flattened feature of shape 12 x 4 x 4 (=192) before the FC.
Teacher: exactly 6,690 params. Student: exactly 1,725 params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


class BaseCNN12x4x4(nn.Module):
    """Three-conv CNN that yields 12x4x4 features before the classifier head."""

    def __init__(
        self,
        c1: int,
        c2: int,
        f_hidden: int
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)  # 28 -> 14
        self.down4 = nn.AdaptiveAvgPool2d((4, 4))  # -> 4x4
        self.conv3 = nn.Conv2d(c2, 12, 1, bias=False)
        self.fc1 = nn.Linear(12 * 4 * 4, f_hidden, bias=True)
        self.head = nn.Linear(f_hidden, 10, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = self.down4(x)
        x = f.relu(self.conv3(x))
        feat = torch.flatten(x, 1)
        feat = f.relu(self.fc1(feat))
        logits = self.head(feat)
        return logits, feat


class TeacherCNN(BaseCNN12x4x4):
    def __init__(self) -> None:
        # Exact 6,690 params with (c1,c2,F)=(1,18,31)
        super().__init__(
            c1=1,
            c2=18,
            f_hidden=31
        )


class StudentCNN(nn.Module):
    """Student without FC hidden layer, exactly 1,725 params.

    Architecture:
    - conv1: 1 -> 3 channels, 3x3, bias=False
    - conv2: 3 -> 1 channel, 3x3, bias=False
    - maxpool 2x2 (28 -> 14)
    - adaptive avg pool to 1x1
    - conv3 (1x1): 1 -> 151 channels, bias=False
    - head: Linear(151 -> 10), bias=True

    Param count:
    9*c1 + 9*c1*c2 + c2*K + 10*K + 10 = 9*3 + 9*3*1 + 1*151 + 10*151 + 10 = 27+27+151+1510+10 = 1,725
    Features returned for RKD are the flattened 151-dim vector before the logits.
    """

    def __init__(self) -> None:
        super().__init__()
        c1, c2, k = 3, 1, 151
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(c2, k, 1, bias=False)
        self.head = nn.Linear(k, 10, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = self.gap(x)
        x = self.conv3(x)
        feat = torch.flatten(x, 1)  # (N, 151)
        logits = self.head(feat)
        return logits, feat
