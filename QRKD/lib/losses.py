"""KD and RKD (distance & angle) classical losses for MNIST reproduction."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float = 4.0, alpha: float = 0.5) -> torch.Tensor:
    """Standard KD with temperature T and mixing alpha (Hinton et al.)."""
    p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    loss_kd = F.kl_div(p_s, p_t, reduction="batchmean") * (T * T)
    return alpha * loss_kd


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances in a batch (N, D) -> (N, N)."""
    x2 = (x * x).sum(dim=1, keepdim=True)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    dist2 = dist2.clamp_min(1e-9)
    return torch.sqrt(dist2)


def rkd_distance_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    sd = pairwise_distances(student_feat)
    td = pairwise_distances(teacher_feat)
    # Normalize by mean to stabilize
    sd = sd / (sd.mean() + 1e-8)
    td = td / (td.mean() + 1e-8)
    return F.smooth_l1_loss(sd, td)


def rkd_angle_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    def angles(x: torch.Tensor) -> torch.Tensor:
        # Centered differences: (x_i - x_j) normalized
        diffs = x.unsqueeze(0) - x.unsqueeze(1)  # (N, N, D)
        norms = diffs.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        dirs = diffs / norms
        # Angles via cosine between direction pairs relative to reference index 0
        ref = dirs[:, 0:1, :]  # (N,1,D)
        cos = (dirs * ref).sum(dim=-1)  # (N,N)
        return cos

    sa = angles(student_feat)
    ta = angles(teacher_feat)
    return F.smooth_l1_loss(sa, ta)


class QRKDWeights(nn.Module):
    def __init__(self, kd: float = 0.5, dr: float = 0.1, ar: float = 0.1):
        super().__init__()
        self.kd = kd
        self.dr = dr
        self.ar = ar

    def forward(self, logits_s, logits_t, feat_s, feat_t):
        loss = 0.0
        if self.kd:
            loss = loss + self.kd * kd_loss(logits_s, logits_t)
        if self.dr:
            loss = loss + self.dr * rkd_distance_loss(feat_s, feat_t)
        if self.ar:
            loss = loss + self.ar * rkd_angle_loss(feat_s, feat_t)
        return loss
