"""KD and RKD (distance & angle) classical losses for MNIST reproduction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    t: float = 4.0,
    alpha: float = 0.5,
) -> torch.Tensor:
    """Standard KD with temperature T and mixing alpha (Hinton et al.)."""
    p_s = f.log_softmax(student_logits / t, dim=1)
    p_t = f.softmax(teacher_logits / t, dim=1)
    loss_kd = f.kl_div(p_s, p_t, reduction="batchmean") * (t * t)
    return alpha * loss_kd


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    """Pairwise Euclidean distances (full NxN matrix, zeros on diagonal)."""
    x2 = (x * x).sum(dim=1, keepdim=True)  # (N,1)
    dist2 = x2 + x2.t() - 2.0 * (x @ x.t())
    dist2 = dist2.clamp_min(0.0)
    return dist2.clamp_min(1e-12).sqrt()


def rkd_distance_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Relational KD distance term (matches official RKD: mean over positive distances)."""
    with torch.no_grad():
        td = pairwise_distances(teacher_feat)
        mean_td = td[td > 0].mean()
        td = td / (mean_td + 1e-12)
    sd = pairwise_distances(student_feat)
    mean_sd = sd[sd > 0].mean()
    sd = sd / (mean_sd + 1e-12)
    return f.smooth_l1_loss(sd, td)


def rkd_angle_loss(student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
    """Relational KD angle term (full pairwise angular relations, as in RKD repo)."""
    with torch.no_grad():
        td = teacher_feat.unsqueeze(0) - teacher_feat.unsqueeze(1)  # (N,N,D)
        norm_td = f.normalize(td, p=2, dim=2)  # (N,N,D)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)  # (N*N*N)
    sd = student_feat.unsqueeze(0) - student_feat.unsqueeze(1)
    norm_sd = f.normalize(sd, p=2, dim=2)
    s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
    return f.smooth_l1_loss(s_angle, t_angle)


class QRKDWeights(nn.Module):
    def __init__(self, kd: float = 0.5, dr: float = 0.1, ar: float = 0.1):
        super().__init__()
        self.kd = kd
        self.dr = dr
        self.ar = ar

    def forward(self, logits_s, logits_t, feat_s, feat_t):
        # Always return a Tensor (even if all weights are zero)
        loss = torch.zeros((), device=logits_s.device)
        if self.kd:
            loss = loss + self.kd * kd_loss(logits_s, logits_t)
        if self.dr:
            loss = loss + self.dr * rkd_distance_loss(feat_s, feat_t)
        if self.ar:
            loss = loss + self.ar * rkd_angle_loss(feat_s, feat_t)
        return loss
