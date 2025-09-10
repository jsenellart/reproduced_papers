"""Training and evaluation loops to reproduce MNIST classical baselines (KD, RKD, QRKD classical parts)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .losses import QRKDWeights


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    weights: QRKDWeights,
) -> Dict[str, float]:
    device = torch.device(cfg.device)
    student.to(device)
    teacher.to(device)
    teacher.eval()

    opt = Adam(student.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()

    def evaluate(model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / total * 100.0

    for _ in range(cfg.epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                logits_t, feat_t = teacher(x)
            logits_s, feat_s = student(x)
            loss_task = ce(logits_s, y)
            loss_rkd = weights(logits_s, logits_t, feat_s, feat_t)
            loss = loss_task + loss_rkd
            opt.zero_grad()
            loss.backward()
            opt.step()

    acc = evaluate(student, test_loader)
    return {"test_acc": acc}
