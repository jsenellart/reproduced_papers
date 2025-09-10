"""Implementation entrypoint to reproduce MNIST classical results from QRKD paper.

This script trains a teacher CNN, then trains a student with:
- From scratch (no distillation)
- KD (logit matching)
- RKD (distance + angle)
- QRKD (classical components only: KD + RKD; quantum part excluded for now)

It reports a compact summary similar to Table 1 (test accuracy only here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .lib.datasets import DataConfig, mnist_loaders
from .lib.models import TeacherCNN, StudentCNN
from .lib.losses import QRKDWeights, kd_loss, rkd_distance_loss, rkd_angle_loss
from .lib.train import TrainConfig, train_student
from .lib.utils import set_seed


@dataclass
class ExpConfig:
    seed: int = 0
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64


def train_teacher(train_loader, test_loader, epochs: int = 5, lr: float = 1e-3) -> TeacherCNN:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TeacherCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def run() -> Dict[str, float]:
    cfg = ExpConfig()
    set_seed(cfg.seed)

    dcfg = DataConfig(batch_size=cfg.batch_size)
    train_loader, test_loader = mnist_loaders(dcfg)

    # 1) Train teacher quickly (could be pre-trained in practice)
    teacher = train_teacher(train_loader, test_loader, epochs=5, lr=cfg.lr)

    # 2) Student from scratch (task loss only)
    student_scratch = StudentCNN()
    res_scratch = train_student(
        student_scratch,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=cfg.epochs, lr=cfg.lr),
        QRKDWeights(kd=0.0, dr=0.0, ar=0.0),
    )

    # 3) KD only
    student_kd = StudentCNN()
    res_kd = train_student(
        student_kd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=cfg.epochs, lr=cfg.lr),
        QRKDWeights(kd=0.5, dr=0.0, ar=0.0),
    )

    # 4) RKD (distance + angle)
    student_rkd = StudentCNN()
    res_rkd = train_student(
        student_rkd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=cfg.epochs, lr=cfg.lr),
        QRKDWeights(kd=0.0, dr=0.1, ar=0.1),
    )

    # 5) QRKD classical (KD + RKD). Quantum loss will be added later.
    student_qrkd = StudentCNN()
    res_qrkd = train_student(
        student_qrkd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=cfg.epochs, lr=cfg.lr),
        QRKDWeights(kd=0.5, dr=0.1, ar=0.1),
    )

    summary = {
        "F. Scratch": res_scratch["test_acc"],
        "KD": res_kd["test_acc"],
        "RKD": res_rkd["test_acc"],
        "QRKD (classical)": res_qrkd["test_acc"],
    }

    print("MNIST Test Accuracy (%)")
    for k, v in summary.items():
        print(f"{k:>16}: {v:5.2f}")

    return summary


if __name__ == "__main__":
    run()
