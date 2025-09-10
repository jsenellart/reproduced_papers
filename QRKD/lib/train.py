"""Training and evaluation loops to reproduce MNIST classical baselines (KD, RKD, QRKD classical parts)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import QRKDWeights


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True


def train_student(
    student: nn.Module,
    teacher: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    weights: QRKDWeights,
) -> dict[str, float]:
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

    for epoch in range(cfg.epochs):
        total_loss = 0.0
        total_task = 0.0
        total_rkd = 0.0
        n_batches = 0
        student.train()
        iterator = train_loader
        if cfg.verbose:
            iterator = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False
            )
        for x, y in iterator:
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
            # accumulate metrics
            total_loss += loss.item()
            total_task += loss_task.item()
            total_rkd += loss_rkd.item()
            n_batches += 1
            if cfg.verbose:
                avg_loss = total_loss / max(n_batches, 1)
                avg_task = total_task / max(n_batches, 1)
                avg_rkd = total_rkd / max(n_batches, 1)
                iterator.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    task=f"{avg_task:.4f}",
                    distill=f"{avg_rkd:.4f}",
                )

        if cfg.verbose and n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_task = total_task / n_batches
            avg_rkd = total_rkd / n_batches
            print(
                f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} (task: {avg_task:.4f}, distill: {avg_rkd:.4f})"
            )

    acc = evaluate(student, test_loader)
    return {"test_acc": acc}
