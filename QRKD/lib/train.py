"""Training and evaluation loops to reproduce MNIST classical baselines (KD, RKD, QRKD classical parts)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .losses import QRKDWeights
from .utils import count_parameters


@dataclass
class TrainConfig:
    epochs: int = 10
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    verbose: bool = True
    max_batches: int | None = None


def train_teacher(
    model: nn.Module, train_loader: DataLoader, cfg: TrainConfig, test_loader: DataLoader | None = None
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Supervised training loop for a standalone model (teacher).

    Uses TrainConfig for epochs/lr/device/verbose to normalize API with student training.
    Returns the trained model in eval mode.
    """
    device = torch.device(cfg.device)
    model.to(device)
    opt = Adam(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    if cfg.verbose:
        print(f"[Teacher] Params: {count_parameters(model):,}")
        if cfg.max_batches is not None:
            print(f"[Checkrun] Limiting to {cfg.max_batches} batches per epoch")

    history_loss: List[float] = []
    history_train_acc: List[float] = []
    history_test_acc: List[float] = []

    def _evaluate_acc(m: nn.Module, loader: DataLoader) -> float:
        device = torch.device(cfg.device)
        m.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = m(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        return correct / max(total, 1) * 100.0
    for epoch in range(cfg.epochs):
        total_loss = 0.0
        n_batches = 0
        model.train()
        iterator = train_loader
        if cfg.verbose:
            iterator = tqdm(
                train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}", leave=False
            )
        for batch_idx, (x, y) in enumerate(iterator, 1):
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            if cfg.verbose:
                avg_loss = total_loss / max(n_batches, 1)
                iterator.set_postfix(loss=f"{avg_loss:.4f}")
            if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
                break
        if n_batches > 0:
            avg = total_loss / n_batches
            history_loss.append(avg)
            # Per-epoch accuracies
            train_acc = _evaluate_acc(model, train_loader)
            history_train_acc.append(train_acc)
            if test_loader is not None:
                test_acc = _evaluate_acc(model, test_loader)
                history_test_acc.append(test_acc)
            if cfg.verbose:
                extra = ""
                if test_loader is not None:
                    extra = f", train_acc: {train_acc:.2f}%, test_acc: {history_test_acc[-1]:.2f}%"
                else:
                    extra = f", train_acc: {train_acc:.2f}%"
                print(
                    f"[Teacher] Epoch {epoch + 1}/{cfg.epochs} - loss: {avg:.4f}{extra}"
                )

    model.eval()
    return model, {"loss": history_loss, "train_acc": history_train_acc, "test_acc": history_test_acc}


def train_student(
    student: nn.Module,
    teacher: nn.Module | None,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
    weights: QRKDWeights,
    student_name: str | None = None,
) -> dict[str, float | Dict[str, List[float]]]:
    device = torch.device(cfg.device)
    student.to(device)
    # Determine whether any distillation terms are active and a teacher is provided
    distill_active = bool((getattr(weights, "kd", 0.0) or getattr(weights, "dr", 0.0) or getattr(weights, "ar", 0.0)) and (teacher is not None))
    if teacher is not None:
        teacher.to(device)
        teacher.eval()
    if cfg.verbose:
        name = f"{student_name} " if student_name else ""
        if teacher is not None:
            print(f"[Student] {name}Params: {count_parameters(student):,} | Teacher Params: {count_parameters(teacher):,}")
        else:
            print(f"[Student] {name}Params: {count_parameters(student):,} | Teacher: None (scratch)")
        if cfg.max_batches is not None:
            print(f"[Checkrun] Limiting to {cfg.max_batches} batches per epoch")

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

    hist_loss: List[float] = []
    hist_task: List[float] = []
    hist_distill: List[float] = []
    hist_train_acc: List[float] = []
    hist_test_acc: List[float] = []
    for epoch in range(cfg.epochs):
        total_loss = total_task = total_rkd = 0.0
        n_batches = 0
        student.train()
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs}", leave=False) if cfg.verbose else train_loader
        for batch_idx, (x, y) in enumerate(iterator, 1):
            x, y = x.to(device), y.to(device)
            if distill_active:
                with torch.no_grad():
                    assert teacher is not None
                    logits_t, feat_t = teacher(x)
            logits_s, feat_s = student(x)
            loss_task = ce(logits_s, y)
            loss_rkd = weights(logits_s, logits_t, feat_s, feat_t) if distill_active else torch.zeros((), device=device)
            loss = loss_task + loss_rkd
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_task += loss_task.item()
            total_rkd += loss_rkd.item()
            n_batches += 1
            if cfg.verbose:
                iterator.set_postfix(
                    loss=f"{total_loss/n_batches:.4f}",
                    task=f"{total_task/n_batches:.4f}",
                    distill=f"{total_rkd/n_batches:.4f}"
                )
            if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
                break
        if n_batches == 0:
            continue
        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        avg_rkd = total_rkd / n_batches
        hist_loss.append(avg_loss)
        hist_task.append(avg_task)
        hist_distill.append(avg_rkd)
        train_acc = evaluate(student, train_loader)
        test_acc = evaluate(student, test_loader)
        hist_train_acc.append(train_acc)
        hist_test_acc.append(test_acc)
        if cfg.verbose:
            print(
                f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} (task: {avg_task:.4f}, distill: {avg_rkd:.4f}), train_acc: {train_acc:.2f}%, test_acc: {test_acc:.2f}%"
            )

    acc = evaluate(student, test_loader)
    return {
        "test_acc": acc,
        "history": {
            "loss": hist_loss,
            "task": hist_task,
            "distill": hist_distill,
            "train_acc": hist_train_acc,
            "test_acc": hist_test_acc,
        },
    }
