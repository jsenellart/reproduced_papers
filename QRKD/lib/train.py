"""Training and evaluation loops to reproduce MNIST classical baselines (KD, RKD, QRKD classical parts).

Debug instrumentation:
    If environment variable ``QRKD_DEBUG`` is set to a non-empty value, the student
    training loop will collect per-batch gradient norms and logits statistics for
    the first ``QRKD_DEBUG_MAXBATCH`` batches (default: 5) of each epoch. The
    collected information is added under the ``debug`` key of the returned dict
    (``train_student`` only) so that pathological non-learning runs can be
    diagnosed post-hoc without flooding stdout.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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
    # Debug instrumentation containers (activated via env var)
    debug_active = bool(os.environ.get("QRKD_DEBUG"))
    debug_max_batches = int(os.environ.get("QRKD_DEBUG_MAXBATCH", "5"))
    debug: Dict[str, List] = {}
    if debug_active:
        debug["batch_grad_norms"] = []  # list of lists per epoch
        debug["batch_logits_mean"] = []
        debug["batch_logits_std"] = []
        debug["batch_pred_entropy"] = []
        debug["batch_label_entropy"] = []
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
        epoch_grad_norms: List[float] = [] if debug_active else []
        epoch_logits_mean: List[float] = [] if debug_active else []
        epoch_logits_std: List[float] = [] if debug_active else []
        epoch_pred_entropy: List[float] = [] if debug_active else []
        epoch_label_entropy: List[float] = [] if debug_active else []

        for batch_idx, (x, y) in enumerate(iterator, 1):
            x, y = x.to(device), y.to(device)
            if distill_active:
                with torch.no_grad():
                    assert teacher is not None  # for type checkers
                    logits_t, feat_t = teacher(x)
            logits_s, feat_s = student(x)
            loss_task = ce(logits_s, y)
            if distill_active:
                loss_rkd = weights(logits_s, logits_t, feat_s, feat_t)
            else:
                loss_rkd = torch.zeros((), device=device)
            loss = loss_task + loss_rkd
            opt.zero_grad()
            loss.backward()
            if debug_active and batch_idx <= debug_max_batches:
                # Gradient global L2 norm
                total_norm = 0.0
                for p in student.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.detach().data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                epoch_grad_norms.append(total_norm)
                # Logits stats (pre-softmax)
                logits_mean = logits_s.detach().mean().item()
                logits_std = logits_s.detach().std(unbiased=False).item()
                epoch_logits_mean.append(logits_mean)
                epoch_logits_std.append(logits_std)
                # Prediction entropy (after softmax)
                probs = torch.softmax(logits_s.detach(), dim=1)
                ent = (-probs * (probs.clamp_min(1e-8).log())).sum(dim=1).mean().item()
                epoch_pred_entropy.append(ent)
                # Label entropy for the mini-batch (should reflect class diversity)
                with torch.no_grad():
                    counts = torch.bincount(y, minlength=probs.shape[1]).float()
                    p_lab = counts / counts.sum().clamp_min(1.0)
                    lab_ent = (-p_lab * (p_lab.clamp_min(1e-8).log())).sum().item()
                epoch_label_entropy.append(lab_ent)
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
            if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
                break
        # end epoch: aggregate and evaluate
        if n_batches > 0:
            avg_loss = total_loss / n_batches
            avg_task = total_task / n_batches
            avg_rkd = total_rkd / n_batches
            hist_loss.append(avg_loss)
            hist_task.append(avg_task)
            hist_distill.append(avg_rkd)
            # per-epoch accuracies
            train_acc = evaluate(student, train_loader)
            test_acc = evaluate(student, test_loader)
            hist_train_acc.append(train_acc)
            hist_test_acc.append(test_acc)
            if debug_active:
                debug["batch_grad_norms"].append(epoch_grad_norms)
                debug["batch_logits_mean"].append(epoch_logits_mean)
                debug["batch_logits_std"].append(epoch_logits_std)
                debug["batch_pred_entropy"].append(epoch_pred_entropy)
                debug["batch_label_entropy"].append(epoch_label_entropy)
            if cfg.verbose:
                print(
                    f"Epoch {epoch + 1}/{cfg.epochs} - loss: {avg_loss:.4f} (task: {avg_task:.4f}, distill: {avg_rkd:.4f}), train_acc: {train_acc:.2f}%, test_acc: {test_acc:.2f}%"
                )

    acc = evaluate(student, test_loader)
    result = {"test_acc": acc, "history": {"loss": hist_loss, "task": hist_task, "distill": hist_distill, "train_acc": hist_train_acc, "test_acc": hist_test_acc}}
    if debug_active:
        result["debug"] = debug
    return result
