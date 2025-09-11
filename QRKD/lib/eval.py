"""Reusable evaluation helpers (accuracy metrics, simple reporting)."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader


def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return accuracy in percent on the given loader."""
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
    return (correct / max(total, 1)) * 100.0
