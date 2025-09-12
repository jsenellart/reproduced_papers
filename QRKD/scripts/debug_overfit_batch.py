#!/usr/bin/env python3
"""Minimal overfit-batch diagnostic.

Purpose:
  - Detect pathological non-learning seeds early.
  - Runs a single batch repeatedly (by reusing its tensors) and tracks
    loss + accuracy over a number of steps. A healthy setup should drive
    loss -> ~0 and accuracy -> 100% quickly.

Usage (from QRKD/):
  QRKD_DEBUG=1 python scripts/debug_overfit_batch.py --steps 200 --seed 5

Outputs:
  Prints every 10 steps: step, loss, accuracy, grad_norm.
  Final JSON (optional) if --json-out provided, containing the curves.
"""
from __future__ import annotations
import argparse, json, os
import torch
import torch.nn as nn
from torch.optim import Adam
from lib.datasets import DataConfig, mnist_loaders
from lib.models import StudentCNN
from lib.utils import set_seed


def grad_l2(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            total += g.pow(2).sum().item()
    return total ** 0.5


def run(seed: int, steps: int, lr: float, batch_size: int, device: str, json_out: str | None):
    set_seed(seed)
    dcfg = DataConfig(batch_size=batch_size)
    train_loader, _ = mnist_loaders(dcfg)
    # Grab one batch
    x0, y0 = next(iter(train_loader))
    x0, y0 = x0.to(device), y0.to(device)
    model = StudentCNN().to(device)
    opt = Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    losses = []
    accs = []
    grad_norms = []
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        logits, _ = model(x0)
        loss = ce(logits, y0)
        loss.backward()
        gnorm = grad_l2(model)
        opt.step()
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            acc = (pred == y0).float().mean().item() * 100.0
        losses.append(loss.item())
        accs.append(acc)
        grad_norms.append(gnorm)
        if step % 10 == 0 or step == 1:
            print(f"[Overfit] step={step:04d} loss={loss.item():.4f} acc={acc:.2f}% grad_norm={gnorm:.2f}")
    if json_out:
        payload = {"loss": losses, "acc": accs, "grad_norm": grad_norms}
        with open(json_out, "w") as f:
            json.dump(payload, f)
        print(f"[Saved] {json_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Overfit a single batch for debugging")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json-out", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.seed, args.steps, args.lr, args.batch_size, args.device, args.json_out)
