from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    device: torch.device,
    target_scale: float | None = None,
    target_mean: float | None = None,
    target_std: float | None = None,
) -> tuple[float, float | None]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    rel_sq_sum = 0.0
    rel_count = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        mask = batch_y != 0
        if mask.any():
            t_mean = torch.tensor(
                target_mean if target_mean is not None else 0.0,
                device=batch_y.device,
                dtype=batch_y.dtype,
            )
            t_std = torch.tensor(
                target_std if target_std is not None else 1.0,
                device=batch_y.device,
                dtype=batch_y.dtype,
            )
            raw_t = batch_y[mask] * t_std + t_mean
            raw_p = preds[mask] * t_std + t_mean
            scale = target_scale if target_scale is not None else 0.0
            denom = torch.clamp(raw_t.abs(), min=max(1e-6, scale))
            rel = ((raw_t - raw_p) / denom).detach()
            rel_sq_sum += torch.sum(rel**2).item()
            rel_count += rel.numel()

    loss_value = total_loss / max(total_samples, 1)
    if rel_count == 0:
        return loss_value, None
    rmse_rel = (rel_sq_sum / rel_count) ** 0.5
    accuracy = (1 - rmse_rel) * 100
    return loss_value, float(accuracy)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float | None = None,
) -> float:
    model.train()
    running_loss = 0.0
    total_samples = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = batch_x.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    cfg: dict,
    run_dir: Path,
) -> dict:
    device = torch.device(cfg.get("device", "cpu"))
    model.to(device)

    training_cfg = cfg.get("training", {})
    epochs = int(training_cfg.get("epochs", 1))
    lr = float(training_cfg.get("lr", 1e-3))
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    grad_clip = training_cfg.get("clip_grad_norm")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    metadata = cfg.get("metadata", {}) or {}
    target_scale = metadata.get("target_abs_mean")
    target_mean = metadata.get("target_mean")
    target_std = metadata.get("target_std")

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        val_loss, val_acc = evaluate_metrics(
            model, val_loader, criterion, device, target_scale, target_mean, target_std
        )
        test_loss = test_acc = None
        if test_loader is not None and len(test_loader.dataset) > 0:  # type: ignore[arg-type]
            test_loss, test_acc = evaluate_metrics(
                model,
                test_loader,
                criterion,
                device,
                target_scale,
                target_mean,
                target_std,
            )

        LOGGER.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - val_acc=%s - test_acc=%s",
            epoch,
            epochs,
            train_loss,
            val_loss,
            f"{val_acc:.2f}%" if val_acc is not None else "n/a",
            f"{test_acc:.2f}%" if test_acc is not None else "n/a",
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
            }
        )

    last = history[-1]
    metrics = {
        "final_train_loss": last["train_loss"],
        "final_val_loss": last["val_loss"],
        "final_val_accuracy": last["val_accuracy"],
        "final_test_loss": last["test_loss"],
        "final_test_accuracy": last["test_accuracy"],
        "history": history,
    }

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    LOGGER.info("Saved metrics to %s", metrics_path)
    return metrics


__all__ = ["fit", "train_one_epoch", "evaluate_metrics"]
