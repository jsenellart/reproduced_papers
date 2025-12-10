from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from runtime_lib.dtypes import describe_dtype, dtype_torch

from .data import build_dataloaders
from .model import RNNRegressor
from .training import fit


def _build_model(cfg: dict, input_size: int) -> RNNRegressor:
    model_cfg = cfg.get("model", {})
    params = model_cfg.get("params", {})
    hidden_dim = int(params.get("hidden_dim", 64))
    layers = int(params.get("layers", 1))
    dropout = float(params.get("dropout", 0.0))
    model = RNNRegressor(
        input_size=input_size, hidden_dim=hidden_dim, layers=layers, dropout=dropout
    )
    return model


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    logger = logging.getLogger(__name__)
    dtype_label = describe_dtype(cfg.get("dtype"))
    logger.info("Using dtype: %s", dtype_label)

    train_loader, val_loader, test_loader, metadata = build_dataloaders(cfg)
    model = _build_model(cfg, metadata["input_size"])

    logger.info(
        "Config summary | dataset=%s path=%s preprocess=%s seq_len=%d pred_horizon=%d batch=%d "
        "splits(train/val/test)=%d/%d/%d | model hidden=%d layers=%d dropout=%.2f | epochs=%d lr=%g",
        cfg.get("dataset", {}).get("name"),
        cfg.get("dataset", {}).get("path"),
        cfg.get("dataset", {}).get("preprocess"),
        cfg.get("dataset", {}).get("sequence_length"),
        cfg.get("dataset", {}).get("prediction_horizon"),
        cfg.get("dataset", {}).get("batch_size"),
        metadata.get("splits", {}).get("train", 0),
        metadata.get("splits", {}).get("val", 0),
        metadata.get("splits", {}).get("test", 0),
        cfg.get("model", {}).get("params", {}).get("hidden_dim"),
        cfg.get("model", {}).get("params", {}).get("layers"),
        cfg.get("model", {}).get("params", {}).get("dropout"),
        cfg.get("training", {}).get("epochs"),
        cfg.get("training", {}).get("lr"),
    )

    dtype = dtype_torch(cfg.get("dtype"))
    if dtype is not None:
        model = model.to(dtype=dtype)

    device = torch.device(cfg.get("device", "cpu"))
    cfg_with_metadata = dict(cfg)
    cfg_with_metadata["metadata"] = metadata
    metrics = fit(
        model, train_loader, val_loader, test_loader, cfg_with_metadata, run_dir
    )

    model_path = run_dir / "rnn_baseline.pt"
    torch.save(model.state_dict(), model_path)
    logger.info("Saved model checkpoint to %s", model_path)

    accuracies = _write_predictions_csv(
        model, cfg_with_metadata, train_loader, val_loader, test_loader, device, run_dir
    )
    if accuracies:
        metrics["accuracy"] = accuracies
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metadata["predictions_csv"] = str(run_dir / "predictions.csv")
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Saved metadata to %s", metadata_path)


def _build_prediction_loaders(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    batch_size: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for name, loader in (
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader),
    ):
        if loader is None or len(loader.dataset) == 0:  # type: ignore[arg-type]
            continue
        subset = loader.dataset
        loaders[name] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    return loaders


def _write_predictions_csv(
    model: torch.nn.Module,
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    run_dir: Path,
) -> dict[str, float | None]:
    model.eval()
    batch_size = int(cfg.get("dataset", {}).get("batch_size", 16))
    metadata = cfg.get("metadata", {}) or {}
    target_scale = metadata.get("target_abs_mean")
    target_mean = metadata.get("target_mean", 0.0)
    target_std = metadata.get("target_std", 1.0)
    loaders = _build_prediction_loaders(
        train_loader, val_loader, test_loader, batch_size
    )

    rows = []
    with torch.no_grad():
        for split, loader in loaders.items():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                preds_norm = model(batch_x).cpu().view(-1)
                targets_norm = batch_y.view(-1)
                preds = (preds_norm * target_std) + target_mean
                targets = (targets_norm * target_std) + target_mean
                for pred, target in zip(preds.tolist(), targets.tolist()):
                    rows.append({"split": split, "prediction": pred, "target": target})

    pred_path = run_dir / "predictions.csv"
    df = pd.DataFrame(rows)
    df.to_csv(pred_path, index=False)
    logging.getLogger(__name__).info("Saved predictions to %s", pred_path)

    accuracies: dict[str, float | None] = {}
    for split, split_df in df.groupby("split"):
        valid = split_df[split_df["target"] != 0]
        if valid.empty:
            accuracies[split] = None
            continue
        scale = target_scale if target_scale is not None else 0.0
        denom = np.clip(np.abs(valid["target"]), max(1e-6, scale), None)
        rel_error = (valid["target"] - valid["prediction"]) / denom
        rmse_rel = float(np.sqrt(np.mean(np.square(rel_error))))
        accuracies[split] = float((1 - rmse_rel) * 100)
    return accuracies


__all__ = ["train_and_evaluate"]
