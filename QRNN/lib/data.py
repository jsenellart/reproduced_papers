from __future__ import annotations

import logging
import re
import shutil
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from runtime_lib.dtypes import dtype_torch

from .preprocess import PREPROCESSORS

LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"


def _find_first_csv(directory: Path) -> Path:
    candidates = sorted(directory.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found under downloaded dataset directory: {directory}"
        )
    return candidates[0]


def _resolve_project_path(raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None
    path_value = Path(raw_path).expanduser()
    if not path_value.is_absolute():
        path_value = (PROJECT_DIR / path_value).resolve()
    return path_value


def _derive_preprocessed_path(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.stem}.preprocess{raw_path.suffix}")


def _maybe_preprocess(csv_path: Path, dataset_cfg: dict) -> Path:
    preprocess_name = dataset_cfg.get("preprocess")
    if not preprocess_name:
        return csv_path

    if csv_path.name.endswith(".preprocess.csv") and csv_path.exists():
        return csv_path

    try:
        preprocessor = PREPROCESSORS[preprocess_name]
    except KeyError as exc:  # pragma: no cover - guardrail
        raise ValueError(f"Unknown dataset preprocess '{preprocess_name}'") from exc

    preprocessed_path = _derive_preprocessed_path(csv_path)
    return preprocessor(csv_path, preprocessed_path)


def resolve_dataset_path(dataset_cfg: dict) -> Path:
    """Resolve the dataset path, downloading via kagglehub only when needed."""

    configured_path = _resolve_project_path(dataset_cfg.get("path"))
    if configured_path and configured_path.exists():
        return _maybe_preprocess(configured_path, dataset_cfg)

    kaggle_dataset = dataset_cfg.get("kaggle_dataset")
    if kaggle_dataset:
        import kagglehub

        LOGGER.info("Downloading dataset '%s' via kagglehub", kaggle_dataset)
        dataset_dir = Path(kagglehub.dataset_download(kaggle_dataset)).resolve()
        csv_path = _find_first_csv(dataset_dir)

        destination = configured_path or (DATA_DIR / csv_path.name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            LOGGER.info("Using existing dataset at %s", destination)
            return _maybe_preprocess(destination, dataset_cfg)

        shutil.copyfile(csv_path, destination)
        LOGGER.info("Copied dataset to %s", destination)
        return _maybe_preprocess(destination, dataset_cfg)

    msg_path = configured_path or (DATA_DIR / "dataset.csv")
    raise FileNotFoundError(
        f"Dataset path {msg_path} does not exist and no kaggle_dataset is configured"
    )


def _sanitize_column_name(name: str) -> str:
    no_paren = re.sub(r"\([^)]*\)", "", name)
    return re.sub(r"[^a-z0-9]", "", no_paren.lower())


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    alias_map: dict[str, str] = {}
    for existing in df.columns:
        sanitized = _sanitize_column_name(existing)
        alias_map.setdefault(sanitized, existing)

    resolved: list[str] = []
    missing: list[str] = []
    for requested in columns:
        if requested in df.columns:
            resolved.append(requested)
            continue
        alias = alias_map.get(_sanitize_column_name(requested))
        if alias:
            resolved.append(alias)
        else:
            missing.append(requested)

    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[resolved].copy()


class WeatherSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Sliding-window time-series dataset for weather forecasting."""

    def __init__(
        self,
        data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        sequence_length: int,
        prediction_horizon: int,
        dtype: torch.dtype,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")

        self.features = _select_columns(data, feature_columns).to_numpy(dtype=float)
        self.targets = (
            _select_columns(data, [target_column]).to_numpy(dtype=float).ravel()
        )
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.dtype = dtype
        # Normalization stats (populated later from the training split)
        self.feature_mean = torch.zeros((1, self.features.shape[1]), dtype=self.dtype)
        self.feature_std = torch.ones((1, self.features.shape[1]), dtype=self.dtype)
        self.target_mean = torch.tensor(0.0, dtype=self.dtype)
        self.target_std = torch.tensor(1.0, dtype=self.dtype)

    def __len__(self) -> int:
        return max(
            0, len(self.targets) - self.sequence_length - self.prediction_horizon + 1
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = idx + self.sequence_length
        target_idx = end + self.prediction_horizon - 1
        raw_seq = torch.as_tensor(self.features[start:end], dtype=self.dtype)
        sequence = (raw_seq - self.feature_mean) / self.feature_std
        raw_target = torch.as_tensor(self.targets[target_idx], dtype=self.dtype)
        target = (raw_target - self.target_mean) / self.target_std
        return sequence, target

    def set_normalization(
        self,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> None:
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.target_mean = target_mean
        self.target_std = target_std

    def compute_stats_for_indices(
        self, indices: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_tensor = torch.as_tensor(self.features[indices], dtype=self.dtype)
        targets_tensor = torch.as_tensor(self.targets[indices], dtype=self.dtype)
        feat_mean = feature_tensor.mean(dim=0, keepdim=True)
        feat_std = feature_tensor.std(dim=0, keepdim=True).clamp_min(1e-6)
        tgt_mean = targets_tensor.mean()
        tgt_std = targets_tensor.std().clamp_min(1e-6)
        return feat_mean, feat_std, tgt_mean, tgt_std


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    dataset_cfg = cfg.get("dataset", {})
    dtype = dtype_torch(cfg.get("dtype")) or torch.float32

    csv_path = resolve_dataset_path(dataset_cfg)
    raw_df = pd.read_csv(csv_path)
    max_rows = dataset_cfg.get("max_rows")
    if max_rows is not None:
        raw_df = raw_df.head(int(max_rows))
    feature_columns = dataset_cfg.get("feature_columns") or []
    target_column = dataset_cfg.get("target_column")
    if not target_column:
        raise ValueError("dataset.target_column must be provided")

    sequence_length = int(dataset_cfg.get("sequence_length", 8))
    prediction_horizon = int(dataset_cfg.get("prediction_horizon", 1))

    dataset = WeatherSequenceDataset(
        raw_df,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        dtype=dtype,
    )

    total_len = len(dataset)
    if total_len == 0:
        raise ValueError(
            "Dataset is too small for the requested sequence and horizon lengths"
        )

    train_ratio = float(dataset_cfg.get("train_ratio", 0.7))
    val_ratio = float(dataset_cfg.get("val_ratio", 0.15))
    train_cutoff = int(total_len * train_ratio)
    val_cutoff = int(total_len * (train_ratio + val_ratio))
    train_indices = list(range(0, train_cutoff))
    val_indices = list(range(train_cutoff, val_cutoff))

    feat_mean, feat_std, tgt_mean, tgt_std = dataset.compute_stats_for_indices(
        train_indices
    )
    dataset.set_normalization(feat_mean, feat_std, tgt_mean, tgt_std)

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)
    test_indices = list(range(val_cutoff, total_len))
    test_ds = torch.utils.data.Subset(dataset, test_indices)

    batch_size = int(dataset_cfg.get("batch_size", 16))
    shuffle = bool(dataset_cfg.get("shuffle", True))

    sample_input, _ = dataset[0]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    metadata = {
        "input_size": sample_input.shape[-1],
        "sequence_length": sequence_length,
        "prediction_horizon": prediction_horizon,
        "feature_mean": dataset.feature_mean.squeeze().tolist(),
        "feature_std": dataset.feature_std.squeeze().tolist(),
        "target_mean": float(dataset.target_mean),
        "target_std": float(dataset.target_std),
        "target_abs_mean": float(
            torch.mean(torch.abs(torch.as_tensor(dataset.targets)))
        ),
        "dataset_path": str(csv_path),
        "splits": {
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }
    return train_loader, val_loader, test_loader, metadata


__all__ = ["WeatherSequenceDataset", "build_dataloaders", "resolve_dataset_path"]
