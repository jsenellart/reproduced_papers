from __future__ import annotations

import pandas as pd
import torch

from common import load_runtime_ready_config
from QRNN.lib.data import build_dataloaders
from QRNN.lib.model import RNNRegressor


def test_rnn_regressor_matches_rnncell_unroll():
    torch.manual_seed(0)
    batch_size, seq_len, features = 2, 5, 3
    model = RNNRegressor(input_size=features, hidden_dim=4, layers=1, dropout=0.0)
    sample = torch.randn(batch_size, seq_len, features)

    rnn_outputs, _ = model.rnn(sample)
    final_hidden = rnn_outputs[:, -1]

    cell = torch.nn.RNNCell(features, model.rnn.hidden_size)
    cell.weight_ih.data.copy_(model.rnn.weight_ih_l0.data)
    cell.weight_hh.data.copy_(model.rnn.weight_hh_l0.data)
    cell.bias_ih.data.copy_(model.rnn.bias_ih_l0.data)
    cell.bias_hh.data.copy_(model.rnn.bias_hh_l0.data)

    hidden = torch.zeros(batch_size, model.rnn.hidden_size)
    for t in range(seq_len):
        hidden = cell(sample[:, t, :], hidden)

    assert torch.allclose(hidden, final_hidden, atol=1e-6)

    preds = model(sample)
    expected = model.head(final_hidden).squeeze(-1)
    assert torch.allclose(preds, expected, atol=1e-6)


def test_build_dataloaders_parses_alias_columns(tmp_path):
    rows = []
    for idx in range(20):
        rows.append(
            {
                "timestamp": f"2020-01-01 {idx:02d}:00:00",
                "Temperature (C)": 10.0 + idx * 0.5,
                "Humidity": 0.4 + idx * 0.01,
            }
        )
    csv_path = tmp_path / "synthetic_weather.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    cfg = load_runtime_ready_config()
    dataset_cfg = cfg["dataset"]
    dataset_cfg["path"] = str(csv_path)
    dataset_cfg["kaggle_dataset"] = None
    dataset_cfg["preprocess"] = None
    dataset_cfg["target_column"] = "temperature"
    dataset_cfg["feature_columns"] = ["temperature", "humidity"]
    dataset_cfg["sequence_length"] = 4
    dataset_cfg["prediction_horizon"] = 1
    dataset_cfg["batch_size"] = 2
    dataset_cfg["max_rows"] = None
    dataset_cfg["train_ratio"] = 0.5
    dataset_cfg["val_ratio"] = 0.25

    train_loader, val_loader, test_loader, metadata = build_dataloaders(cfg)

    assert metadata["input_size"] == 2
    assert metadata["splits"]["train"] > 0
    assert metadata["splits"]["val"] > 0
    assert metadata["splits"]["test"] > 0

    batch_sequences, batch_targets = next(iter(train_loader))
    assert batch_sequences.shape[-1] == 2
    assert batch_targets.ndim == 1
    assert len(val_loader.dataset) + len(test_loader.dataset) + len(train_loader.dataset) == sum(
        metadata["splits"].values()
    )
