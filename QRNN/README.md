# QRNN — Quantum Recurrent Neural Networks (Neural Networks, 2023)

This project bootstraps a reproduction of the paper on quantum recurrent neural networks (QRNN, Neural Networks, 2023). It currently ships a classical RNN baseline over weather time-series data and a scaffold for extending the work toward the quantum architecture.

## Reference and Attribution

- Paper: Quantum Recurrent Neural Networks for sequential learning (Neural Networks, 2023)
- Authors: Yanan Li, Zhimin Wang, Rongbing Han, Shangshang Shi, Jiaxin Li, Ruimin Shang, Haiyong Zheng, Guoqiang Zhong, Yongjian Gu
- DOI/ArXiv: https://doi.org/10.1016/j.neunet.2023.07.003 (publisher page: https://www.sciencedirect.com/science/article/abs/pii/S089360802300360X)
- Original repository (if any): not referenced here
- License and attribution notes: cite the published Neural Networks article when using results derived from this code.

## Overview

The reproduction is staged:

- **Stage 1 (implemented here):** classical RNN baseline for sequence forecasting on a meteorological dataset.
- **Stage 2:** swap in the QRNN architecture described in the paper and compare against the baseline metrics.

Defaults target the Kaggle Szeged weather dataset (`budincsevity/szeged-weather`) with a preprocessing step that aggregates daily statistics. A small synthetic CSV (`data/sample_weather.csv`) remains available via the example config for quick local smoke tests.

## How to Run

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Command-line interface

Use the repository-level runner (`implementation.py` at the repo root) with `--project QRNN`.

```bash
# From the repo root
python implementation.py --project QRNN --help

# From inside this folder
python ../implementation.py --project QRNN --help
```

Common options (see `configs/cli.json` for the full schema alongside the global flags injected by the shared runner):

- `--config PATH` Load an additional JSON config (merged over defaults).
- `--epochs INT` Override `training.epochs`.
- `--batch-size INT` Override `dataset.batch_size`.
- `--sequence-length INT` History length used for forecasting.
- `--hidden-dim INT` RNN hidden dimension.
- Standard global flags: `--seed`, `--dtype`, `--device`, `--log-level`, `--outdir`.

Example runs:

```bash
# Use the bundled synthetic CSV for a quick smoke test (from repo root)
python implementation.py --project QRNN --config QRNN/configs/example.json --epochs 1

# Same run from inside QRNN/
python ../implementation.py --project QRNN --config configs/example.json --epochs 1

# Train on the Szeged Kaggle dataset (downloads on first run if needed)
python implementation.py --project QRNN --outdir runs/qrnn_baseline
```

### Dataset setup & preprocessing

- The default configuration points to `data/szeged-weather.csv` and sets `dataset.preprocess="szeged_weather"`. If the raw file is missing, it will be downloaded from `budincsevity/szeged-weather`, then preprocessed into `data/szeged-weather.preprocess.csv` the first time. Subsequent runs reuse the preprocessed file.
- The Szeged-specific preprocessor consumes `Formatted Date`, `Temperature (C)`, `Humidity`, `Wind Speed (km/h)`, and `Pressure (millibars)` and outputs a daily CSV with columns: `date`, `min_temperature`, `max_temperature`, `avg_humidity`, `avg_wind_speed`, `avg_pressure`.
- To run on the small bundled sample instead, use `configs/example.json` (no preprocessing) which points to `data/sample_weather.csv`.
- Additional dataset-specific preprocessors can be registered in `lib/preprocess.py`; set `dataset.preprocess` to the corresponding key to enable them. If a `<file>.preprocess.csv` already exists, it is reused without rebuilding.
- Use `dataset.max_rows` to cap the number of rows ingested from the CSV (applied before splitting). Normalization stats are computed on the resulting training split to avoid leakage.

### Outputs

Each run writes to `<outdir>/run_YYYYMMDD-HHMMSS/` and includes:

- `config_snapshot.json` — resolved configuration used for the run
- `metrics.json` — train/validation loss history
- `predictions.csv` — reference vs model predictions for train/val/test splits
- `metadata.json` — dataset and preprocessing metadata
- `rnn_baseline.pt` — PyTorch checkpoint for the baseline model
- `done.txt` — completion marker

Plotting utility: `python utils/plot_predictions.py <run_dir>` renders a matplotlib view of reference vs prediction and saves `predictions_plot.png` inside the run folder.

## Configuration

Key files under `configs/`:

- `defaults.json` — baseline hyperparameters and dataset paths
- `example.json` — example experiment overriding sequence length and batch size
- `cli.json` — CLI schema consumed by the shared runner

Precision control: include `"dtype"` (e.g., `"float32"`) at the top level or under `model` to run in a specific torch dtype.

## Results and Next Steps

- Baseline metrics: mean squared error on validation splits of the weather sequences (see `metrics.json`).
- Planned extensions: implement the QRNN cell described in the paper, run ablations versus the classical RNN, and add visualization notebooks for sequence reconstruction.

## Testing

Run tests from inside the `QRNN/` directory:

```bash
cd QRNN
pytest -q
```

Tests cover the CLI, config loading, and a smoke run of the training loop on the synthetic dataset.
