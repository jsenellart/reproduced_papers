"""Hyperparameter search for the quantum student on quantum-generated data.

Runs a grid of (optimizer, lr, epochs) configurations for QuantumClassifier at
depth=1 (single sandwich, matching the teacher structure) and logs every result
to a CSV file so you can compare later.

Usage
-----
    python work/hparam_search_quantum.py               # use defaults below
    python work/hparam_search_quantum.py --out my.csv  # custom output path

Edit the SEARCH_GRID and FIXED sections below to explore different settings.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, fields, asdict
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, random_split

from data import GENERATORS
from learner.photonic_quantum import validate_teacher_params


# ---------------------------------------------------------------------------
# ❶  Fixed experimental settings — change these to explore different regimes
# ---------------------------------------------------------------------------

FIXED = dict(
    generator    = "quantum",
    m            = 6,
    k            = 3,
    depth        = 1,
    m_circuit    = None,   # None → same as m; set e.g. to 8 for a larger interferometer
    dataset_size = 2000,
    test_fraction= 0.20,
    data_seed    = 42,
    batch_size   = 64,
)

# ---------------------------------------------------------------------------
# ❷  Hyperparameter grid — add / remove rows freely
# ---------------------------------------------------------------------------

@dataclass
class Config:
    optimizer : str
    lr        : float
    epochs    : int
    weight_decay: float = 0.0
    momentum  : float  = 0.9   # only used for SGD


SEARCH_GRID: list[Config] = [
    # --- Adam ---
    Config("Adam",  lr=1e-3,  epochs=300),
    Config("Adam",  lr=3e-3,  epochs=300),
    Config("Adam",  lr=1e-2,  epochs=300),
    Config("Adam",  lr=3e-2,  epochs=300),
    Config("Adam",  lr=1e-2,  epochs=600),
    Config("Adam",  lr=3e-3,  epochs=600),
    # --- AdamW ---
    Config("AdamW", lr=1e-3,  epochs=300, weight_decay=1e-4),
    Config("AdamW", lr=3e-3,  epochs=300, weight_decay=1e-4),
    Config("AdamW", lr=1e-2,  epochs=300, weight_decay=1e-4),
    Config("AdamW", lr=3e-3,  epochs=600, weight_decay=1e-4),
    # --- SGD + momentum ---
    Config("SGD",   lr=1e-2,  epochs=300, momentum=0.9),
    Config("SGD",   lr=3e-2,  epochs=300, momentum=0.9),
    Config("SGD",   lr=1e-1,  epochs=300, momentum=0.9),
    Config("SGD",   lr=3e-2,  epochs=600, momentum=0.9),
    # --- Gradient-free (lr/weight_decay/momentum ignored) ---
    # epochs = max objective function evaluations
    Config("CMA",       lr=0.0, epochs=2000),
    Config("CMA",       lr=0.0, epochs=5000),
    Config("Powell",    lr=0.0, epochs=2000),
    Config("NelderMead",lr=0.0, epochs=5000),
    Config("COBYLA",    lr=0.0, epochs=2000),
]


# ---------------------------------------------------------------------------
# Training helper
# ---------------------------------------------------------------------------

def run_config(
    cfg: Config,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test:  torch.Tensor,
    y_test:  torch.Tensor,
    m: int,
    k: int,
    depth: int,
    m_circuit: int | None,
    parity_modes,
    batch_size: int,
    model_seed: int = 0,
) -> tuple[float, float]:
    """Train one model and return (test_accuracy, wall_time_seconds)."""
    from learner.photonic_quantum import train_and_eval_quantum
    t0 = time.perf_counter()
    acc = train_and_eval_quantum(
        X_train, y_train, X_test, y_test,
        m=m, k=k, depth=depth, m_circuit=m_circuit,
        epochs=cfg.epochs,
        lr=cfg.lr,
        batch_size=batch_size,
        seed=model_seed,
        parity_modes=parity_modes,
        optimizer_name=cfg.optimizer,
    )
    elapsed = time.perf_counter() - t0
    return acc, elapsed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum student hyperparameter search")
    parser.add_argument("--out", default="work/hparam_results_quantum.csv",
                        help="Output CSV path (default: work/hparam_results_quantum.csv)")
    parser.add_argument("--model-seed", type=int, default=0)
    cli = parser.parse_args()

    # --- Dataset ---
    f = FIXED
    print(f"Generating dataset  generator={f['generator']}  m={f['m']}  k={f['k']}  size={f['dataset_size']} ...")
    ds = GENERATORS[f["generator"]](size=f["dataset_size"], m=f["m"], k=f["k"], seed=f["data_seed"])

    n_test  = int(len(ds.X) * f["test_fraction"])
    n_train = len(ds.X) - n_test
    train_ds, test_ds = random_split(
        TensorDataset(ds.X, ds.y), [n_train, n_test],
        generator=torch.Generator().manual_seed(f["data_seed"]),
    )
    X_train, y_train = train_ds[:][0], train_ds[:][1]
    X_test,  y_test  = test_ds[:][0],  test_ds[:][1]

    # Sanity check
    teacher_acc = validate_teacher_params(X_test, y_test, ds.metadata)
    parity_modes = ds.metadata.get("parity_modes", None)
    m_circuit = f["m_circuit"]
    print(f"Teacher (exact params) validation: {teacher_acc:.2%}  ← should be 100%")
    print(f"Train: {n_train}  Test: {n_test}  Depth: {f['depth']}  m_circuit: {m_circuit or f['m']}\n")

    # --- CSV setup ---
    out_path = Path(cli.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_fields = [fld.name for fld in fields(Config)]
    csv_header = cfg_fields + ["accuracy", "time_s"]

    # --- Grid search ---
    print(f"{'optimizer':>6}  {'lr':>8}  {'epochs':>6}  {'wd':>8}  {'mom':>5}  {'accuracy':>10}  {'time':>8}")
    print("-" * 65)

    rows = []
    for i, cfg in enumerate(SEARCH_GRID, 1):
        acc, elapsed = run_config(
            cfg, X_train, y_train, X_test, y_test,
            m=f["m"], k=f["k"], depth=f["depth"],
            m_circuit=m_circuit,
            parity_modes=parity_modes,
            batch_size=f["batch_size"],
            model_seed=cli.model_seed,
        )
        print(f"{cfg.optimizer:>6}  {cfg.lr:>8.0e}  {cfg.epochs:>6}  "
              f"{cfg.weight_decay:>8.0e}  {cfg.momentum:>5.2f}  "
              f"{acc:>10.2%}  {elapsed:>7.1f}s  [{i}/{len(SEARCH_GRID)}]")
        rows.append({**asdict(cfg), "accuracy": acc, "time_s": round(elapsed, 2)})

    with open(out_path, "w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {out_path}")

    # Quick summary: top 3
    rows.sort(key=lambda r: r["accuracy"], reverse=True)
    print("\nTop 3:")
    for r in rows[:3]:
        print(f"  {r['optimizer']:>6}  lr={r['lr']:.0e}  epochs={r['epochs']}  "
              f"wd={r['weight_decay']:.0e}  → {r['accuracy']:.2%}")
