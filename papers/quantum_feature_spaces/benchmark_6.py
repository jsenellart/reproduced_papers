#!/usr/bin/env python
"""benchmark_6.py — full 3 × 4 grid of learner × generator experiments at m=6, k=3.

Three learners  ×  four generators  =  twelve cells:

                            ┌────────── learner ──────────┐
                            │  mlp   photonic   qubit     │
   ┌─photonic_quantum───────┼─────────────────────────────┤
   ├─qubit_quantum──────────┤   (12 cells covered)        │
   ├─analytical─────────────┤                             │
   └─mlp───────────────────┘                              │

For each cell we look for the minimum model configuration that reaches
``--target-accuracy 0.90`` and print a summary table.

Usage
-----
    python work/benchmark_6.py

Each experiment streams live output to the terminal and is also saved to
/tmp/bench_<learner>_<generator>.log for post-hoc inspection.
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

_print_lock = threading.Lock()

PYTHON = sys.executable
TRAIN  = os.path.join(os.path.dirname(__file__), "train.py")

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

# Shared CLI flags — same problem size and dataset filter for every cell so
# the cross-comparison is apples-to-apples.
#
# `--bail-threshold 0` lets the resampler always reach --dataset-size, even
# when --balanced+--min-margin combined yield is below the default 30 %
# guard (e.g. for the mlp generator with margin=0.10 at m=4 — not at m=6,
# but kept for safety across the whole grid).
BASE = [
    "--m", "8", "--k", "4",
    "--target-accuracy", "0.90",
    "--balanced",
    "--epochs", "300",
    "--early-stopping-patience", "50",
    "--min-margin", "0.10",
    "--dataset-size", "5000",
    "--bail-threshold", "0"
]

# `hloss` works with every generator (it only needs hard 0/1 labels and a
# trainable scalar bias, no soft teacher targets) — so it's the natural
# loss for every quantum-learner cell. `mse` would also work but only when
# the generator emits soft parity targets (photonic_quantum or qubit_quantum).
QUANTUM_DEPTHS = ["3", "4", "5"]
QUANTUM_SIZES  = ["8", "10", "12"]
MLP_HIDDEN     = ["32", "32,32", "32,32,32", "32,32,32,32"]


def _mlp_args(generator: str) -> list[str]:
    return [
        "--learner", "mlp",
        "--generator", generator,
        "--hidden-sizes", *MLP_HIDDEN,
    ]


def _photonic_args(generator: str) -> list[str]:
    return [
        "--learner", "photonic_quantum",
        "--generator", generator,
        "--loss", "hloss",
        "--depths", *QUANTUM_DEPTHS,
        "--sizes",  *QUANTUM_SIZES,
    ]


def _qubit_args(generator: str) -> list[str]:
    # The qubit learner ignores --sizes (qubit count is fixed at m-1).
    # Adam + hloss converges much faster than SPSA on small budgets;
    # SPSA matches the paper but typically needs more iterations.
    return [
        "--learner", "qubit_quantum",
        "--generator", generator,
        "--loss", "hloss",
        "--depths", "0", "1", "2", "3", "4",
    ]


GENERATORS = ["photonic_quantum", "qubit_quantum", "analytical", "mlp"]
GEN_LABEL  = {                       # for nice display in the log/summary
    "photonic_quantum": "photonic_quantum",
    "qubit_quantum":    "qubit_quantum",
    "analytical":       "analytical",
    "mlp":              "mlp-teacher",
}


def _build_experiments() -> list[dict]:
    out: list[dict] = []
    for gen in GENERATORS:
        out.append({
            "label":     f"mlp / {GEN_LABEL[gen]}",
            "learner":   "mlp",
            "generator": gen,
            "args":      _mlp_args(gen),
        })
    for gen in GENERATORS:
        out.append({
            "label":     f"photonic_quantum / {GEN_LABEL[gen]}",
            "learner":   "photonic_quantum",
            "generator": gen,
            "args":      _photonic_args(gen),
        })
    for gen in GENERATORS:
        out.append({
            "label":     f"qubit_quantum / {GEN_LABEL[gen]}",
            "learner":   "qubit_quantum",
            "generator": gen,
            "args":      _qubit_args(gen),
        })
    return out


EXPERIMENTS: list[dict] = _build_experiments()

# ---------------------------------------------------------------------------
# Result data class
# ---------------------------------------------------------------------------

@dataclass
class Result:
    label:        str
    reached:      bool            = False
    config:       str             = "—"   # e.g. "depth=1, m_c=6" or "hidden=64"
    params:       int | None      = None
    accuracy:     float | None    = None  # accuracy at first hit
    best_accuracy: float | None   = None  # best seen even if target not reached
    wall_sec:     float           = 0.0
    log_path:     str             = ""
    lines:        list[str]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# photonic multi-size output:  "     6       1        66     100.00% <-- first hit"
_RE_Q_MULTI  = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s*<--\s*first hit", re.IGNORECASE
)
# photonic single-size and qubit output: "     1        66     100.00% <-- first hit"
_RE_Q_SINGLE = re.compile(
    r"^\s*(\d+)\s+(\d+)\s+([\d.]+)%\s*<--\s*first hit", re.IGNORECASE
)
# mlp output: "                  64      91.20% <-- first hit"
_RE_MLP      = re.compile(
    r"^\s*(\S+)\s+([\d.]+)%\s*<--\s*first hit", re.IGNORECASE
)

# Any table row (with or without <-- first hit) — used to track best accuracy
_RE_Q_MULTI_ROW  = re.compile(r"^\s*\d+\s+\d+\s+\d+\s+([\d.]+)%")
_RE_Q_SINGLE_ROW = re.compile(r"^\s*\d+\s+\d+\s+([\d.]+)%")
_RE_MLP_ROW      = re.compile(r"^\s*\S+\s+([\d.]+)%")


def _parse_result(r: Result, learner: str) -> None:
    """Parse captured lines and fill r.reached / r.config / r.params / r.accuracy.

    The learner kind disambiguates the layouts: photonic-quantum may have a
    multi-size table (4 columns) while qubit-quantum and single-size photonic
    use the 3-column layout shared with mlp's 2-column layout.

    Also tracks the best accuracy seen across all table rows, so it can be
    reported in the summary even when the target was not reached.
    """
    best: float = -1.0

    for line in r.lines:
        # --- track best accuracy from any table row ---
        if learner == "photonic_quantum":
            bm = _RE_Q_MULTI_ROW.match(line)
            if bm:
                best = max(best, float(bm.group(1)) / 100.0)
            else:
                bm = _RE_Q_SINGLE_ROW.match(line)
                if bm:
                    best = max(best, float(bm.group(1)) / 100.0)
        elif learner == "qubit_quantum":
            bm = _RE_Q_SINGLE_ROW.match(line)
            if bm:
                best = max(best, float(bm.group(1)) / 100.0)
        elif learner == "mlp":
            bm = _RE_MLP_ROW.match(line)
            if bm:
                best = max(best, float(bm.group(1)) / 100.0)

        # --- detect first-hit line ---
        if not r.reached:
            if learner == "photonic_quantum":
                m = _RE_Q_MULTI.match(line)
                if m:
                    m_c, dep, par, acc = m.groups()
                    r.reached  = True
                    r.config   = f"m_c={m_c}, depth={dep}"
                    r.params   = int(par)
                    r.accuracy = float(acc) / 100.0
                    continue
                m = _RE_Q_SINGLE.match(line)
                if m:
                    dep, par, acc = m.groups()
                    r.reached  = True
                    r.config   = f"depth={dep}"
                    r.params   = int(par)
                    r.accuracy = float(acc) / 100.0
                    continue
            elif learner == "qubit_quantum":
                m = _RE_Q_SINGLE.match(line)
                if m:
                    dep, par, acc = m.groups()
                    r.reached  = True
                    r.config   = f"depth={dep}"
                    r.params   = int(par)
                    r.accuracy = float(acc) / 100.0
                    continue
            elif learner == "mlp":
                m = _RE_MLP.match(line)
                if m:
                    hidden, acc = m.groups()
                    r.reached  = True
                    r.config   = f"hidden={hidden}"
                    r.accuracy = float(acc) / 100.0
                    continue

    if best >= 0.0:
        r.best_accuracy = best


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def run_experiment(exp: dict, idx: int, n_total: int) -> Result:
    cmd = [PYTHON, TRAIN] + BASE + exp["args"]
    log_path = f"/tmp/bench_{exp['learner']}_{exp['generator']}.log"
    prefix = f"[{exp['label']}] "

    header = (
        f"\n{'=' * 70}\n"
        f"[{idx}/{n_total}]  {exp['label']}\n"
        f"cmd: python train.py {' '.join(BASE + exp['args'])}\n"
        f"log: {log_path}\n"
        f"{'=' * 70}\n"
    )
    with _print_lock:
        print(header, flush=True)

    result = Result(label=exp["label"], log_path=log_path)
    t0 = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
            cwd=os.path.dirname(TRAIN),
        )
        for line in proc.stdout:
            with _print_lock:
                sys.stdout.write(prefix + line)
                sys.stdout.flush()
            log_f.write(line)
            result.lines.append(line.rstrip())
        proc.wait()

    result.wall_sec = time.time() - t0
    _parse_result(result, exp["learner"])
    return result


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _fmt_acc(r: Result) -> str:
    if r.accuracy is not None:
        return f"{r.accuracy:.1%}"
    if r.best_accuracy is not None:
        return f"{r.best_accuracy:.1%}*"
    return "  —  "

def _fmt_params(r: Result) -> str:
    if r.params is None:
        return "  —  "
    return str(r.params)

def print_summary(results: list[Result]) -> None:
    col_label  = max(len(r.label)  for r in results) + 2
    col_config = max(len(r.config) for r in results) + 2
    w = col_label + col_config + 34

    sep = "─" * w
    print(f"\n{'─' * w}")
    print(f"{'BENCHMARK SUMMARY':^{w}}")
    print(f"{'─' * w}")
    print(
        f"  {'experiment':<{col_label}}"
        f"  {'reached 90%':<12}"
        f"  {'min config':<{col_config}}"
        f"  {'params':>7}"
        f"  {'accuracy':>9}"
        f"  {'wall time':>10}"
    )
    print(sep)
    for r in results:
        reached_str = "✓ YES" if r.reached else "✗ no"
        wall_str    = f"{r.wall_sec/60:.1f} min"
        print(
            f"  {r.label:<{col_label}}"
            f"  {reached_str:<12}"
            f"  {r.config:<{col_config}}"
            f"  {_fmt_params(r):>7}"
            f"  {_fmt_acc(r):>9}"
            f"  {wall_str:>10}"
        )
    print(sep)
    print(f"\nLogs saved to /tmp/bench_<learner>_<generator>.log")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print each command without running it, then exit.")
    args = ap.parse_args()

    if args.dry_run:
        for exp in EXPERIMENTS:
            cmd = " ".join([PYTHON, TRAIN] + BASE + exp["args"])
            print(f"[{exp['label']}]\n  {cmd}\n")
        sys.exit(0)

    t_start = time.time()
    n = len(EXPERIMENTS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(run_experiment, exp, i, n): exp
            for i, exp in enumerate(EXPERIMENTS, 1)
        }
        label_order = {exp["label"]: i for i, exp in enumerate(EXPERIMENTS)}
        results: list[Result] = []
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: label_order[r.label])
    print_summary(results)
    print(f"Total wall time: {(time.time() - t_start) / 60:.1f} min\n")
