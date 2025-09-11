#!/usr/bin/env python3
"""
Plot training loss and test accuracy curves (Figure 5-like) from saved JSON histories.

It scans models/ for files matching the naming convention and loads the
adjacent -loss.json histories. It then aggregates across seeds (by simple mean)
and plots two figures:
  - loss_vs_epoch.png
  - testacc_vs_epoch.png

Variants: Teacher, Scratch, KD, RKD. QRKD is optional; disabled by default.

Usage (from QRKD/):
  python scripts/plot_curves.py --epochs 10 --save-dir models --out-dir results --include-qrkd
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt


def load_histories(save_dir: str, dataset: str, epochs: int, include_qrkd: bool) -> Dict[str, List[dict]]:
    patterns = {
        "Teacher": f"{dataset}_teacher_seed*_e{epochs}*.pt",
        "Scratch": f"{dataset}_student-scratch_seed*_e{epochs}*.pt",
        "KD": f"{dataset}_student-kd_seed*_e{epochs}*.pt",
        "RKD": f"{dataset}_student-rkd_seed*_e{epochs}*.pt",
    }
    if include_qrkd:
        patterns["QRKD"] = f"{dataset}_student-qrkd_seed*_e{epochs}.pt"

    out: Dict[str, List[dict]] = defaultdict(list)
    for name, pat in patterns.items():
        for ckpt in glob.glob(os.path.join(save_dir, pat)):
            jpath = os.path.splitext(ckpt)[0] + "-loss.json"
            if not os.path.exists(jpath):
                continue
            try:
                with open(jpath, "r") as f:
                    hist = json.load(f)
                out[name].append(hist)
            except Exception:
                pass
    return out


def aggregate_curve(hists: List[dict], key: str) -> List[float]:
    if not hists:
        return []
    # Determine maximum possible epochs available across histories for this key
    max_len = 0
    for h in hists:
        seq = h.get(key, [])
        if isinstance(seq, list):
            max_len = max(max_len, len(seq))
    if max_len == 0:
        return []
    vals: List[float] = []
    for e in range(max_len):
        bucket = []
        for h in hists:
            seq = h.get(key, [])
            if isinstance(seq, list) and len(seq) > e:
                bucket.append(seq[e])
        if bucket:
            vals.append(mean(bucket))
        else:
            break
    return vals


def plot_curves(agg: Dict[str, Dict[str, List[float]]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(6,4))
    for name, series in agg.items():
        if series.get("loss"):
            plt.plot(series["loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss vs. epoch")
    plt.legend()
    loss_path = os.path.join(out_dir, "loss_vs_epoch.png")
    plt.tight_layout()
    plt.savefig(loss_path)
    plt.close()

    # Test accuracy
    plt.figure(figsize=(6,4))
    for name, series in agg.items():
        if series.get("test_acc"):
            plt.plot(series["test_acc"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy (%)")
    plt.title("Test accuracy vs. epoch")
    plt.legend()
    acc_path = os.path.join(out_dir, "testacc_vs_epoch.png")
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()

    print(f"Saved: {loss_path}\nSaved: {acc_path}")


def main():
    p = argparse.ArgumentParser(description="Plot loss and test accuracy curves from histories")
    p.add_argument("--dataset", default="mnist", choices=["mnist"]) 
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save-dir", default="models")
    p.add_argument("--out-dir", default="results")
    p.add_argument("--include-qrkd", action="store_true")
    args = p.parse_args()

    hists = load_histories(args.save_dir, args.dataset, args.epochs, args.include_qrkd)
    # Build aggregated series per variant
    agg: Dict[str, Dict[str, List[float]]] = {}
    for name, lst in hists.items():
        agg[name] = {
            "loss": aggregate_curve(lst, "loss"),
            "test_acc": aggregate_curve(lst, "test_acc"),
        }

    plot_curves(agg, args.out_dir)


if __name__ == "__main__":
    main()
