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
from statistics import mean, pstdev
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


def aggregate_curve(hists: List[dict], key: str) -> tuple[List[float], List[float]]:
    """Return (mean_per_epoch, std_per_epoch)."""
    if not hists:
        return [], []
    max_len = 0
    for h in hists:
        seq = h.get(key, [])
        if isinstance(seq, list):
            max_len = max(max_len, len(seq))
    if max_len == 0:
        return [], []
    means: List[float] = []
    stds: List[float] = []
    for e in range(max_len):
        bucket = [h.get(key, [])[e] for h in hists if isinstance(h.get(key, []), list) and len(h.get(key, [])) > e]
        if not bucket:
            break
        if len(bucket) == 1:
            means.append(bucket[0])
            stds.append(0.0)
        else:
            means.append(mean(bucket))
            stds.append(pstdev(bucket))
    return means, stds


def plot_curves(agg: Dict[str, Dict[str, List[float]]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(6,4))
    for name, series in agg.items():
        loss_mean = series.get("loss_mean")
        loss_std = series.get("loss_std")
        if loss_mean:
            plt.plot(loss_mean, label=name)
            if loss_std:
                import numpy as np
                x = range(len(loss_mean))
                lm = loss_mean
                ls = loss_std
                plt.fill_between(x, [m - s for m,s in zip(lm,ls)], [m + s for m,s in zip(lm,ls)], alpha=0.15)
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
        acc_mean = series.get("test_acc_mean")
        acc_std = series.get("test_acc_std")
        if acc_mean:
            plt.plot(acc_mean, label=name)
            if acc_std:
                x = range(len(acc_mean))
                plt.fill_between(x, [m - s for m,s in zip(acc_mean,acc_std)], [m + s for m,s in zip(acc_mean,acc_std)], alpha=0.15)
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
        loss_mean, loss_std = aggregate_curve(lst, "loss")
        acc_mean, acc_std = aggregate_curve(lst, "test_acc")
        agg[name] = {
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "test_acc_mean": acc_mean,
            "test_acc_std": acc_std,
        }

    plot_curves(agg, args.out_dir)


if __name__ == "__main__":
    main()
