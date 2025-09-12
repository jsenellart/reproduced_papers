#!/usr/bin/env python3
"""
Run multi-seed trainings for Teacher and Students (scratch, KD, RKD) on MNIST
and report mean/std metrics like Table 1.

Usage (from QRKD/):
  python scripts/run_table1_sweep.py --epochs 10 --seeds 0 1 2 3 4
  # or
  python scripts/run_table1_sweep.py --epochs 10 --num-seeds 5 --seed-base 0

Options:
  --checkrun limits each epoch to 50 batches for a quick dry-run.
  --data-dir sets a custom MNIST root.

Artifacts:
  - Saves checkpoints under models/ with the same naming convention as the CLI
  - Saves loss histories as sibling JSON files with "-loss.json" suffix
  - Prints aggregate mean±std metrics at the end
"""

from __future__ import annotations

import argparse
import json
import os
import csv
from statistics import mean, pstdev
from typing import Dict, List, Tuple

import torch

import sys
sys.path.append(os.path.abspath('.'))  # allow `from lib.*` when run from QRKD/

from lib.datasets import DataConfig, mnist_loaders
from lib.eval import evaluate_accuracy
from lib.losses import QRKDWeights
from lib.models import StudentCNN, TeacherCNN
from lib.train import TrainConfig, train_student, train_teacher
from lib.utils import set_seed


def save_model(model: torch.nn.Module, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    return path


def sweep(
    seeds: List[int],
    epochs: int,
    batch_size: int,
    lr: float,
    dataset: str = "mnist",
    data_dir: str | None = None,
    save_dir: str = "models",
    checkrun: bool = False,
) -> None:
    assert dataset.lower() == "mnist", "Only MNIST is supported in this script."

    dcfg = (
        DataConfig(batch_size=batch_size)
        if data_dir is None
        else DataConfig(batch_size=batch_size, root=data_dir)
    )

    all_metrics: Dict[str, List[Tuple[float, float, float]]] = {
        "Teacher": [],
        "Scratch": [],
        "KD": [],
        "RKD": [],
    }

    # Phase 1: train all teachers (or reuse if exists), collect metrics
    chk = "_chk" if checkrun else ""
    teacher_paths: Dict[int, str] = {}
    teacher_tests: Dict[int, float] = {}
    def load_final_acc(json_path: str) -> tuple[float | None, float | None]:
        """Load last epoch (train_acc, test_acc) from history JSON if present."""
        if not os.path.exists(json_path):
            return None, None
        try:
            with open(json_path, "r") as f:
                hist = json.load(f)
            train_hist = hist.get("train_acc") or []
            test_hist = hist.get("test_acc") or []
            tr = train_hist[-1] if train_hist else None
            te = test_hist[-1] if test_hist else None
            return tr, te
        except Exception:
            return None, None

    for seed in seeds:
        set_seed(seed)
        train_loader, test_loader = mnist_loaders(dcfg)
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            verbose=True,
            max_batches=50 if checkrun else None,
        )
        t_fname = f"{dataset}_teacher_seed{seed}_e{epochs}{chk}.pt"
        t_path = os.path.join(save_dir, t_fname)
        if os.path.exists(t_path):
            print(f"[Reuse] Teacher exists: {t_path}")
            teacher = TeacherCNN()
            teacher.load_state_dict(torch.load(t_path, map_location=torch.device(cfg.device)))
            teacher.eval()
            # Load metrics from JSON if available
            json_path = os.path.splitext(t_path)[0] + "-loss.json"
            t_train, t_test = load_final_acc(json_path)
            if t_train is None or t_test is None:
                # Fallback evaluation only if missing
                t_train = evaluate_accuracy(teacher, train_loader, torch.device(cfg.device))
                t_test = evaluate_accuracy(teacher, test_loader, torch.device(cfg.device))
        else:
            teacher = TeacherCNN()
            teacher, hist_t = train_teacher(teacher, train_loader, cfg, test_loader)
            t_path = save_model(teacher, save_dir, t_fname)
            with open(os.path.splitext(t_path)[0] + "-loss.json", "w") as f:
                json.dump(hist_t, f)
            # Use freshly produced history for metrics
            t_train = hist_t.get("train_acc", [None])[-1]
            t_test = hist_t.get("test_acc", [None])[-1]
        all_metrics["Teacher"].append((t_train, t_test, t_train - t_test))
        teacher_paths[seed] = t_path
        teacher_tests[seed] = t_test

    # Choose best teacher by test accuracy
    best_seed = max(teacher_tests, key=teacher_tests.get)
    best_teacher_path = teacher_paths[best_seed]
    print(f"[Select] Best teacher seed={best_seed} test={teacher_tests[best_seed]:.2f}% -> {best_teacher_path}")

    # Phase 2: for each seed, train Scratch (no teacher), KD and RKD using the best teacher
    for seed in seeds:
        set_seed(seed)
        train_loader, test_loader = mnist_loaders(dcfg)
        cfg = TrainConfig(
            epochs=epochs,
            lr=lr,
            verbose=True,
            max_batches=50 if checkrun else None,
        )

        # Scratch
        sc_fname = f"{dataset}_student-scratch_seed{seed}_e{epochs}{chk}.pt"
        sc_path = os.path.join(save_dir, sc_fname)
        if os.path.exists(sc_path):
            print(f"[Reuse] Scratch exists: {sc_path}")
            scratch = StudentCNN()
            scratch.load_state_dict(torch.load(sc_path, map_location=torch.device(cfg.device)))
            scratch.eval()
            sc_json = os.path.splitext(sc_path)[0] + "-loss.json"
            sc_train, sc_test = load_final_acc(sc_json)
            if sc_train is None or sc_test is None:
                sc_train = evaluate_accuracy(scratch, train_loader, torch.device(cfg.device))
                sc_test = evaluate_accuracy(scratch, test_loader, torch.device(cfg.device))
        else:
            scratch = StudentCNN()
            res_sc = train_student(scratch, None, train_loader, test_loader, cfg, QRKDWeights(kd=0.0, dr=0.0, ar=0.0), student_name="scratch")
            sc_path = save_model(scratch, save_dir, sc_fname)
            hist_s = res_sc.get("history") if isinstance(res_sc, dict) else None
            if isinstance(hist_s, dict):
                with open(os.path.splitext(sc_path)[0] + "-loss.json", "w") as f:
                    json.dump(hist_s, f)
                sc_train = hist_s.get("train_acc", [None])[-1]
                sc_test = hist_s.get("test_acc", [None])[-1]
        all_metrics["Scratch"].append((sc_train, sc_test, sc_train - sc_test))

        # Load best teacher once per seed
        best_teacher = TeacherCNN()
        best_teacher.load_state_dict(torch.load(best_teacher_path, map_location=torch.device(cfg.device)))
        best_teacher.eval()

        # KD
        kd_fname = f"{dataset}_student-kd_seed{seed}_e{epochs}{chk}.pt"
        kd_path = os.path.join(save_dir, kd_fname)
        if os.path.exists(kd_path):
            print(f"[Reuse] KD exists: {kd_path}")
            kd = StudentCNN()
            kd.load_state_dict(torch.load(kd_path, map_location=torch.device(cfg.device)))
            kd.eval()
            kd_json = os.path.splitext(kd_path)[0] + "-loss.json"
            kd_train, kd_test = load_final_acc(kd_json)
            if kd_train is None or kd_test is None:
                kd_train = evaluate_accuracy(kd, train_loader, torch.device(cfg.device))
                kd_test = evaluate_accuracy(kd, test_loader, torch.device(cfg.device))
        else:
            kd = StudentCNN()
            res_kd = train_student(kd, best_teacher, train_loader, test_loader, cfg, QRKDWeights(kd=0.5, dr=0.0, ar=0.0), student_name="kd")
            kd_path = save_model(kd, save_dir, kd_fname)
            hist_kd = res_kd.get("history") if isinstance(res_kd, dict) else None
            if isinstance(hist_kd, dict):
                with open(os.path.splitext(kd_path)[0] + "-loss.json", "w") as f:
                    json.dump(hist_kd, f)
                kd_train = hist_kd.get("train_acc", [None])[-1]
                kd_test = hist_kd.get("test_acc", [None])[-1]
        all_metrics["KD"].append((kd_train, kd_test, kd_train - kd_test))

        # RKD
        rkd_fname = f"{dataset}_student-rkd_seed{seed}_e{epochs}{chk}.pt"
        rkd_path = os.path.join(save_dir, rkd_fname)
        if os.path.exists(rkd_path):
            print(f"[Reuse] RKD exists: {rkd_path}")
            rkd = StudentCNN()
            rkd.load_state_dict(torch.load(rkd_path, map_location=torch.device(cfg.device)))
            rkd.eval()
            rkd_json = os.path.splitext(rkd_path)[0] + "-loss.json"
            rkd_train, rkd_test = load_final_acc(rkd_json)
            if rkd_train is None or rkd_test is None:
                rkd_train = evaluate_accuracy(rkd, train_loader, torch.device(cfg.device))
                rkd_test = evaluate_accuracy(rkd, test_loader, torch.device(cfg.device))
        else:
            rkd = StudentCNN()
            res_rkd = train_student(rkd, best_teacher, train_loader, test_loader, cfg, QRKDWeights(kd=0.0, dr=0.1, ar=0.1), student_name="rkd")
            rkd_path = save_model(rkd, save_dir, rkd_fname)
            hist_rkd = res_rkd.get("history") if isinstance(res_rkd, dict) else None
            if isinstance(hist_rkd, dict):
                with open(os.path.splitext(rkd_path)[0] + "-loss.json", "w") as f:
                    json.dump(hist_rkd, f)
                rkd_train = hist_rkd.get("train_acc", [None])[-1]
                rkd_test = hist_rkd.get("test_acc", [None])[-1]
        all_metrics["RKD"].append((rkd_train, rkd_test, rkd_train - rkd_test))

    # Aggregate and print (mean ± std)
    def agg(rows: List[Tuple[float, float, float]]):
        trains = [a for a, _, _ in rows]
        tests = [b for _, b, _ in rows]
        gaps = [g for _, _, g in rows]
        return (
            mean(trains), pstdev(trains),
            mean(tests), pstdev(tests),
            mean(gaps), pstdev(gaps),
        )

    # Derived gaps per seed
    t_tests = [x[1] for x in all_metrics["Teacher"]]
    sc_tests = [x[1] for x in all_metrics["Scratch"]]
    kd_tests = [x[1] for x in all_metrics["KD"]]
    rkd_tests = [x[1] for x in all_metrics["RKD"]]
    ts_gap_kd = [t - s for t, s in zip(t_tests, kd_tests)]
    ts_gap_rkd = [t - s for t, s in zip(t_tests, rkd_tests)]
    dist_gain_kd = [s - sc for s, sc in zip(kd_tests, sc_tests)]
    dist_gain_rkd = [s - sc for s, sc in zip(rkd_tests, sc_tests)]

    print("\nTable 1 (mean ± std over seeds)")
    print("Model, Train Acc, Test Acc, Acc. Gap, T&S Gap, Dist. Gain")

    def fmt(m, s):
        return f"{m:.2f} ± {s:.2f}"

    summary_rows = []  # for summary CSV
    for name in ["Teacher", "Scratch", "KD", "RKD"]:
        m_tr, s_tr, m_te, s_te, m_gp, s_gp = agg(all_metrics[name])
        if name in ("KD", "RKD"):
            if name == "KD":
                ts_gap_mean, ts_gap_std = mean(ts_gap_kd), pstdev(ts_gap_kd)
                dist_gain_mean, dist_gain_std = mean(dist_gain_kd), pstdev(dist_gain_kd)
            else:
                ts_gap_mean, ts_gap_std = mean(ts_gap_rkd), pstdev(ts_gap_rkd)
                dist_gain_mean, dist_gain_std = mean(dist_gain_rkd), pstdev(dist_gain_rkd)
            print(f"{name:8s}, {fmt(m_tr,s_tr)}, {fmt(m_te,s_te)}, {fmt(m_gp,s_gp)}, {fmt(ts_gap_mean, ts_gap_std)}, {fmt(dist_gain_mean, dist_gain_std)}")
        else:
            print(f"{name:8s}, {fmt(m_tr,s_tr)}, {fmt(m_te,s_te)}, {fmt(m_gp,s_gp)}, N/A, N/A")
            ts_gap_mean = ts_gap_std = dist_gain_mean = dist_gain_std = None
        summary_rows.append({
            "model": name,
            "train_mean": f"{m_tr:.4f}",
            "train_std": f"{s_tr:.4f}",
            "test_mean": f"{m_te:.4f}",
            "test_std": f"{s_te:.4f}",
            "acc_gap_mean": f"{m_gp:.4f}",
            "acc_gap_std": f"{s_gp:.4f}",
            "t_s_gap_mean": (f"{ts_gap_mean:.4f}" if ts_gap_mean is not None else ""),
            "t_s_gap_std": (f"{ts_gap_std:.4f}" if ts_gap_std is not None else ""),
            "dist_gain_mean": (f"{dist_gain_mean:.4f}" if dist_gain_mean is not None else ""),
            "dist_gain_std": (f"{dist_gain_std:.4f}" if dist_gain_std is not None else ""),
        })

    # ---------------- CSV EXPORT ----------------
    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    tag_chk = "_chk" if checkrun else ""
    seed_tag = "-".join(str(s) for s in seeds)
    base_name = f"table1_{dataset}_e{epochs}_seeds_{seed_tag}{tag_chk}"

    # Per-seed raw metrics
    per_seed_path = os.path.join(results_dir, base_name + "_per_seed.csv")
    with open(per_seed_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "seed", "model", "train_acc", "test_acc", "acc_gap", "teacher_student_gap", "distillation_gain"
        ])
        # reconstruct per-seed rows
        for idx, seed in enumerate(seeds):
            # Teacher & Scratch first
            t_tr, t_te, t_gap = all_metrics["Teacher"][idx]
            sc_tr, sc_te, sc_gap = all_metrics["Scratch"][idx]
            kd_tr, kd_te, kd_gap = all_metrics["KD"][idx]
            rkd_tr, rkd_te, rkd_gap = all_metrics["RKD"][idx]
            writer.writerow([seed, "Teacher", t_tr, t_te, t_gap, "", ""])  # teacher row
            writer.writerow([seed, "Scratch", sc_tr, sc_te, sc_gap, "", ""])  # scratch row
            writer.writerow([seed, "KD", kd_tr, kd_te, kd_gap, t_te - kd_te, kd_te - sc_te])
            writer.writerow([seed, "RKD", rkd_tr, rkd_te, rkd_gap, t_te - rkd_te, rkd_te - sc_te])

    # Summary metrics
    summary_path = os.path.join(results_dir, base_name + "_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [
            "model", "train_mean", "train_std", "test_mean", "test_std", "acc_gap_mean", "acc_gap_std",
            "t_s_gap_mean", "t_s_gap_std", "dist_gain_mean", "dist_gain_std"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
    print(f"\n[Saved] Per-seed metrics -> {per_seed_path}")
    print(f"[Saved] Summary metrics -> {summary_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Run multi-seed trainings and report Table 1 mean±std")
    p.add_argument("--dataset", default="mnist", choices=["mnist"]) 
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save-dir", default="models")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--checkrun", action="store_true")
    p.add_argument("--seeds", nargs='*', type=int, help="Explicit list of seeds, e.g., --seeds 0 1 2 3 4")
    p.add_argument("--num-seeds", type=int, default=5, help="If --seeds not provided, number of seeds starting at --seed-base")
    p.add_argument("--seed-base", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = args.seeds if args.seeds else list(range(args.seed_base, args.seed_base + args.num_seeds))
    sweep(
        seeds=seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset=args.dataset,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        checkrun=args.checkrun,
    )
