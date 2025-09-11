#!/usr/bin/env python3
"""
Compute Table 1-style metrics on MNIST for Teacher and Students:
- Train Acc, Test Acc
- Acc. Gap = Train - Test
- T&S Gap = Teacher(Test) - Student(Test)
- Distillation Gain = Student(Test) - Scratch(Test)

Run from QRKD/ directory. Examples:
  python scripts/evaluate_table1.py \
    --teacher models/mnist_teacher_seed0_e10.pt \
    --scratch models/mnist_student-scratch_seed0_e10.pt \
    --kd      models/mnist_student-kd_seed0_e10.pt \
    --rkd     models/mnist_student-rkd_seed0_e10.pt \
    --qrkd    models/mnist_student-qrkd_seed0_e10.pt

Optionally add --data-dir if datasets are in a custom path.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import torch

import sys, os
sys.path.append(os.path.abspath('.'))  # ensure top-level QRKD imports resolve when run from QRKD/
from lib.datasets import DataConfig, mnist_loaders
from lib.models import TeacherCNN, StudentCNN
from lib.eval import evaluate_accuracy


@dataclass
class Paths:
    teacher: str
    scratch: Optional[str] = None
    kd: Optional[str] = None
    rkd: Optional[str] = None
    qrkd: Optional[str] = None


def load_teacher(path: str, device: torch.device) -> torch.nn.Module:
    m = TeacherCNN().to(device)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


def load_student(path: str, device: torch.device) -> torch.nn.Module:
    m = StudentCNN().to(device)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd)
    m.eval()
    return m


def compute_metrics(model, train_loader, test_loader, device: torch.device):
    acc_train = evaluate_accuracy(model, train_loader, device)
    acc_test = evaluate_accuracy(model, test_loader, device)
    return {
        "train": acc_train,
        "test": acc_test,
        "gap": acc_train - acc_test,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Table 1-style metrics on MNIST")
    p.add_argument("--teacher", required=True, help="Path to teacher checkpoint (.pt)")
    p.add_argument("--scratch", help="Path to student scratch checkpoint")
    p.add_argument("--kd", help="Path to student kd checkpoint")
    p.add_argument("--rkd", help="Path to student rkd checkpoint")
    p.add_argument("--qrkd", help="Path to student qrkd checkpoint")
    p.add_argument("--data-dir", default=None, help="Custom MNIST root directory")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = (
        DataConfig(batch_size=args.batch_size)
        if args.data_dir is None
        else DataConfig(batch_size=args.batch_size, root=args.data_dir)
    )
    train_loader, test_loader = mnist_loaders(dcfg)

    # Teacher
    teacher = load_teacher(args.teacher, device)
    m_teacher = compute_metrics(teacher, train_loader, test_loader, device)

    # Students if provided
    results = {
        "Teacher": m_teacher,
    }

    scratch_acc = None

    if args.scratch:
        s = load_student(args.scratch, device)
        m = compute_metrics(s, train_loader, test_loader, device)
        results["F. Scratch"] = m
        scratch_acc = m["test"]

    if args.kd:
        s = load_student(args.kd, device)
        m = compute_metrics(s, train_loader, test_loader, device)
        results["KD"] = m

    if args.rkd:
        s = load_student(args.rkd, device)
        m = compute_metrics(s, train_loader, test_loader, device)
        results["RKD"] = m

    if args.qrkd:
        s = load_student(args.qrkd, device)
        m = compute_metrics(s, train_loader, test_loader, device)
        results["QRKD (classical)"] = m

    # Pretty print + derived metrics
    print("\nTable 1-style metrics (MNIST)")
    print("Model, Train Acc (%), Test Acc (%), Acc. Gap, T&S Gap, Dist. Gain")

    teacher_test = m_teacher["test"]
    if scratch_acc is None and args.scratch:
        # filled above
        pass

    def ts_gap(student_test: float) -> float:
        return teacher_test - student_test

    def dist_gain(student_test: float) -> Optional[float]:
        return None if scratch_acc is None else (student_test - scratch_acc)

    for name in ["Teacher", "F. Scratch", "KD", "RKD", "QRKD (classical)"]:
        if name not in results:
            continue
        r = results[name]
        train, test, gap = r["train"], r["test"], r["gap"]
        tsg = 0.0 if name == "Teacher" else ts_gap(test)
        dg = "-" if name == "Teacher" or scratch_acc is None else f"{dist_gain(test):.2f}"
        print(f"{name:16s}, {train:6.2f}, {test:6.2f}, {gap:6.2f}, {tsg:6.2f}, {dg}")


if __name__ == "__main__":
    main()
