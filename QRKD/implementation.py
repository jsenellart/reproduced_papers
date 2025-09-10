"""Implementation entrypoint to reproduce MNIST classical results from QRKD paper.

Two usage modes:
- CLI (recommended for experiments): Train ONE model/variant at a time with
    configurable dataset, epochs, and seed. Models are saved under ``models/``
    with filenames including dataset, variant, seed and epochs. For student
    training, you can pass a path to a pretrained teacher.
- Programmatic ``run()``: runs a quick end-to-end sweep (teacher + 4 students)
    to mirror Table 1 (classical only), mainly for smoke testing/demo.

It reports a compact summary similar to Table 1 (test accuracy only here).
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import torch
from lib.datasets import DataConfig, mnist_loaders
from lib.losses import QRKDWeights
from lib.models import StudentCNN, TeacherCNN
from lib.train import TrainConfig, train_student, train_teacher
from lib.utils import set_seed


@dataclass
class ExpConfig:
    seed: int = 0
    epochs: int = 10
    lr: float = 1e-3
    batch_size: int = 64


# -------------------- Dataset dispatch --------------------
def get_loaders(
    dataset: str, batch_size: int, data_root: str | None = None
) -> tuple[object, object]:
    dataset = dataset.lower()
    if dataset == "mnist":
        dcfg = (
            DataConfig(batch_size=batch_size)
            if data_root is None
            else DataConfig(batch_size=batch_size, root=data_root)
        )
        return mnist_loaders(dcfg)
    raise ValueError(f"Unsupported dataset: {dataset}")


def _train_teacher_wrapper(
    train_loader, epochs: int = 5, lr: float = 1e-3, verbose: bool = True
) -> TeacherCNN:
    model = TeacherCNN()
    cfg = TrainConfig(epochs=epochs, lr=lr, verbose=verbose)
    return train_teacher(model, train_loader, cfg)  # returns eval() model


def save_model(model: torch.nn.Module, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(model.state_dict(), path)
    return path


def load_teacher(
    path: str, map_location: str | torch.device | None = None
) -> TeacherCNN:
    device = map_location or (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = TeacherCNN().to(device)
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def run() -> dict[str, float]:
    cfg = ExpConfig()
    # Demo-only defaults (programmatic run): keep small for a quick sweep
    env_epochs = cfg.epochs
    env_teacher_epochs = 5
    set_seed(cfg.seed)

    train_loader, test_loader = get_loaders("mnist", cfg.batch_size)

    # 1) Train teacher quickly (could be pre-trained in practice)
    teacher = _train_teacher_wrapper(train_loader, epochs=env_teacher_epochs, lr=cfg.lr)

    # 2) Student from scratch (task loss only)
    student_scratch = StudentCNN()
    res_scratch = train_student(
        student_scratch,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=env_epochs, lr=cfg.lr),
        QRKDWeights(kd=0.0, dr=0.0, ar=0.0),
    )

    # 3) KD only
    student_kd = StudentCNN()
    res_kd = train_student(
        student_kd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=env_epochs, lr=cfg.lr),
        QRKDWeights(kd=0.5, dr=0.0, ar=0.0),
    )

    # 4) RKD (distance + angle)
    student_rkd = StudentCNN()
    res_rkd = train_student(
        student_rkd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=env_epochs, lr=cfg.lr),
        QRKDWeights(kd=0.0, dr=0.1, ar=0.1),
    )

    # 5) QRKD classical (KD + RKD). Quantum loss will be added later.
    student_qrkd = StudentCNN()
    res_qrkd = train_student(
        student_qrkd,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(epochs=env_epochs, lr=cfg.lr),
        QRKDWeights(kd=0.5, dr=0.1, ar=0.1),
    )

    summary = {
        "F. Scratch": res_scratch["test_acc"],
        "KD": res_kd["test_acc"],
        "RKD": res_rkd["test_acc"],
        "QRKD (classical)": res_qrkd["test_acc"],
    }

    print("MNIST Test Accuracy (%)")
    for k, v in summary.items():
        print(f"{k:>16}: {v:5.2f}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="QRKD classical training CLI")
    parser.add_argument(
        "--task",
        choices=[
            "teacher",
            "student_scratch",
            "student_kd",
            "student_rkd",
            "student_qrkd",
        ],
        required=True,
        help="What to train",
    )
    parser.add_argument(
        "--dataset", choices=["mnist"], default="mnist", help="Dataset to use"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train (applies to the selected task)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument(
        "--teacher-path",
        type=str,
        default=None,
        help="Path to a pretrained teacher .pt file; if not provided and a student task is selected, a teacher will be trained on the fly",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Root directory to store/download dataset (default: QRKD/data)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Reduce training logs (default: verbose)"
    )
    parser.add_argument(
        "--checkrun",
        action="store_true",
        help="Limit to 50 batches per epoch for quick training checks",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    train_loader, test_loader = get_loaders(
        args.dataset, args.batch_size, args.data_dir
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single epochs option applies to the selected training task
    # nothing

    if args.task == "teacher":
        # Build TrainConfig for teacher
        tcfg = TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            verbose=not args.quiet,
            max_batches=50 if args.checkrun else None,
        )
        model = TeacherCNN()
        teacher = train_teacher(model, train_loader, tcfg)
        chk = "_chk" if args.checkrun else ""
        fname = f"{args.dataset}_teacher_seed{args.seed}_e{args.epochs}{chk}.pt"
        path = save_model(teacher, args.save_dir, fname)
        print(f"Saved teacher to {path}")
        return

    # For students, a teacher is mandatory
    if args.teacher_path:
        teacher = load_teacher(args.teacher_path)
    else:
        raise SystemExit(
            "--teacher-path is required for student tasks. Train a teacher first with --task teacher."
        )

    # Instantiate student
    student = StudentCNN().to(device)

    # Select weights by variant
    if args.task == "student_scratch":
        weights = QRKDWeights(kd=0.0, dr=0.0, ar=0.0)
        variant = "scratch"
    elif args.task == "student_kd":
        weights = QRKDWeights(kd=0.5, dr=0.0, ar=0.0)
        variant = "kd"
    elif args.task == "student_rkd":
        weights = QRKDWeights(kd=0.0, dr=0.1, ar=0.1)
        variant = "rkd"
    elif args.task == "student_qrkd":
        weights = QRKDWeights(kd=0.5, dr=0.1, ar=0.1)
        variant = "qrkd"
    else:
        raise ValueError(f"Unknown task {args.task}")

    results = train_student(
        student,
        teacher,
        train_loader,
        test_loader,
        TrainConfig(
            epochs=args.epochs,
            lr=args.lr,
            verbose=not args.quiet,
            max_batches=50 if args.checkrun else None,
        ),
        weights,
    )

    chk = "_chk" if args.checkrun else ""
    fname = f"{args.dataset}_student-{variant}_seed{args.seed}_e{args.epochs}{chk}.pt"
    path = save_model(student, args.save_dir, fname)
    print(f"Saved student to {path}")
    print(f"Test accuracy: {results['test_acc']:.2f}%")


if __name__ == "__main__":
    main()
