"""Datasets and loaders for MNIST (classical baseline)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 2
    # Default to <repo_root>/QRKD/data regardless of current working directory
    root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def mnist_loaders(cfg: DataConfig) -> tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_root = os.path.abspath(os.path.expanduser(cfg.root))
    train = datasets.MNIST(data_root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(data_root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, test_loader
