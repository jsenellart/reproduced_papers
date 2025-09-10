"""Datasets and loaders for MNIST (classical baseline)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


@dataclass
class DataConfig:
    batch_size: int = 64
    num_workers: int = 2
    root: str = "./QRKD/data"


def mnist_loaders(cfg: DataConfig) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train = datasets.MNIST(cfg.root, train=True, download=True, transform=tfm)
    test = datasets.MNIST(cfg.root, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, test_loader
