"""Common Dataset dataclass returned by every generator in this package."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Dataset:
    """A binary classification dataset returned by the generators in :mod:`data`.

    Attributes
    ----------
    X : torch.Tensor
        Features, shape ``(N, m-1)``, values in ``[0, 2π]``.
    y : torch.Tensor
        Binary labels in ``{0, 1}``, shape ``(N,)``.
    metadata : dict
        Generator-specific bookkeeping. Always contains at least
        ``generator``, ``m``, ``k``, ``seed``, ``balanced``, ``min_margin``,
        ``total_raw_drawn``, ``class_1_fraction``.
    soft_targets : torch.Tensor | None
        Continuous teacher signal, when available:

        - quantum    : ``(N, 1)`` parity expectation in ``[-1, +1]``
        - analytical : ``(N, 1)`` normalised score in ``[-1, +1]``
        - mlp        : ``(N, 2)`` softmax probabilities

        ``None`` for raw label-only generators.
    """

    X: torch.Tensor
    y: torch.Tensor
    metadata: dict
    soft_targets: torch.Tensor | None = None
