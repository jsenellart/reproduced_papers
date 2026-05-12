"""Analytical pairwise-phase interference dataset.

Score: ``s(x) = sum_{i<j} sin(k * (x_i - x_j))``
For ``m=2`` (single feature ``d=1``): ``s(x) = sin(k * x_0)``.

``k`` controls the spatial frequency of the decision boundary; higher ``k``
creates finer-grained alternating regions, analogous to higher-harmonic
quantum interference. The function is naturally balanced (~50 % per class).
"""

from __future__ import annotations

import torch

from .base import Dataset
from ._resample import filter_resample


def generate_analytical(
    size: int,
    m: int,
    k: int,
    seed: int,
    min_margin: float = 0.0,
    balanced: bool = False,
    bail_threshold: float = 0.30,
    max_iter: int = 10,
    dtype: torch.dtype = torch.float32,
    **_,
) -> Dataset:
    """Pairwise-phase interference labels.

    Parameters
    ----------
    size : int
        Final number of examples (post-filter, post-balance).
    m, k : int
        Feature dimension is ``m - 1``; ``k`` is the spatial-frequency
        multiplier inside ``sin(k · ...)``.
    seed : int
        RNG seed for the dataset draws.
    min_margin : float
        Drop samples with ``|score| / n_pairs < min_margin`` (normalised to
        ``[0, 1]`` to match the quantum-generator scale).
    balanced : bool
        If True, return ``size // 2`` rows per class.
    """
    d = m - 1
    rng = torch.Generator().manual_seed(seed)

    # Pair indices are deterministic in m and shared across draws
    if d == 1:
        rows = cols = None
        n_pairs = 1
    else:
        rows, cols = torch.triu_indices(d, d, offset=1)
        n_pairs = int(rows.shape[0])

    def _draw(n: int):
        Xb = 2 * torch.pi * torch.rand(n, d, generator=rng, dtype=dtype)
        if d == 1:
            scores = torch.sin(k * Xb[:, 0])
        else:
            scores = torch.sin(k * (Xb[:, rows] - Xb[:, cols])).sum(dim=-1)
        yb = (scores >= 0).long()
        # Normalised confidence in [0, 1] — same scale as the quantum case
        confidence = scores.abs() / n_pairs
        return Xb, yb, scores, confidence

    (X, y, scores), info = filter_resample(
        target_size=size,
        balanced=balanced,
        min_margin=min_margin,
        draw_fn=_draw,
        perm_seed=seed + 7,
        low_survival_threshold=bail_threshold,
        max_iter=max_iter,
    )

    if info["n_iters"] > 0:
        print(
            f"  Drew {info['total_raw_drawn']} raw samples to reach {len(X)} "
            f"after filter+balance "
            f"(margin_survival={info['margin_survival']:.1%}, "
            f"balance_yield_initial={info['balance_yield']:.1%}, "
            f"iters={info['n_iters']})."
        )

    return Dataset(
        X=X,
        y=y,
        soft_targets=(scores / n_pairs).unsqueeze(-1),
        metadata={
            "generator": "analytical",
            "m": m,
            "k": k,
            "seed": seed,
            "min_margin": min_margin,
            "balanced": balanced,
            "formula": "sign(sum_{i<j} sin(k*(x_i - x_j)))",
            "num_pairs": n_pairs,
            "class_1_fraction": (
                round(y.float().mean().item(), 4) if len(y) > 0 else float("nan")
            ),
            "total_raw_drawn": info["total_raw_drawn"],
        },
    )
