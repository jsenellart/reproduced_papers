"""Random-teacher MLP dataset with Fourier-feature preprocessing.

Architecture::

    phi(x) = [sin(j*x_i), cos(j*x_i)  for j=1..fourier_order, i=0..d-1]
    teacher(phi)   : k hidden tanh layers of width hidden_size, no biases
    y = teacher(phi).argmax(dim=-1)

The student MLP in :mod:`train_mlp` reuses the same Fourier preprocessing,
so it can in principle distil the teacher given enough capacity.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import Dataset
from ._resample import filter_resample


def _fourier_features(X: torch.Tensor, fourier_order: int) -> torch.Tensor:
    parts = []
    for j in range(1, fourier_order + 1):
        parts.append(torch.sin(j * X))
        parts.append(torch.cos(j * X))
    return torch.cat(parts, dim=-1)   # (N, 2 * fourier_order * d)


def generate_mlp_teacher(
    size: int,
    m: int,
    k: int,
    seed: int,
    fourier_order: int = 3,
    hidden_size: int | None = None,
    min_margin: float = 0.0,
    balanced: bool = True,
    bail_threshold: float = 0.30,
    max_iter: int = 10,
    dtype: torch.dtype = torch.float32,
    **_,
) -> Dataset:
    """Random-teacher MLP labels.

    Parameters
    ----------
    size : int
        Final number of examples (post-filter, post-balance).
    m, k : int
        Feature dimension is ``m - 1``; ``k`` is the number of teacher hidden
        layers.
    fourier_order : int
        Number of Fourier harmonics used in the input encoding.
    hidden_size : int, optional
        Width of each teacher hidden layer; defaults to ``max(2(m-1), 8)``.
    min_margin : float
        Drop samples with ``|p1 - p0| < min_margin`` (softmax confidence).
    balanced : bool
        If True (default), return ``size // 2`` rows per class.
    """
    d = m - 1
    feat_size = 2 * fourier_order * d
    width = hidden_size if hidden_size is not None else max(2 * (m - 1), 8)
    n_layers = max(k, 1)

    # Build teacher: tanh activations, no bias terms.
    # With zero bias and tanh, the network maps a zero-mean input (Fourier
    # features are zero-mean on [0, 2π]) to a zero-mean output by symmetry,
    # so predictions are approximately balanced (≈50% per class) regardless
    # of the random weights.
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    in_size = feat_size
    for _ in range(n_layers):
        lin = nn.Linear(in_size, width, bias=False)
        nn.init.xavier_uniform_(lin.weight, gain=nn.init.calculate_gain("tanh"))
        layers += [lin, nn.Tanh()]
        in_size = width
    out = nn.Linear(in_size, 2, bias=False)
    nn.init.xavier_uniform_(out.weight)
    layers.append(out)
    teacher = nn.Sequential(*layers)
    teacher.eval()

    data_rng = torch.Generator().manual_seed(seed + 1)

    def _draw(n: int):
        Xb = 2 * torch.pi * torch.rand(n, d, generator=data_rng, dtype=dtype)
        phi = _fourier_features(Xb, fourier_order=fourier_order)
        with torch.no_grad():
            logits = teacher(phi)                          # (n, 2)
        probs = nn.functional.softmax(logits, dim=-1)     # (n, 2) for KD
        yb = logits.argmax(dim=-1)                         # (n,) labels in {0, 1}
        confidence = (probs[:, 1] - probs[:, 0]).abs()    # (n,) ∈ [0, 1]
        return Xb, yb, probs, confidence

    (X, y, probs), info = filter_resample(
        target_size=size,
        balanced=balanced,
        min_margin=min_margin,
        draw_fn=_draw,
        perm_seed=seed + 2,
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
        soft_targets=probs,   # teacher softmax probabilities for KD loss
        metadata={
            "generator": "mlp",
            "m": m,
            "k": k,
            "seed": seed,
            "fourier_order": fourier_order,
            "hidden_size": width,
            "n_hidden_layers": n_layers,
            "class_1_fraction": (
                round(y.float().mean().item(), 4) if len(y) > 0 else float("nan")
            ),
            "min_margin": min_margin,
            "balanced": balanced,
            "total_raw_drawn": info["total_raw_drawn"],
            # Teacher parameters — use validate_teacher_params() to verify 100% accuracy
            "teacher_state_dict": {
                key: val.cpu().clone() for key, val in teacher.state_dict().items()
            },
            "teacher_fourier_order": fourier_order,
        },
    )
