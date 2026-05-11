"""Merlin photonic sandwich dataset generator.

Teacher circuit::

    W1 (Haar) → phase encode(x) → W2 (Haar) → photon-number measurement
    label = parity of photon counts on `parity_modes`

This module provides:

- :func:`generate_photonic_quantum`  : the public entry point that returns a
  :class:`data.base.Dataset` and is registered in
  :data:`data.GENERATORS`.
- helper builders (:func:`build_sandwich_experiment`,
  :func:`make_quantum_layer`) and the parity-from-Fock-key utility
  :func:`parity_of_key`, used by the training code in
  :mod:`train_quantum`.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import perceval as pcvl
import merlin as ML

from .base import Dataset
from ._resample import filter_resample


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def build_sandwich_experiment(
    m: int,
    phase_prefix: str = "x",
) -> tuple[pcvl.Experiment, "np.ndarray", "np.ndarray"]:
    """Build the W1 → P(x) → W2 circuit.

    P(x) = ``diag(exp(i x_0), ..., exp(i x_{m-2}), 1)`` — the final mode
    receives no phase, removing the global phase redundancy.

    Returns
    -------
    experiment : pcvl.Experiment
    W1 : np.ndarray  shape (m, m) complex  – first Haar random unitary
    W2 : np.ndarray  shape (m, m) complex  – second Haar random unitary
    """
    circuit = pcvl.Circuit(m, name="haar_phase_haar")

    # Fixed Haar block W1
    _W1 = pcvl.Matrix.random_unitary(m)
    circuit.add(0, pcvl.Unitary(_W1), merge=True)

    # Input-dependent phase column
    for i in range(m - 1):
        circuit.add(i, pcvl.PS(pcvl.P(f"{phase_prefix}{i}")))

    # Fixed Haar block W2
    _W2 = pcvl.Matrix.random_unitary(m)
    circuit.add(0, pcvl.Unitary(_W2), merge=True)

    return pcvl.Experiment(circuit), np.asarray(_W1), np.asarray(_W2)


def default_input_state(m: int, k: int) -> list[int]:
    """Inject one photon in each of the first k modes."""
    state = [0] * m
    for i in range(k):
        state[i] = 1
    return state


def parity_of_key(key, parity_modes: Iterable[int]) -> int:
    """Return +1 for even photon number on ``parity_modes``, -1 otherwise."""
    n = sum(int(key[i]) for i in parity_modes)
    return +1 if n % 2 == 0 else -1


def make_quantum_layer(
    m: int,
    k: int,
    seed: int = 1234,
    input_state: list[int] | None = None,
    phase_prefix: str = "x",
    return_object: bool = False,
) -> tuple[ML.QuantumLayer, dict]:
    """Build the merlin QuantumLayer + a metadata dict (also stores W1, W2)."""
    experiment, W1, W2 = build_sandwich_experiment(
        m=m,
        phase_prefix=phase_prefix,
    )

    if input_state is None:
        input_state = [1] * k + [0] * (m - k)

    layer = ML.QuantumLayer(
        input_size=m - 1,
        experiment=experiment,
        input_state=input_state,
        input_parameters=[phase_prefix],
        measurement_strategy=ML.MeasurementStrategy.probs(ML.ComputationSpace.FOCK),
        return_object=return_object,
    )

    return layer, {
        "m": m,
        "k": k,
        "seed": seed,
        "input_state": input_state,
        "phase_prefix": phase_prefix,
        "teacher_W1": W1,   # (m, m) complex numpy array — exact teacher unitary
        "teacher_W2": W2,   # (m, m) complex numpy array — exact teacher unitary
    }


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_photonic_quantum(
    size: int,
    m: int,
    k: int,
    seed: int = 1234,
    nsample: int = 0,
    parity_modes: Iterable[int] | None = None,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_iter: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
    **_,
) -> Dataset:
    """Quantum parity dataset from a Merlin sandwich circuit.

    Parameters
    ----------
    size : int
        Final number of examples (post-filter, post-balance).
    m : int
        Number of optical modes.
    k : int
        Number of injected photons (1 ≤ k ≤ m).
    seed : int
        RNG seed for the Haar matrices and dataset draws.
    nsample : int
        0 ⇒ exact probabilities returned by the simulator (default).
        >0 ⇒ sample ``nsample`` shots; the layer returns empirical probabilities.
    parity_modes : iterable of int, optional
        Modes used for parity. Default: first ``ceil(m/2)`` modes. Do not use
        all modes — the total parity is fixed by ``k``.
    balanced : bool
        If True, return ``size // 2`` rows per class.
    min_margin : float
        Drop samples with ``|parity_expectation| < min_margin``. Mimics the
        Havlíček et al. (2019) separation gap.
    bail_threshold : float
        If the post-balance yield from the first batch is below this fraction,
        return a short dataset rather than iteratively resampling. Default
        0.30. Set to 0 to disable the bail-out and always iterate to ``size``
        (potentially expensive).
    max_iter : int
        Maximum resample iterations after the first batch. Default 10.
    device, dtype
        Tensor placement / precision.

    Returns
    -------
    Dataset
        ``X`` ``(N, m-1)`` features in [0, 2π], ``y`` ``(N,)`` binary labels,
        ``soft_targets`` ``(N, 1)`` continuous parity expectation in [-1, +1],
        ``metadata`` includes the teacher unitaries W1/W2 (used for sanity
        checking by ``train_quantum.validate_teacher_params``).
    """
    if not (1 <= k <= m):
        raise ValueError("Require 1 <= k <= m.")
    if nsample < 0:
        raise ValueError("nsample must be >= 0.")

    if parity_modes is None:
        parity_modes = tuple(range((m + 1) // 2))
    else:
        parity_modes = tuple(parity_modes)

    torch.manual_seed(seed)
    pcvl.random_seed(seed)

    layer, metadata = make_quantum_layer(m=m, k=k, seed=seed)
    layer = layer.to(device)
    output_keys = list(layer.output_keys)
    parity_vec = torch.tensor(
        [parity_of_key(key, parity_modes) for key in output_keys],
        dtype=dtype,
        device=device,
    )

    def _draw(n: int):
        Xb = 2.0 * np.pi * torch.rand(n, m - 1, dtype=dtype, device=device)
        probs = layer.forward(Xb, shots=nsample if nsample > 0 else None)
        parity = probs @ parity_vec
        yb = (parity >= 0).to(torch.long)
        confidence = parity.abs()
        return Xb, yb, parity, confidence

    (X, y, parity_expectation), info = filter_resample(
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

    metadata.update(
        {
            "generator": "quantum",
            "nsample": nsample,
            "parity_modes": tuple(parity_modes),
            "balanced": balanced,
            "min_margin": min_margin,
            "total_raw_drawn": info["total_raw_drawn"],
            "class_1_fraction": (
                round(y.float().mean().item(), 4) if len(y) > 0 else float("nan")
            ),
        }
    )

    return Dataset(
        X=X.cpu(),
        y=y.cpu(),
        soft_targets=parity_expectation.cpu().unsqueeze(-1),
        metadata=metadata,
    )


if __name__ == "__main__":
    ds = generate_photonic_quantum(size=10000, m=6, k=3, nsample=0, seed=69)
    print("X:", ds.X.shape)
    print("labels:", ds.y[:10])
    print("0:", (ds.y == 0).sum().item(), " 1:", (ds.y == 1).sum().item())
