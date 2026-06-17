"""Qubit-based quantum dataset (Havlíček et al. 2019, *Nature* 567, 209).

Implements the IQP-style feature map exactly as in the reference paper::

    U_Φ(x) = exp(i [ Σ_i x_i Z_i + Σ_{i<j} (π − x_i)(π − x_j) Z_i Z_j ])
    |ψ_Φ(x)⟩ = U_Φ(x) · H^⊗n · U_Φ(x) · H^⊗n · |0^n⟩

The teacher then applies a random variational unitary ``W_teacher(θ)`` from
the same ansatz family used by :mod:`learner.qubit_quantum` (alternating
single-qubit Y/Z rotation layers and a fixed CZ-chain entangler), and the
label is the sign of the parity expectation::

    label = sign( ⟨ψ| W_teacher^† Z^⊗n W_teacher |ψ⟩ )

This matches the "data generation by an unknown classifier" recipe from
Havlíček et al. — the ``--min-margin`` flag implements their 0.3 separation
gap by dropping samples whose absolute parity expectation is below the gap.

Number of qubits ``n = m − 1``, matching the feature dimension shared
across all generators.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import Dataset
from ._resample import filter_resample


# ---------------------------------------------------------------------------
# Tiny pure-torch qubit simulator
# ---------------------------------------------------------------------------

def _walsh_hadamard_matrix(n_qubits: int, dtype: torch.dtype) -> torch.Tensor:
    """Build H^⊗n as a (2^n, 2^n) real-valued matrix (cast to ``dtype``)."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) / math.sqrt(2.0)
    H = H1
    for _ in range(n_qubits - 1):
        H = torch.kron(H, H1)
    return H.to(dtype)


def _bitstring_signs(n_qubits: int, dtype: torch.dtype) -> torch.Tensor:
    """Return a (2^n, n) tensor of ±1 — the eigenvalue of ``Z_i`` on each basis state.

    ``signs[z, i] = +1`` if bit ``i`` of ``z`` is 0, ``-1`` otherwise.
    Bit 0 is the least-significant bit.
    """
    z = torch.arange(2 ** n_qubits)
    bits = (z.unsqueeze(-1) >> torch.arange(n_qubits)) & 1
    return (1 - 2 * bits).to(dtype)


def _iqp_diag(
    x: torch.Tensor,
    signs: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute the IQP feature-map diagonal ``exp(i φ(z, x))`` per sample.

    Parameters
    ----------
    x : (B, n) tensor of feature angles in [0, 2π].
    signs : (2^n, n) precomputed Z-eigenvalues per basis state.
    pair_mask : (n, n) upper-triangular bool mask (i < j).

    Returns
    -------
    (B, 2^n) complex tensor — the diagonal entries of ``U_Φ(x)``.
    """
    # Single-qubit term: φ_single[b, z] = Σ_i signs[z, i] · x[b, i]
    phi_single = x @ signs.T  # (B, 2^n)

    # Pair term: φ_pair[b, z] = Σ_{i<j} signs[z, i] · signs[z, j] · (π−x_i)(π−x_j)
    diff = math.pi - x  # (B, n)
    diff_outer = diff.unsqueeze(-1) * diff.unsqueeze(-2)  # (B, n, n)
    diff_outer = diff_outer * pair_mask
    sign_outer = signs.unsqueeze(-1) * signs.unsqueeze(-2)  # (2^n, n, n)
    sign_outer = sign_outer * pair_mask
    # Contract over the last two dims; broadcast over batch and basis state
    phi_pair = (
        sign_outer.unsqueeze(0) * diff_outer.unsqueeze(1)
    ).sum(dim=(-2, -1))  # (B, 2^n)

    phi = phi_single + phi_pair
    return torch.exp(1j * phi.to(torch.complex64))


def feature_map_state(
    x: torch.Tensor,
    n_qubits: int,
    H_matrix: torch.Tensor,
    signs: torch.Tensor,
    pair_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply ``U_Φ(x) H^⊗n U_Φ(x) H^⊗n |0^n⟩`` for each row of ``x``.

    Returns a complex tensor of shape ``(B, 2^n)``.
    """
    H_c = H_matrix.to(torch.complex64)

    # |0^n⟩ has amplitude 1 on basis state 0, so H^⊗n|0^n⟩ = (1/√2^n)·1ᵀ.
    # That is exactly the first column of H^⊗n.
    plus = H_c[:, 0]                              # (2^n,)
    state = plus.unsqueeze(0).expand(x.shape[0], -1).clone()  # (B, 2^n)

    state = state * _iqp_diag(x, signs, pair_mask)        # 1st U_Φ(x)
    state = state @ H_c.T                                  # 2nd H^⊗n
    state = state * _iqp_diag(x, signs, pair_mask)        # 2nd U_Φ(x)
    return state


# ---------------------------------------------------------------------------
# Variational ansatz (matches Havlíček paper)
# ---------------------------------------------------------------------------

def _apply_single_qubit_gate(
    state: torch.Tensor,
    gate: torch.Tensor,
    qubit_idx: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply a single-qubit gate to the given qubit of every state in the batch.

    state : (B, 2^n) complex
    gate  : (2, 2) complex
    """
    B = state.shape[0]
    # Reshape (B, 2^n) → (B, 2^{n−i−1}, 2, 2^i) so qubit i is the middle dim.
    high = 2 ** (n_qubits - qubit_idx - 1)
    low = 2 ** qubit_idx
    s = state.reshape(B, high, 2, low)
    # gate[out, in] · s[B, hi, in, lo] → (B, hi, out, lo)
    out = torch.einsum("oi,bhik->bhok", gate, s)
    return out.reshape(B, -1)


def _apply_cz(
    state: torch.Tensor,
    qubit_i: int,
    qubit_j: int,
    n_qubits: int,
) -> torch.Tensor:
    """Apply CZ_{i, j} (diagonal) to every state in the batch."""
    z = torch.arange(2 ** n_qubits, device=state.device)
    both_one = ((z >> qubit_i) & 1) & ((z >> qubit_j) & 1)
    sign = (1 - 2 * both_one.to(torch.complex64))  # −1 where both bits 1
    return state * sign


def _ry_gate(theta: torch.Tensor) -> torch.Tensor:
    """RY(θ) as a (2, 2) complex tensor (autograd-friendly)."""
    half = theta / 2
    c = torch.cos(half).to(torch.complex64)
    s = torch.sin(half).to(torch.complex64)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _rz_gate(theta: torch.Tensor) -> torch.Tensor:
    """RZ(θ) as a (2, 2) complex tensor (autograd-friendly)."""
    half = theta / 2
    e_minus = torch.exp(-1j * half.to(torch.complex64))
    e_plus = torch.exp(1j * half.to(torch.complex64))
    zero = torch.zeros_like(e_minus)
    return torch.stack([torch.stack([e_minus, zero]), torch.stack([zero, e_plus])])


def variational_apply(
    state: torch.Tensor,
    theta: torch.Tensor,
    n_qubits: int,
) -> torch.Tensor:
    """Apply ``W(θ) = U_loc(θ_L) [U_ent U_loc]_{l=L−1..1} U_ent U_loc(θ_0)``.

    Each ``U_loc(θ_l)`` applies ``RY(θ_l_i_y) · RZ(θ_l_i_z)`` to qubit i.
    ``U_ent`` is a fixed CZ chain ``CZ_{0,1} CZ_{1,2} … CZ_{n−2, n−1}``.

    Parameters
    ----------
    state : (B, 2^n) complex
    theta : ((L+1) * n_qubits * 2,) flat parameter vector

    Returns
    -------
    (B, 2^n) complex
    """
    n = n_qubits
    n_per_layer = n * 2
    n_layers = theta.numel() // n_per_layer  # this is L+1
    theta = theta.reshape(n_layers, n, 2)

    s = state
    for layer in range(n_layers):
        # Single-qubit rotations: RY then RZ on each qubit
        for q in range(n):
            s = _apply_single_qubit_gate(s, _ry_gate(theta[layer, q, 0]), q, n)
            s = _apply_single_qubit_gate(s, _rz_gate(theta[layer, q, 1]), q, n)
        # Entangler (skip after the last rotation layer to mirror Havlíček)
        if layer < n_layers - 1 and n >= 2:
            for q in range(n - 1):
                s = _apply_cz(s, q, q + 1, n)
    return s


def parity_expectation(state: torch.Tensor, n_qubits: int) -> torch.Tensor:
    """Compute ``⟨ψ| Z^⊗n |ψ⟩`` for each batched state vector.

    Returns a real tensor of shape ``(B,)``.
    """
    z = torch.arange(2 ** n_qubits)
    parity_signs = (1 - 2 * (bin_popcount(z) % 2)).to(torch.float32)  # (2^n,) ±1
    probs = (state.conj() * state).real  # (B, 2^n)
    return probs @ parity_signs


def bin_popcount(t: torch.Tensor) -> torch.Tensor:
    """Element-wise number of set bits (Hamming weight)."""
    out = torch.zeros_like(t)
    while t.any():
        out += (t & 1)
        t = t >> 1
    return out


def n_variational_params(n_qubits: int, n_layers: int) -> int:
    """Number of free parameters in ``variational_apply`` for L layers (i.e. L+1 rotation layers)."""
    return (n_layers + 1) * n_qubits * 2


# ---------------------------------------------------------------------------
# Public generator
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_qubit_quantum(
    size: int,
    m: int,
    k: int = 4,
    seed: int = 1234,
    nsample: int = 0,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_iter: int = 10,
    dtype: torch.dtype = torch.float32,
    **_,
) -> Dataset:
    """Qubit IQP-feature-map dataset (Havlíček et al. 2019).

    Parameters
    ----------
    size : int
        Final number of examples (post-filter, post-balance).
    m : int
        Number of qubits is ``n = m − 1`` (matches the feature dim convention
        of the other generators).
    k : int
        Variational depth ``L`` of the random teacher unitary
        ``W_teacher(θ)``.  Default 4 — the deepest setting in the paper.
    seed : int
        RNG seed for the random teacher and for the dataset draws.
    nsample : int
        0 (default) ⇒ exact parity expectation.  >0 ⇒ simulate finite-shot
        measurement noise: the estimated mean of nsample ±1 Pauli measurements
        has std = sqrt((1 − E²) / nsample) by the CLT, so Gaussian noise with
        that std is added to the exact expectation and the result is clamped to
        [−1, +1].  Labels and margin filter are applied to the noisy value.
    balanced : bool
        If True, return ``size // 2`` rows per class.
    min_margin : float
        Drop samples with ``|parity_expectation| < min_margin``.  Default 0
        (keep all); use ``0.3`` for the paper's separation gap.

    Returns
    -------
    Dataset
        ``X`` ``(N, n)`` with ``n = m − 1`` features in [0, 2π],
        ``y`` ``(N,)`` binary labels,
        ``soft_targets`` ``(N, 1)`` continuous parity expectation in [−1, +1],
        ``metadata`` includes the random teacher angles for sanity checking.
    """
    if m < 2:
        raise ValueError("Require m >= 2 (so the qubit count n = m − 1 is at least 1).")
    if k < 0:
        raise ValueError("k (variational depth) must be >= 0.")

    n_qubits = m - 1
    torch.manual_seed(seed)

    # Precompute fixed structural tensors
    H_matrix = _walsh_hadamard_matrix(n_qubits, dtype=dtype)
    signs = _bitstring_signs(n_qubits, dtype=dtype)
    pair_mask = torch.triu(torch.ones(n_qubits, n_qubits), diagonal=1).bool()

    # Random teacher variational angles
    n_params = n_variational_params(n_qubits, n_layers=k)
    theta_teacher = 2 * torch.pi * torch.rand(n_params, dtype=dtype)

    data_rng  = torch.Generator().manual_seed(seed + 1)
    noise_rng = torch.Generator().manual_seed(seed + 13)

    def _draw(n: int):
        Xb = 2 * torch.pi * torch.rand(n, n_qubits, generator=data_rng, dtype=dtype)
        psi = feature_map_state(Xb, n_qubits, H_matrix, signs, pair_mask)
        psi = variational_apply(psi, theta_teacher, n_qubits)
        parity = parity_expectation(psi, n_qubits)  # (n,) real, exact
        if nsample > 0:
            # CLT: std of sample mean = sqrt((1 − E²) / nsample)
            noise_std = ((1.0 - parity ** 2) / nsample).clamp(min=0.0).sqrt()
            parity = (parity + torch.randn(n, generator=noise_rng, dtype=dtype) * noise_std).clamp(-1.0, 1.0)
        yb = (parity >= 0).long()
        confidence = parity.abs()
        return Xb, yb, parity, confidence

    (X, y, parity_exp), info = filter_resample(
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
        soft_targets=parity_exp.unsqueeze(-1),
        metadata={
            "generator": "qubit_quantum",
            "m": m,
            "n_qubits": n_qubits,
            "k": k,
            "seed": seed,
            "nsample": nsample,
            "balanced": balanced,
            "min_margin": min_margin,
            "class_1_fraction": (
                round(y.float().mean().item(), 4) if len(y) > 0 else float("nan")
            ),
            "total_raw_drawn": info["total_raw_drawn"],
            # Teacher parameters — for sanity-checking via validate_teacher_params_qubit
            "teacher_theta": theta_teacher.cpu().clone(),
            "teacher_n_layers": k,
        },
    )
