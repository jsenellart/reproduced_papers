"""Unified dataset generation for quantum-parity-style classification experiments.

Three generation strategies share the same call signature::

    generate_*(size, m, k, seed, ...) -> Dataset

and return a :class:`Dataset` with:

- ``X``  – shape ``(N, m-1)``, values uniformly sampled from ``[0, 2π]``
- ``y``  – shape ``(N,)``, binary labels in ``{0, 1}``
- ``soft_targets`` – continuous teacher signal (parity expectation,
  normalised pairwise score, or softmax probs), when available

Generators
----------
quantum
    Merlin sandwich circuit: label = parity of photon distribution.
    ``k = number of injected photons``. Requires perceval + merlin.

analytical
    Pairwise-phase interference:
    ``y = sign( sum_{i<j} sin(k*(x_i - x_j)) )``.
    For ``m=2`` (one feature): ``y = sign(sin(k*x_0))``.
    ``k`` controls the spatial frequency of the decision boundary.

mlp
    Random teacher MLP with Fourier-feature preprocessing:
    ``phi(x) = [sin(j*x_i), cos(j*x_i)  for j=1..fourier_order, i=0..d-1]``
    ``y = argmax( teacher(phi(x)) )``.
    ``k`` = number of hidden layers (each of width ``2*(m-1)``).

All three accept ``min_margin`` (Havlíček-style separation gap) and
``balanced`` (force exact 50/50 split) and resample iteratively to reach
``size`` post-filter samples. See :mod:`data._resample` for the algorithm.
"""

from __future__ import annotations

from typing import Callable

from .base import Dataset
from .photonic_quantum import generate_photonic_quantum
from .qubit_quantum import generate_qubit_quantum
from .analytical import generate_analytical
from .mlp import generate_mlp_teacher


GeneratorFn = Callable[..., Dataset]


GENERATORS: dict[str, GeneratorFn] = {
    "photonic_quantum": generate_photonic_quantum,
    "qubit_quantum":    generate_qubit_quantum,
    "analytical":       generate_analytical,
    "mlp":              generate_mlp_teacher,
}


__all__ = [
    "Dataset",
    "GENERATORS",
    "GeneratorFn",
    "generate_photonic_quantum",
    "generate_qubit_quantum",
    "generate_analytical",
    "generate_mlp_teacher",
]
