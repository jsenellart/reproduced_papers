"""Student-model trainers used by ``work/train.py``.

Two learners share the same ``find_min_*`` interface (search a model-capacity
axis until a target accuracy is reached, on a dataset produced by one of the
generators in :mod:`data`):

- :mod:`learner.mlp`     — Fourier-feature MLP student.
- :mod:`learner.quantum` — variational photonic sandwich student
  (`W_train → encode(x) → W_train`) with parity readout, optional bias,
  optional Havlíček-style sigmoid loss, and SPSA among the optimisers.
"""

from .mlp import find_min_hidden_size, HiddenLayerSpec, train_and_eval as train_and_eval_mlp
from .photonic_quantum import (
    find_min_depth,
    train_and_eval_quantum,
    validate_teacher_params,
)
from .qubit_quantum import (
    find_min_qubit_depth,
    train_and_eval_qubit_quantum,
    validate_teacher_params_qubit,
    QubitClassifier,
)

__all__ = [
    "find_min_hidden_size",
    "HiddenLayerSpec",
    "train_and_eval_mlp",
    "find_min_depth",
    "train_and_eval_quantum",
    "validate_teacher_params",
    "find_min_qubit_depth",
    "train_and_eval_qubit_quantum",
    "validate_teacher_params_qubit",
    "QubitClassifier",
]
