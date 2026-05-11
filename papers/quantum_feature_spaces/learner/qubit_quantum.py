"""Qubit IQP feature-map classifier — direct re-implementation of
Havlíček et al., *Nature* 567, 209 (2019), arXiv:1804.11326.

Student architecture::

    |0^n⟩  →  H^⊗n  →  U_Φ(x)  →  H^⊗n  →  U_Φ(x)  →  W(θ)  →  Z^⊗n parity

where ``U_Φ`` is the IQP feature map with single + (π−x_i)(π−x_j) cross
terms (see :mod:`data.qubit_quantum`) and ``W(θ)`` is a layered ansatz
``U_loc(θ_L) [U_ent U_loc]_{l=L−1..1} U_ent U_loc(θ_0)`` with a CZ-chain
entangler.

The student matches the teacher's architecture (same feature map, same
ansatz family) so the only thing it has to learn is the parameter vector
``θ`` (and an optional bias ``b`` for the Havlíček sigmoid loss).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from data import GENERATORS
from data.qubit_quantum import (
    _walsh_hadamard_matrix,
    _bitstring_signs,
    feature_map_state,
    variational_apply,
    parity_expectation,
    n_variational_params,
)


# ---------------------------------------------------------------------------
# Teacher validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_teacher_params_qubit(
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    metadata: dict,
) -> float:
    """Rebuild the teacher and measure 0/1 accuracy on ``X_test``.

    Returns 1.0 when the dataset is well-formed (the dataset's labels are
    *defined* by the teacher, so this should always be 100 % modulo
    boundary samples filtered out by ``min_margin``).
    """
    n_qubits = metadata["n_qubits"]
    theta_teacher = metadata["teacher_theta"]
    H = _walsh_hadamard_matrix(n_qubits, dtype=X_test.dtype)
    signs = _bitstring_signs(n_qubits, dtype=X_test.dtype)
    pair_mask = torch.triu(torch.ones(n_qubits, n_qubits), diagonal=1).bool()

    psi = feature_map_state(X_test, n_qubits, H, signs, pair_mask)
    psi = variational_apply(psi, theta_teacher, n_qubits)
    parity = parity_expectation(psi, n_qubits)
    y_pred = (parity >= 0).long()
    return (y_pred == y_test.long()).float().mean().item()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QubitClassifier(nn.Module):
    """Variational qubit classifier (Havlíček ansatz).

    Parameters
    ----------
    n_qubits : int
        Number of qubits = feature dimension. Must equal ``m − 1`` for the
        dataset's ``m``.
    depth : int
        Variational depth ``L``. Total free parameters = ``(L+1) · n_qubits · 2``.
    learnable_bias : bool
        If True, adds a single trainable scalar ``b`` that is added to the
        parity expectation before the decision rule (used by the Havlíček
        sigmoid loss). If False, ``b`` is fixed to 0.
    """

    def __init__(
        self,
        n_qubits: int,
        depth: int = 4,
        learnable_bias: bool = True,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.depth = depth

        # Trainable variational angles: small init to start near identity-ish
        n_params = n_variational_params(n_qubits, n_layers=depth)
        self.theta = nn.Parameter(0.1 * torch.randn(n_params))
        if learnable_bias:
            self.bias = nn.Parameter(torch.zeros(()))
        else:
            self.register_buffer("bias", torch.zeros(()))

        # Precomputed structural tensors (registered as buffers)
        self.register_buffer(
            "H_matrix", _walsh_hadamard_matrix(n_qubits, dtype=torch.float32)
        )
        self.register_buffer(
            "signs", _bitstring_signs(n_qubits, dtype=torch.float32)
        )
        self.register_buffer(
            "pair_mask",
            torch.triu(torch.ones(n_qubits, n_qubits), diagonal=1).bool(),
        )

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parity_expectation(self, x: torch.Tensor) -> torch.Tensor:
        """Continuous parity expectation ⟨Z^⊗n⟩ ∈ [−1, +1], shape (B,)."""
        psi = feature_map_state(
            x, self.n_qubits, self.H_matrix, self.signs, self.pair_mask
        )
        psi = variational_apply(psi, self.theta, self.n_qubits)
        return parity_expectation(psi, self.n_qubits) + self.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Two-class logits compatible with :class:`nn.CrossEntropyLoss`."""
        logit = self.parity_expectation(x)
        return torch.stack([-logit, logit], dim=1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _get_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])


def _set_params(model: nn.Module, vec: torch.Tensor) -> None:
    offset = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n = p.numel()
        p.data.copy_(vec[offset:offset + n].reshape(p.shape))
        offset += n


def train_and_eval_qubit_quantum(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    n_qubits: int,
    depth: int = 4,
    epochs: int = 300,
    lr: float = 1e-2,
    batch_size: int = 64,
    seed: int = 0,
    optimizer_name: str = "SPSA",
    loss: str = "hloss",
    soft_train: torch.Tensor | None = None,
    soft_test: torch.Tensor | None = None,
) -> float:
    """Train one :class:`QubitClassifier` and return test accuracy.

    Defaults match Havlíček et al.: depth ``L=4``, optimizer SPSA, sigmoid
    ``hloss`` with a learnable bias.
    """
    if loss not in ("ce", "mse", "hloss"):
        raise ValueError(f"loss must be one of 'ce'/'mse'/'hloss', got {loss!r}")
    if loss == "mse" and (soft_train is None or soft_test is None):
        raise ValueError("loss='mse' requires soft_train and soft_test.")

    torch.manual_seed(seed)
    y_tr = y_train.long()
    y_te = y_test.long()

    learnable_bias = (loss == "hloss")
    model = QubitClassifier(
        n_qubits=n_qubits, depth=depth, learnable_bias=learnable_bias,
    )

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    if loss == "mse":
        s_tr = soft_train.view(-1).to(torch.float32)
    else:
        s_tr = None

    def _batch_loss(model, x, yb, sb):
        if loss == "ce":
            return ce(model(x), yb)
        if loss == "mse":
            return mse(model.parity_expectation(x), sb)
        # hloss: -log σ(α · y_pm · (parity_exp + b)), with α = 5 (paper-inspired)
        # y_pm ∈ {-1, +1}: derived from yb ∈ {0, 1}
        y_pm = 2.0 * yb.to(torch.float32) - 1.0
        alpha = 5.0
        return -F.logsigmoid(alpha * y_pm * model.parity_expectation(x)).mean()

    # Gradient-free branch (CMA / scipy) ---------------------------------
    if optimizer_name in ("CMA", "COBYLA", "NelderMead", "Powell"):
        import numpy as np
        with torch.no_grad():
            def objective(w: np.ndarray) -> float:
                _set_params(model, torch.tensor(w, dtype=torch.float32))
                return _batch_loss(model, X_train, y_tr, s_tr).item()

        x0 = _get_params(model).numpy()
        max_evals = max(epochs, 10)

        if optimizer_name == "CMA":
            import cma
            opts = cma.CMAOptions()
            opts["maxfevals"] = max_evals
            opts["verbose"] = -9
            opts["tolx"] = 1e-5
            opts["tolfun"] = 1e-5
            es = cma.CMAEvolutionStrategy(x0, 0.5, opts)
            es.optimize(objective)
            best = es.result.xbest
        else:
            from scipy.optimize import minimize
            method = {"COBYLA": "COBYLA",
                      "NelderMead": "Nelder-Mead",
                      "Powell": "Powell"}[optimizer_name]
            kw = ({"maxiter": max_evals, "disp": False}
                  if optimizer_name == "COBYLA"
                  else {"maxfev": max_evals, "disp": False})
            res = minimize(objective, x0, method=method, options=kw)
            best = res.x
        _set_params(model, torch.tensor(best, dtype=torch.float32))

    # SPSA branch — the paper's optimizer ------------------------------
    elif optimizer_name == "SPSA":
        # Simultaneous Perturbation Stochastic Approximation (Spall 1992).
        # Two function evaluations per step, gradient-free, noise-robust.
        loader = DataLoader(
            TensorDataset(X_train, y_tr) if loss != "mse"
            else TensorDataset(X_train, y_tr, s_tr),
            batch_size=batch_size, shuffle=True,
        )
        # SPSA hyperparameters (Spall recommends a, c decaying with k)
        a, c = lr, 0.1
        A = epochs * len(loader) // 10
        alpha_decay, gamma_decay = 0.602, 0.101

        step = 0
        for _ in range(epochs):
            for batch in loader:
                if loss == "mse":
                    xb, yb, sb = batch
                else:
                    xb, yb, sb = batch[0], batch[1], None
                w = _get_params(model)
                ak = a / (step + 1 + A) ** alpha_decay
                ck = c / (step + 1) ** gamma_decay
                # Random ±1 perturbation
                delta = (torch.randint(0, 2, w.shape).float() * 2 - 1)
                with torch.no_grad():
                    _set_params(model, w + ck * delta)
                    L_plus = _batch_loss(model, xb, yb, sb).item()
                    _set_params(model, w - ck * delta)
                    L_minus = _batch_loss(model, xb, yb, sb).item()
                    g = (L_plus - L_minus) / (2 * ck) * (1.0 / delta)
                    _set_params(model, w - ak * g)
                step += 1

    # Gradient-based branch (Adam / AdamW / SGD) -------------------------
    else:
        if loss == "mse":
            loader = DataLoader(
                TensorDataset(X_train, y_tr, s_tr),
                batch_size=batch_size, shuffle=True,
            )
        else:
            loader = DataLoader(
                TensorDataset(X_train, y_tr),
                batch_size=batch_size, shuffle=True,
            )
        if optimizer_name == "AdamW":
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        elif optimizer_name == "SGD":
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        model.train()
        for _ in range(epochs):
            for batch in loader:
                opt.zero_grad()
                if loss == "mse":
                    xb, yb, sb = batch
                    _batch_loss(model, xb, yb, sb).backward()
                else:
                    xb, yb = batch
                    _batch_loss(model, xb, yb, None).backward()
                opt.step()
            sched.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_test).argmax(dim=1)
        return (preds == y_te).float().mean().item()


# ---------------------------------------------------------------------------
# Search over depths
# ---------------------------------------------------------------------------

def find_min_qubit_depth(
    m: int,
    k: int,
    generator: str = "qubit_quantum",
    target_accuracy: float = 0.90,
    dataset_size: int = 2000,
    depths: list[int] | None = None,
    optimizer_name: str = "SPSA",
    test_fraction: float = 0.20,
    epochs: int = 300,
    lr: float = 1e-2,
    batch_size: int = 64,
    data_seed: int = 42,
    model_seed: int = 0,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_resample_iter: int = 10,
    loss: str = "hloss",
) -> None:
    """Find the minimum variational depth reaching the target accuracy.

    Defaults match Havlíček et al.: SPSA optimizer + sigmoid ``hloss``.
    """
    if depths is None:
        depths = [0, 1, 2, 3, 4]
    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator!r}.")

    margin_str = f"  min_margin={min_margin}" if min_margin > 0 else ""
    print(
        f"Generating dataset  generator={generator}  m={m}  k={k}  "
        f"size={dataset_size}  balanced={balanced}{margin_str} ..."
    )
    ds = GENERATORS[generator](
        size=dataset_size, m=m, k=k, seed=data_seed,
        balanced=balanced, min_margin=min_margin,
        bail_threshold=bail_threshold, max_iter=max_resample_iter,
    )

    if loss == "mse" and ds.soft_targets is None:
        raise ValueError(
            f"loss='mse' requires soft_targets; generator={generator!r} does not provide them."
        )

    n_test = int(len(ds.X) * test_fraction)
    n_train = len(ds.X) - n_test
    if ds.soft_targets is not None:
        soft_col = ds.soft_targets.view(-1)
        full = TensorDataset(ds.X, ds.y, soft_col)
    else:
        full = TensorDataset(ds.X, ds.y)
    tr, te = random_split(
        full, [n_train, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    X_train, y_train = tr[:][0], tr[:][1]
    X_test, y_test = te[:][0], te[:][1]
    if ds.soft_targets is not None:
        soft_train, soft_test = tr[:][2], te[:][2]
    else:
        soft_train = soft_test = None

    print(f"Train: {n_train}  Test: {n_test}")
    print(f"Target accuracy: {target_accuracy:.0%}")

    # Sanity check: the teacher (when available) labels its own dataset perfectly
    if generator == "qubit_quantum" and "teacher_theta" in ds.metadata:
        acc = validate_teacher_params_qubit(X_test, y_test, ds.metadata)
        print(f"Teacher (exact params) validation: {acc:.2%}  ← should be 100%")

    n_qubits = m - 1
    print(f"\n{'depth':>6}  {'params':>8}  {'accuracy':>10}")
    print("-" * 30)

    found = None
    for d in depths:
        n_p = n_variational_params(n_qubits, d) + (1 if loss == "hloss" else 0)
        acc = train_and_eval_qubit_quantum(
            X_train, y_train, X_test, y_test,
            n_qubits=n_qubits, depth=d,
            epochs=epochs, lr=lr, batch_size=batch_size,
            seed=model_seed, optimizer_name=optimizer_name,
            loss=loss, soft_train=soft_train, soft_test=soft_test,
        )
        marker = " <-- first hit" if found is None and acc >= target_accuracy else ""
        print(f"{d:>6}  {n_p:>8}  {acc:>10.2%}{marker}")
        if found is None and acc >= target_accuracy:
            found = d
            break

    print()
    if found is not None:
        print(f"Minimal depth to reach {target_accuracy:.0%}: {found}")
    else:
        print(f"No depth in {depths} reached {target_accuracy:.0%}.")
