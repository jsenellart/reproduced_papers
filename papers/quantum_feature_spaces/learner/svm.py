"""SVM learner for quantum feature-space benchmark.

Uses sklearn's SVC with an RBF kernel on Fourier-lifted features.
Inputs are angles in [0, 2π] so wrapping-aware features matter:
  φ(x) = [sin(x), cos(x), sin(2x), cos(2x), ..., sin(Fx), cos(Fx)]
Dimensionality: 2 * fourier_order * n_features.

The search axis is the SVM regularisation parameter C.
Large C → low bias / high variance; small C → more regularisation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split

from data import GENERATORS


# ---------------------------------------------------------------------------
# Feature transform
# ---------------------------------------------------------------------------

def _fourier_features(X: np.ndarray, fourier_order: int) -> np.ndarray:
    """Expand angles X (N, d) into Fourier features (N, 2*fourier_order*d)."""
    parts = []
    for k in range(1, fourier_order + 1):
        parts.append(np.sin(k * X))
        parts.append(np.cos(k * X))
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
# Single SVM fit + eval
# ---------------------------------------------------------------------------

def train_and_eval_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    fourier_order: int = 3,
) -> tuple[float, int]:
    """Fit one SVC and return (test_accuracy, n_support_vectors)."""
    from sklearn.svm import SVC

    if fourier_order > 0:
        X_tr = _fourier_features(X_train, fourier_order)
        X_te = _fourier_features(X_test,  fourier_order)
    else:
        X_tr, X_te = X_train, X_test

    clf = SVC(C=C, kernel=kernel, gamma=gamma)
    clf.fit(X_tr, y_train)
    acc = (clf.predict(X_te) == y_test).mean()
    return float(acc), int(clf.support_vectors_.shape[0])


# ---------------------------------------------------------------------------
# Search for minimal C
# ---------------------------------------------------------------------------

def find_min_svm(
    m: int,
    k: int,
    generator: str = "photonic_quantum",
    target_accuracy: float = 0.90,
    dataset_size: int = 10000,
    C_values: list[float] | None = None,
    kernel: str = "rbf",
    gamma: str | float = "scale",
    fourier_order: int = 3,
    test_fraction: float = 0.20,
    data_seed: int = 42,
    balanced: bool = False,
    min_margin: float = 0.0,
    bail_threshold: float = 0.30,
    max_resample_iter: int = 10,
    # Accept (and ignore) gradient-based training args so callers can pass a
    # uniform **common dict without KeyErrors.
    **_,
) -> None:
    """Sweep SVM regularisation values and find the first C reaching target accuracy.

    Parameters
    ----------
    C_values : list of float
        Regularisation values to try in ascending order (larger C = less regularisation).
        Default: [0.01, 0.1, 1, 10, 100, 1000].
    kernel : str
        SVM kernel type (default 'rbf').
    gamma : str or float
        RBF kernel coefficient (default 'scale' = 1/(n_features*X.var())).
    fourier_order : int
        Wrap-aware Fourier expansion order applied before the kernel.
        0 = raw angle features (not recommended for periodic data).
    """
    from sklearn.preprocessing import StandardScaler

    if C_values is None:
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    if generator not in GENERATORS:
        raise ValueError(f"Unknown generator {generator!r}. Choose from {list(GENERATORS)}.")

    margin_str = f"  min_margin={min_margin}" if min_margin > 0.0 else ""
    print(f"Generating dataset  generator={generator}  m={m}  k={k}  size={dataset_size}"
          f"  balanced={balanced}{margin_str} ...")
    ds = GENERATORS[generator](size=dataset_size, m=m, k=k, seed=data_seed,
                               balanced=balanced, min_margin=min_margin,
                               bail_threshold=bail_threshold,
                               max_iter=max_resample_iter)

    n_test  = int(len(ds.X) * test_fraction)
    n_train = len(ds.X) - n_test
    train_ds, test_ds = random_split(
        TensorDataset(ds.X, ds.y),
        [n_train, n_test],
        generator=torch.Generator().manual_seed(data_seed),
    )
    X_train = train_ds[:][0].numpy()
    y_train = train_ds[:][1].numpy().astype(int)
    X_test  = test_ds[:][0].numpy()
    y_test  = test_ds[:][1].numpy().astype(int)

    print(f"Train: {n_train}  Test: {n_test}")
    print(f"Target accuracy: {target_accuracy:.0%}")
    feat_dim = 2 * fourier_order * X_train.shape[1] if fourier_order > 0 else X_train.shape[1]
    print(f"SVM kernel={kernel}  gamma={gamma}  fourier_order={fourier_order}  feat_dim={feat_dim}")

    # Scale features to zero-mean / unit-variance (standard for SVM)
    if fourier_order > 0:
        X_tr_feat = _fourier_features(X_train, fourier_order)
        X_te_feat = _fourier_features(X_test,  fourier_order)
    else:
        X_tr_feat, X_te_feat = X_train, X_test

    scaler = StandardScaler()
    X_tr_feat = scaler.fit_transform(X_tr_feat)
    X_te_feat = scaler.transform(X_te_feat)

    print(f"\n{'C':>10}  {'accuracy':>10}  {'n_sv':>8}")
    print("-" * 32)

    found: float | None = None
    for C in C_values:
        from sklearn.svm import SVC
        clf = SVC(C=C, kernel=kernel, gamma=gamma)
        clf.fit(X_tr_feat, y_train)
        acc   = float((clf.predict(X_te_feat) == y_test).mean())
        n_sv  = int(clf.support_vectors_.shape[0])
        marker = " <-- first hit" if found is None and acc >= target_accuracy else ""
        print(f"{C:>10.3g}  {acc:>10.2%}  {n_sv:>8}{marker}")
        if found is None and acc >= target_accuracy:
            found = C
            break

    print()
    if found is not None:
        print(f"Minimal C to reach {target_accuracy:.0%}: C={found}")
    else:
        print(f"No C in {C_values} reached {target_accuracy:.0%}.")
