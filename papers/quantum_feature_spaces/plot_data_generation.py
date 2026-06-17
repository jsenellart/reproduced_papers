"""plot_data_generation.py — unified dataset visualiser across all generators and observables.

Display mode is chosen automatically from --n-features:

  n_features = 2  → direct 2D scatter (x₀ vs x₁)
  n_features = 3  → x₂-sliced: --n-slices rows, each showing x₀ vs x₁
                    for one x₂-bin.  Generates n_slices × --size points so
                    each slice averages --size points visually.
  n_features ≥ 4  → PCA projection to 2D.

Columns (fixed):
  Photonic×parity | Photonic×majority | Photonic×bunching | Photonic×single_output
  | Qubit | Analytical | MLP

Options:
  --n-features N   input / feature dimension  [required]
  --k K            photons (photonic) / depth-complexity (others)  [default 3]
  --m M            photonic circuit modes only (≥ n_features+1, preferably even)
                   [default: smallest even ≥ n_features+2]
  --n-slices S     x₂ bins for 3D mode  [default 3]
  --size N         target points per visual panel  [default 1000]
  --seed S         RNG seed  [default 2]
  --balanced       enforce 50/50 balance  [default on]
  --min-margin F   margin filter  [default 0.0]
  --nshots N       measurement shots for quantum generators (0 = exact)
  --save PATH      output file

Usage examples:
  python papers/quantum_feature_spaces/plot_data_generation.py --n-features 2 --k 3
  python papers/quantum_feature_spaces/plot_data_generation.py --n-features 3 --k 3 --m 6 --seed 2
  python papers/quantum_feature_spaces/plot_data_generation.py --n-features 5 --k 3 --m 6
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import numpy as np

from data import GENERATORS
from data.photonic_quantum import generate_photonic_quantum

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

COLUMNS = [
    ("photonic", "parity"),
    ("photonic", "majority"),
    ("photonic", "bunching"),
    ("photonic", "single_output"),
    ("qubit_quantum", None),
    ("analytical",   None),
    ("mlp",          None),
]

COL_TITLES = {
    ("photonic", "parity"):        "Photonic\n× parity",
    ("photonic", "majority"):      "Photonic\n× majority",
    ("photonic", "bunching"):      "Photonic\n× bunching",
    ("photonic", "single_output"): "Photonic\n× single output",
    ("qubit_quantum", None):       "Qubit\n(IQP, Havlíček)",
    ("analytical",   None):        "Analytical\n(Σ sin)",
    ("mlp",          None):        "MLP teacher",
}

C0, C1 = "#4C72B0", "#DD8452"


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _default_m(n_features: int) -> int:
    """Smallest even number ≥ n_features + 2 (keeps majority valid)."""
    m = n_features + 2
    return m if m % 2 == 0 else m + 1


def _generate_col(
    col: tuple,
    n_features: int,
    k: int,
    m_photonic: int,
    size: int,
    seed: int,
    balanced: bool,
    min_margin: float,
    bail_threshold: float,
    nsample: int = 0,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Generate dataset for one column.  Returns (X, y, n_raw) or None if n/a."""
    gen, obs = col

    if gen == "photonic":
        if obs == "majority" and m_photonic % 2 != 0:
            return None
        ds = generate_photonic_quantum(
            size=size, m=m_photonic, k=k, n_features=n_features,
            seed=seed, observable=obs,
            balanced=balanced, min_margin=min_margin,
            bail_threshold=bail_threshold,
            nsample=nsample,
        )
    else:
        # Non-photonic: feature dim = n_features → pass m = n_features + 1
        m_classic = n_features + 1
        ds = GENERATORS[gen](
            size=size, m=m_classic, k=k,
            seed=seed, balanced=balanced, min_margin=min_margin,
            bail_threshold=bail_threshold,
            nsample=nsample,  # ignored by analytical/mlp via **_
        )

    X, y = ds.X.numpy(), ds.y.numpy()
    n_raw = ds.metadata.get("total_raw_drawn", len(X))
    return X, y, n_raw


def generate_all(
    n_features, k, m_photonic, size, seed, balanced, min_margin, bail_threshold,
    n_slices=1, nsample=0,
) -> dict:
    """Generate datasets for every column.  n_slices > 1 multiplies size."""
    total = size * n_slices
    data = {}
    for col in COLUMNS:
        gen, obs = col
        label = f"{gen}×{obs}" if obs else gen
        print(f"  [{label}] generating {total} ...", flush=True)
        try:
            entry = _generate_col(
                col, n_features=n_features, k=k, m_photonic=m_photonic,
                size=total, seed=seed,
                balanced=balanced, min_margin=min_margin,
                bail_threshold=bail_threshold,
                nsample=nsample,
            )
        except Exception as e:
            print(f"    ERROR: {e}")
            entry = None
        if entry is not None:
            X, y, n_raw = entry
            print(f"    n={len(y)}  n0={(y==0).sum()}  n1={(y==1).sum()}  raw={n_raw}")
        data[col] = entry
    return data


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _setup_ax(ax, xlabel=True, ylabel=True):
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ticks = [0, np.pi, 2 * np.pi]
    if xlabel:
        ax.set_xticks(ticks)
        ax.set_xticklabels(["0", "π", "2π"], fontsize=7)
        ax.set_xlabel("x₀", fontsize=8)
    else:
        ax.set_xticks([])
    if ylabel:
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0", "π", "2π"], fontsize=7)
        ax.set_ylabel("x₁", fontsize=8)
    else:
        ax.set_yticks([])


def _scatter(ax, X, y, xi=0, xj=1, alpha=0.65, ms=12):
    for cls, color in [(0, C0), (1, C1)]:
        mask = y == cls
        ax.scatter(X[mask, xi], X[mask, xj],
                   s=ms, alpha=alpha, color=color,
                   edgecolors="none", rasterized=True)


def _empty(ax, msg="n/a"):
    ax.text(0.5, 0.5, msg, ha="center", va="center",
            transform=ax.transAxes, fontsize=8, color="grey")
    ax.set_axis_off()


def _shared_legend(fig):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C0,
               markersize=7, label="class 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C1,
               markersize=7, label="class 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))


# ---------------------------------------------------------------------------
# Mode: 2D direct
# ---------------------------------------------------------------------------

def plot_2d(data, alpha, ms, save, no_show, title_extra=""):
    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 4.0),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Dataset scatter — " + title_extra, fontsize=11, fontweight="bold")

    for col_idx, col in enumerate(COLUMNS):
        ax = axes[0][col_idx]
        ax.set_title(COL_TITLES[col], fontsize=8)
        entry = data[col]
        if entry is None:
            _empty(ax, "n/a\n(majority: odd m)")
            continue
        X, y, n_raw = entry
        if len(y) == 0:
            _empty(ax, "no samples")
            continue
        _scatter(ax, X, y, alpha=alpha, ms=ms)
        _setup_ax(ax, xlabel=True, ylabel=(col_idx == 0))
        ax.set_title(f"{COL_TITLES[col]}\nn₀={(y==0).sum()} n₁={(y==1).sum()}",
                     fontsize=8)

    _shared_legend(fig)
    _save_show(fig, save, no_show)


# ---------------------------------------------------------------------------
# Mode: 3D sliced
# ---------------------------------------------------------------------------

def plot_sliced(data, n_slices, alpha, ms, save, no_show, title_extra=""):
    slice_edges = np.linspace(0, 2 * np.pi, n_slices + 1)
    n_cols = len(COLUMNS)

    fig, axes = plt.subplots(n_slices, n_cols,
                             figsize=(3.4 * n_cols, 3.4 * n_slices),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Dataset scatter (x₂-sliced) — " + title_extra,
                 fontsize=11, fontweight="bold")

    for col_idx, col in enumerate(COLUMNS):
        entry = data[col]
        for si in range(n_slices):
            ax = axes[si][col_idx]
            lo, hi = slice_edges[si], slice_edges[si + 1]
            slice_label = f"x₂∈[{lo/np.pi:.2g}π,{hi/np.pi:.2g}π)"

            if si == 0:
                ax.set_title(COL_TITLES[col], fontsize=8)

            if entry is None:
                _empty(ax, "n/a")
                continue
            X, y, _ = entry
            if len(y) == 0:
                _empty(ax, "no samples")
                _setup_ax(ax, xlabel=(si == n_slices - 1), ylabel=(col_idx == 0))
                continue

            mask = (X[:, 2] >= lo) & (X[:, 2] < hi)
            Xs, ys = X[mask], y[mask]
            _scatter(ax, Xs, ys, alpha=alpha, ms=ms)
            _setup_ax(ax, xlabel=(si == n_slices - 1), ylabel=(col_idx == 0))

            n0, n1 = (ys == 0).sum(), (ys == 1).sum()
            ax.set_ylabel(f"{slice_label}\nx₁" if col_idx == 0 else "",
                          fontsize=7 if col_idx == 0 else 0)
            if col_idx == 0:
                ax.set_ylabel(f"x₁  [{slice_label}]", fontsize=7)
            ax.set_title(
                (COL_TITLES[col] + "\n" if si == 0 else "") +
                f"n₀={n0} n₁={n1}",
                fontsize=7,
            )

    _shared_legend(fig)
    _save_show(fig, save, no_show)


# ---------------------------------------------------------------------------
# Mode: PCA
# ---------------------------------------------------------------------------

def plot_pca(data, alpha, ms, save, no_show, title_extra=""):
    from sklearn.decomposition import PCA

    n_cols = len(COLUMNS)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.6 * n_cols, 4.2),
                             constrained_layout=True, squeeze=False)
    fig.suptitle("Dataset scatter (PCA→2D) — " + title_extra,
                 fontsize=11, fontweight="bold")

    for col_idx, col in enumerate(COLUMNS):
        ax = axes[0][col_idx]
        entry = data[col]
        if entry is None:
            _empty(ax, "n/a")
            ax.set_title(COL_TITLES[col], fontsize=8)
            continue
        X, y, n_raw = entry
        if len(y) == 0:
            _empty(ax, "no samples")
            ax.set_title(COL_TITLES[col], fontsize=8)
            continue

        pca = PCA(n_components=2).fit(X)
        X2 = pca.transform(X)
        var = pca.explained_variance_ratio_

        for cls, color in [(0, C0), (1, C1)]:
            mask = y == cls
            ax.scatter(X2[mask, 0], X2[mask, 1],
                       s=ms, alpha=alpha, color=color,
                       edgecolors="none", rasterized=True)

        ax.set_xlabel(f"PC1 ({var[0]:.0%})", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(f"PC2 ({var[1]:.0%})", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{COL_TITLES[col]}\n"
            f"n₀={(y==0).sum()} n₁={(y==1).sum()}  "
            f"var={var[:2].sum():.0%}",
            fontsize=8,
        )

    _shared_legend(fig)
    _save_show(fig, save, no_show)


# ---------------------------------------------------------------------------
# Save / show helper
# ---------------------------------------------------------------------------

def _save_show(fig, save, no_show):
    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=180, bbox_inches="tight")
        print(f"\nSaved → {save}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-features", type=int, required=True,
                        help="Feature / input dimension. Controls display mode: "
                             "2=scatter, 3=sliced, ≥4=PCA.")
    parser.add_argument("--k", type=int, default=3,
                        help="Photons (photonic) / depth-complexity (others). Default 3.")
    parser.add_argument("--m", type=int, default=None,
                        help="Photonic circuit modes only (≥ n_features+1, preferably even). "
                             "Default: smallest even ≥ n_features+2.")
    parser.add_argument("--n-slices", type=int, default=3,
                        help="Number of x₂ slices in 3D mode. Default 3.")
    parser.add_argument("--size", type=int, default=1000,
                        help="Target points per visual panel. Default 1000.")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--no-balance", dest="balanced", action="store_false")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--bail-threshold", type=float, default=0.0)
    parser.add_argument("--nshots", type=int, default=0,
                        help="Measurement shots for quantum generators (0 = exact). "
                             "Photonic: Merlin shot sampler. "
                             "Qubit: Gaussian CLT noise with std=sqrt((1-E²)/N).")
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--marker-size", type=float, default=12.0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    n_features = args.n_features
    k = args.k
    m_photonic = args.m if args.m is not None else _default_m(n_features)

    if m_photonic < n_features + 1:
        parser.error(f"--m {m_photonic} must be ≥ n_features+1 = {n_features+1}.")

    mode = "2d" if n_features == 2 else ("sliced" if n_features == 3 else "pca")
    n_slices = args.n_slices if mode == "sliced" else 1

    margin_str = f"min_margin={args.min_margin}" if args.min_margin > 0 else "no margin"
    shots_str  = f", {args.nshots} shots" if args.nshots > 0 else ", exact"
    title_extra = (
        f"n_features={n_features}, k={k}"
        + (f", m={m_photonic} (photonic)" if True else "")
        + f", {args.size} pts/panel, {margin_str}{shots_str}"
        + (" (balanced)" if args.balanced else "")
    )
    print(f"Mode: {mode}  |  {title_extra}")

    data = generate_all(
        n_features=n_features, k=k, m_photonic=m_photonic,
        size=args.size, seed=args.seed,
        balanced=args.balanced, min_margin=args.min_margin,
        bail_threshold=args.bail_threshold,
        n_slices=n_slices,
        nsample=args.nshots,
    )

    kw = dict(alpha=args.alpha, ms=args.marker_size,
              save=args.save, no_show=args.no_show, title_extra=title_extra)

    if mode == "2d":
        plot_2d(data, **kw)
    elif mode == "sliced":
        plot_sliced(data, n_slices=n_slices, **kw)
    else:
        plot_pca(data, **kw)
