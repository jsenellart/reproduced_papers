"""plot_photonic_observables.py — photonic observable comparison across (m, k) configs.

Compares parity / majority / bunching / single_output for multiple (m, k) configurations.

Two display modes:

Default (--n-features 2):
    Direct 2D scatter (x0 vs x1) for each (observable, m, k) config.
    Rows = observables, Cols = (m, k) configs.

Slice mode (--n-features 3, automatically activated when n_features >= 3):
    For a single (m, k) config, divides x2 into N equal slices and plots
    x0 vs x1 within each slice.
    Rows = observables, Cols = x2-slices.
    Generates N × size points so each slice averages ~size points visually.

Usage:
    # 2D direct (n_features=2), multiple configs
    python papers/quantum_feature_spaces/plot_photonic_observables.py --seed 2

    # 3D slices (n_features=3) for a single config
    python papers/quantum_feature_spaces/plot_photonic_observables.py \\
        --n-features 3 --configs 6,3 --n-slices 3 --seed 2 --save out.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from data.photonic_quantum import generate_photonic_quantum

OBSERVABLES = ["parity", "majority", "bunching", "single_output"]
OBS_LABELS = {
    "parity":        "Parity\n(−1)^{n_left}",
    "majority":      "Majority\nsign(n_left − n_right)",
    "bunching":      "Bunching\nP(anti)−P(bunched)",
    "single_output": "Single output\nP(in)−P(rev)",
}
C0, C1 = "#4C72B0", "#DD8452"


def _scatter(ax, X, y, alpha, ms):
    for cls, color in [(0, C0), (1, C1)]:
        mask = y == cls
        ax.scatter(X[mask, 0], X[mask, 1],
                   s=ms, alpha=alpha, color=color,
                   edgecolors="none", rasterized=True)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(0, 2 * np.pi)
    ax.set_aspect("equal")
    ticks = [0, np.pi, 2 * np.pi]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0", "π", "2π"], fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(["0", "π", "2π"], fontsize=7)


def plot_observables(
    configs: list[tuple[int, int]],   # [(m, k), ...]
    n_features: int = 2,
    size: int = 1000,
    seed: int = 42,
    balanced: bool = True,
    min_margin: float = 0.1,
    bail_threshold: float = 0.0,
    alpha: float = 0.65,
    marker_size: float = 12.0,
    save: str | None = None,
    no_show: bool = False,
) -> None:
    n_obs = len(OBSERVABLES)
    n_cfg = len(configs)

    # Generate all datasets
    data = {}   # (observable, m, k) -> (X, y, n_raw)
    for obs in OBSERVABLES:
        for m, k in configs:
            if obs == "majority" and m % 2 != 0:
                data[(obs, m, k)] = None
                continue
            print(f"  [{obs}  m={m} k={k}] generating ...", flush=True)
            ds = generate_photonic_quantum(
                size=size, m=m, k=k, n_features=n_features,
                seed=seed, observable=obs,
                balanced=balanced, min_margin=min_margin,
                bail_threshold=bail_threshold,
            )
            X, y = ds.X.numpy(), ds.y.numpy()
            n_raw = ds.metadata.get("total_raw_drawn", "?")
            print(f"    n={len(y)}  n0={(y==0).sum()}  n1={(y==1).sum()}  raw={n_raw}")
            data[(obs, m, k)] = (X, y, n_raw)

    margin_str = f"min_margin={min_margin}" if min_margin > 0 else "no margin"
    fig, axes = plt.subplots(
        n_obs, n_cfg,
        figsize=(3.8 * n_cfg, 3.8 * n_obs),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Photonic observables — n_features={n_features}, {size} samples, "
        f"{margin_str}" + (" (balanced)" if balanced else ""),
        fontsize=12, fontweight="bold",
    )

    for row, obs in enumerate(OBSERVABLES):
        for col, (m, k) in enumerate(configs):
            ax = axes[row][col]
            entry = data[(obs, m, k)]

            if entry is None:
                ax.text(0.5, 0.5, "n/a\n(majority needs even m)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                continue

            X, y, n_raw = entry
            if len(y) == 0:
                ax.text(0.5, 0.5, "no samples\n(too few marginal survivors)",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
            else:
                _scatter(ax, X, y, alpha=alpha, ms=marker_size)

            n0, n1 = (y == 0).sum(), (y == 1).sum()
            ax.set_title(f"m={m}, k={k}  |  n₀={n0} n₁={n1}\n(drew {n_raw} raw)",
                         fontsize=8)
            ax.set_xlabel("x₀", fontsize=8)
            if col == 0:
                ax.set_ylabel("x₁", fontsize=8)

        # Row label on left
        axes[row][0].annotate(
            OBS_LABELS[obs],
            xy=(-0.28, 0.5), xycoords="axes fraction",
            fontsize=9, fontweight="bold", va="center", ha="right",
        )

    # Shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C0,
               markersize=8, label="class 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C1,
               markersize=8, label="class 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=180, bbox_inches="tight")
        print(f"\nSaved → {save}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


N_SLICES = 3
SLICE_EDGES = np.linspace(0, 2 * np.pi, N_SLICES + 1)


def plot_observables_sliced(
    m: int,
    k: int,
    n_features: int = 3,
    n_slices: int = 3,
    size: int = 1000,
    seed: int = 2,
    balanced: bool = True,
    min_margin: float = 0.0,
    bail_threshold: float = 0.0,
    alpha: float = 0.65,
    marker_size: float = 10.0,
    save: str | None = None,
    no_show: bool = False,
) -> None:
    """3D slice view: rows = observables, cols = x2-slices (x0 vs x1)."""
    slice_edges = np.linspace(0, 2 * np.pi, n_slices + 1)
    slice_labels = [
        f"x₂ ∈ [{slice_edges[i]/np.pi:.2g}π, {slice_edges[i+1]/np.pi:.2g}π)"
        for i in range(n_slices)
    ]

    # Generate n_slices × size points per observable so each slice averages ~size
    total = n_slices * size
    data = {}
    for obs in OBSERVABLES:
        if obs == "majority" and m % 2 != 0:
            data[obs] = None
            continue
        print(f"  [{obs}  m={m} k={k} n_features={n_features}] generating {total} ...",
              flush=True)
        ds = generate_photonic_quantum(
            size=total, m=m, k=k, n_features=n_features,
            seed=seed, observable=obs,
            balanced=balanced, min_margin=min_margin,
            bail_threshold=bail_threshold,
        )
        X, y = ds.X.numpy(), ds.y.numpy()
        n_raw = ds.metadata.get("total_raw_drawn", "?")
        print(f"    n={len(y)}  n0={(y==0).sum()}  n1={(y==1).sum()}  raw={n_raw}")
        data[obs] = (X, y, n_raw)

    margin_str = f"min_margin={min_margin}" if min_margin > 0 else "no margin"
    fig, axes = plt.subplots(
        len(OBSERVABLES), n_slices,
        figsize=(3.6 * n_slices, 3.6 * len(OBSERVABLES)),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Photonic observables — m={m}, k={k}, n_features={n_features}, "
        f"{size} pts/slice, {margin_str}" + (" (balanced)" if balanced else ""),
        fontsize=11, fontweight="bold",
    )

    for row, obs in enumerate(OBSERVABLES):
        entry = data[obs]
        for col in range(n_slices):
            ax = axes[row][col]
            lo, hi = slice_edges[col], slice_edges[col + 1]

            if entry is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_axis_off()
                continue

            X, y, n_raw = entry
            mask = (X[:, 2] >= lo) & (X[:, 2] < hi)
            Xs, ys = X[mask], y[mask]

            if len(ys) == 0:
                ax.text(0.5, 0.5, "no samples", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8)
                ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, 2 * np.pi)
            else:
                _scatter(ax, Xs, ys, alpha=alpha, ms=marker_size)

            n0, n1 = (ys == 0).sum(), (ys == 1).sum()
            ax.set_title(f"{slice_labels[col]}\nn₀={n0} n₁={n1}", fontsize=8)
            if row == len(OBSERVABLES) - 1:
                ax.set_xlabel("x₀", fontsize=8)
            if col == 0:
                ax.set_ylabel("x₁", fontsize=8)

        # Row label
        axes[row][0].annotate(
            OBS_LABELS[obs],
            xy=(-0.30, 0.5), xycoords="axes fraction",
            fontsize=9, fontweight="bold", va="center", ha="right",
        )

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C0,
               markersize=8, label="class 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C1,
               markersize=8, label="class 1"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    if save:
        os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
        fig.savefig(save, dpi=180, bbox_inches="tight")
        print(f"\nSaved → {save}")

    if not no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", nargs="+", default=["4,2", "6,2", "6,3"],
                        metavar="M,K",
                        help="(m,k) configs to show, e.g. --configs 4,2 6,2 6,3")
    parser.add_argument("--n-features", type=int, default=2)
    parser.add_argument("--n-slices", type=int, default=3)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--balanced", action="store_true", default=True)
    parser.add_argument("--no-balance", dest="balanced", action="store_false")
    parser.add_argument("--min-margin", type=float, default=0.0)
    parser.add_argument("--bail-threshold", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--marker-size", type=float, default=12.0)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    configs = [tuple(int(x) for x in c.split(",")) for c in args.configs]

    if args.n_features >= 3:
        # Slice mode: single config expected
        if len(configs) != 1:
            parser.error("Slice mode (--n-features >= 3) expects exactly one --configs M,K.")
        m, k = configs[0]
        plot_observables_sliced(
            m=m, k=k,
            n_features=args.n_features,
            n_slices=args.n_slices,
            size=args.size,
            seed=args.seed,
            balanced=args.balanced,
            min_margin=args.min_margin,
            bail_threshold=args.bail_threshold,
            alpha=args.alpha,
            marker_size=args.marker_size,
            save=args.save,
            no_show=args.no_show,
        )
    else:
        plot_observables(
            configs=configs,
            n_features=args.n_features,
            size=args.size,
            seed=args.seed,
            balanced=args.balanced,
            min_margin=args.min_margin,
            bail_threshold=args.bail_threshold,
            alpha=args.alpha,
            marker_size=args.marker_size,
            save=args.save,
            no_show=args.no_show,
        )
