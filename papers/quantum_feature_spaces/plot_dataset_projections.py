from __future__ import annotations

import argparse
from itertools import combinations
from math import ceil

import matplotlib.pyplot as plt

from data import GENERATORS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a dataset and plot all pairwise axis projections "
            "to inspect class geometry."
        )
    )
    parser.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default="quantum",
        help="Dataset generation strategy (default: quantum)",
    )
    parser.add_argument("--size", type=int, required=True, help="Number of samples")
    parser.add_argument("--m", type=int, required=True, help="Number of optical modes")
    parser.add_argument("--k", type=int, required=True, help="Number of injected photons")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument(
        "--nsample",
        type=int,
        default=0,
        help="Number of measurement shots, 0 for exact probabilities",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Subsample to get balanced classes",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.0,
        metavar="M",
        help=(
            "Drop samples whose decision confidence is below M. "
            "For quantum: |parity_expectation| >= M. "
            "For analytical: |normalised_score| >= M. "
            "For mlp: |p1 - p0| >= M. "
            "Use 0.3 to reproduce the Havl\u00ed\u010dek et al. separation gap. "
            "Default: 0.0 (keep all samples)."
        ),
    )
    parser.add_argument(
        "--bail-threshold",
        type=float,
        default=0.30,
        metavar="T",
        help=(
            "Abort early if first-batch yield is below T. "
            "Set to 0 to always iterate until size is reached (expensive). "
            "Default: 0.30."
        ),
    )
    parser.add_argument(
        "--projection",
        choices=["pairwise", "pca", "tsne"],
        default="pca",
        help=(
            "How to visualize the dataset. "
            "'pairwise': all 2D axis projections (informative only for d≤3). "
            "'pca': PCA to 2D (fast, linear). "
            "'tsne': t-SNE to 2D (slower, reveals non-linear clusters). "
            "Default: pca."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for the 1D fallback histogram",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.65,
        help="Point opacity for scatter plots",
    )
    parser.add_argument(
        "--marker-size",
        type=float,
        default=12.0,
        help="Marker size for scatter plots",
    )
    parser.add_argument(
        "--max-cols",
        type=int,
        default=3,
        help="Maximum number of subplot columns",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path for the figure",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window",
    )
    return parser


def _plot_reduced(X, y, method: str, alpha: float, marker_size: float) -> plt.Figure:
    """Reduce X to 2D with PCA or t-SNE and scatter-plot with class colours."""
    if method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        label = "PCA"
        ax_labels = ("PC1", "PC2")
    else:  # tsne
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=0, perplexity=min(30, max(5, len(X) // 10)))
        label = "t-SNE"
        ax_labels = ("t-SNE 1", "t-SNE 2")

    import numpy as np
    X2 = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, color in [(0, "tab:blue"), (1, "tab:orange")]:
        mask = y == cls
        ax.scatter(X2[mask, 0], X2[mask, 1],
                   s=marker_size, alpha=alpha, color=color,
                   label=f"class {cls}", edgecolors="none")

    if method == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(X.shape[1], 5)).fit(X)
        var = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%} var)")
        ax.set_ylabel(f"PC2 ({var[1]:.1%} var)")
        ax.set_title(f"PCA — 2 components explain {var[:2].sum():.1%} of variance")
    else:
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_title(f"t-SNE (perplexity={reducer.perplexity})")

    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def _plot_one_dimensional_projection(X, y, bins: int) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, color in [(0, "tab:blue"), (1, "tab:orange")]:
        mask = y == label
        ax.hist(
            X[mask, 0],
            bins=bins,
            alpha=0.55,
            label=f"class {label}",
            color=color,
        )
    ax.set_xlabel("x0")
    ax.set_ylabel("count")
    ax.set_title("One-dimensional projection")
    ax.legend()
    fig.tight_layout()
    return fig


def _plot_pairwise_projections(
    X,
    y,
    alpha: float,
    marker_size: float,
    max_cols: int,
) -> plt.Figure:
    n_features = X.shape[1]
    axis_pairs = list(combinations(range(n_features), 2))
    n_plots = len(axis_pairs)
    n_cols = min(max_cols, n_plots)
    n_rows = ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for ax, (i, j) in zip(axes.flat, axis_pairs):
        for label, color in [(0, "tab:blue"), (1, "tab:orange")]:
            mask = y == label
            ax.scatter(
                X[mask, i],
                X[mask, j],
                s=marker_size,
                alpha=alpha,
                label=f"class {label}",
                color=color,
                edgecolors="none",
            )
        ax.set_xlabel(f"x{i}")
        ax.set_ylabel(f"x{j}")
        ax.set_title(f"Projection on (x{i}, x{j})")

    for ax in axes.flat[n_plots:]:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def main() -> None:
    args = _build_parser().parse_args()
    gen_kwargs: dict = dict(
        size=args.size, m=args.m, k=args.k, seed=args.seed,
        balanced=args.balanced, min_margin=args.min_margin,
        bail_threshold=args.bail_threshold,
    )
    if args.generator == "photonic_quantum":
        gen_kwargs["nsample"] = args.nsample
    dataset = GENERATORS[args.generator](**gen_kwargs)

    X = dataset.X.numpy()
    y = dataset.y.numpy()

    print("Dataset metadata:")
    print(dataset.metadata)
    print(f"X shape: {tuple(dataset.X.shape)}")
    print(f"class 0: {(dataset.y == 0).sum().item()}  class 1: {(dataset.y == 1).sum().item()}")

    if X.shape[1] < 2:
        fig = _plot_one_dimensional_projection(X, y, bins=args.bins)
    elif args.projection == "pairwise":
        fig = _plot_pairwise_projections(
            X, y,
            alpha=args.alpha,
            marker_size=args.marker_size,
            max_cols=args.max_cols,
        )
    else:
        fig = _plot_reduced(X, y, method=args.projection,
                            alpha=args.alpha, marker_size=args.marker_size)

    if args.save is not None:
        fig.savefig(args.save, dpi=200, bbox_inches="tight")
        print(f"Saved figure to {args.save}")

    if not args.no_show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()