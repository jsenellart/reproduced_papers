from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from .reference import qubo_basics


@dataclass
class QuboInstance:
    """Container for a QUBO matrix and metadata."""

    matrix: np.ndarray
    constant: float = 0.0
    description: str = ""
    metadata: dict = field(default_factory=dict)


def qubo_objective(Q: np.ndarray, x: np.ndarray, constant: float = 0.0) -> float:
    """Compute x^T Q x + constant."""
    x_vec = np.asarray(x, dtype=float)
    return float(x_vec @ Q @ x_vec + constant)


def _symmetrize(Q: np.ndarray) -> np.ndarray:
    """Make Q symmetric with zero diagonal (matches authors' generators)."""
    upper = np.triu(Q)
    sym = upper + np.triu(upper, 1).T
    np.fill_diagonal(sym, 0.0)
    return sym


def generate_random_qubo(size: int, low: float = -2.0, high: float = 2.0, seed: int | None = None) -> np.ndarray:
    """Random QUBO generator (reference/save_data.py)."""
    Q = qubo_basics.generate_random_qubo(size, low=low, high=high, seed=seed)
    return _symmetrize(Q)


def generate_graph_coloring_qubo(
    num_nodes: int,
    num_colors: int,
    edges: Iterable[tuple[int, int]] | Iterable[list[int]],
    penalty: float = 4.0,
) -> QuboInstance:
    """Graph-coloring QUBO following reference/save_data_graph.py."""
    Q, constant = qubo_basics.generate_graph_coloring_qubo(num_nodes, num_colors, edges, penalty)
    metadata = {"num_nodes": num_nodes, "num_colors": num_colors, "edges": list(map(tuple, edges))}
    return QuboInstance(matrix=Q, constant=constant, description="graph-coloring", metadata=metadata)


def brute_force_solve(instance: QuboInstance, *, show_progress: bool = False) -> tuple[np.ndarray, float]:
    """Exact solver via full enumeration (reference/utils.py)."""
    return qubo_basics.brute_force_solve(instance.matrix, constant=instance.constant, show_progress=show_progress)


def _accept(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / max(temperature, 1e-9))


def simulated_annealing(
    instance: QuboInstance,
    *,
    max_iter: int = 2000,
    temperature: float = 1.0,
    cooling: float = 0.995,
    restarts: int = 5,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Lightweight annealer for moderate sizes; keeps authors' QUBO structure intact."""
    Q = instance.matrix
    rng = np.random.default_rng(seed)
    size = Q.shape[0]

    best_solution: np.ndarray | None = None
    best_value = math.inf

    for _ in range(restarts):
        state = rng.integers(0, 2, size=size, dtype=int)
        value = qubo_objective(Q, state, instance.constant)
        temp = temperature

        for _ in range(max_iter):
            idx = rng.integers(0, size)
            candidate = state.copy()
            candidate[idx] ^= 1
            cand_val = qubo_objective(Q, candidate, instance.constant)
            delta = cand_val - value

            if _accept(delta, temp, rng):
                state = candidate
                value = cand_val

            if value < best_value:
                best_value = value
                best_solution = state.copy()

            temp *= cooling

    assert best_solution is not None
    return best_solution, best_value


def _load_or_create_random(
    size: int,
    low: float,
    high: float,
    seed: int,
    data_dir: Path | None,
) -> tuple[np.ndarray, Path | None]:
    if data_dir is None:
        return generate_random_qubo(size, low=low, high=high, seed=seed), None

    folder = data_dir / "random_instances" / f"n{size}"
    folder.mkdir(parents=True, exist_ok=True)
    fname = folder / f"seed{seed}_low{low}_high{high}.npy"
    if fname.exists():
        Q = np.load(fname)
        return Q, fname

    Q = generate_random_qubo(size, low=low, high=high, seed=seed)
    np.save(fname, Q)
    return Q, fname


def build_problem(
    cfg: dict,
    seed: int | None = None,
    *,
    persist_random: bool = True,
    data_dir: Path | None = None,
) -> QuboInstance:
    """Create a QUBO instance from config."""
    problem_type = cfg.get("type", "random")

    if problem_type == "random":
        size = int(cfg.get("size", 6))
        low = float(cfg.get("low", -2.0))
        high = float(cfg.get("high", 2.0))
        if persist_random and seed is None:
            raise ValueError("Random problems with persistence require a seed")

        if persist_random and seed is not None:
            Q, path = _load_or_create_random(size, low, high, seed, data_dir)
            meta = {"low": low, "high": high, "size": size, "seed": seed, "path": str(path) if path else None}
        else:
            Q = generate_random_qubo(size, low=low, high=high, seed=seed)
            meta = {"low": low, "high": high, "size": size, "seed": seed}

        return QuboInstance(matrix=Q, description="random", metadata=meta)

    if problem_type == "graph-coloring":
        graph_cfg = cfg.get("graph", {})
        num_nodes = int(graph_cfg.get("num_nodes", 4))
        num_colors = int(graph_cfg.get("num_colors", 3))
        edges = graph_cfg.get("edges", [])
        penalty = float(cfg.get("penalty", 4.0))
        instance = generate_graph_coloring_qubo(num_nodes, num_colors, edges, penalty)
        instance.metadata["penalty"] = penalty
        return instance

    raise ValueError(f"Unknown problem type: {problem_type}")


def solve_problem(instance: QuboInstance, solver_cfg: dict, seed: int | None = None) -> tuple[np.ndarray, float, str]:
    """Dispatch to the requested classical solver."""
    method = solver_cfg.get("method", "exhaustive").lower()

    if method == "exhaustive":
        solution, value = brute_force_solve(instance, show_progress=bool(solver_cfg.get("progress", False)))
        return solution, value, "exhaustive"

    if method == "annealing":
        solution, value = qubo_basics.simulated_annealing(
            instance.matrix,
            constant=instance.constant,
            max_iter=int(solver_cfg.get("max_iter", 2000)),
            temperature=float(solver_cfg.get("temperature", 1.0)),
            cooling=float(solver_cfg.get("cooling", 0.995)),
            restarts=int(solver_cfg.get("restarts", 5)),
            seed=seed,
        )
        return solution, value, "annealing"

    raise ValueError(f"Unknown solver: {method}")
