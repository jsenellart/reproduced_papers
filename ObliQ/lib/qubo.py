from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np

from .reference import qubo_basics


@dataclass
class QuboInstance:
    """Container for a QUBO matrix plus metadata shared across solvers."""

    matrix: np.ndarray
    constant: float = 0.0
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def _random_cache_path(
    size: int,
    low: float,
    high: float,
    seed: int,
    data_dir: Path,
) -> Path:
    sanitized_low = f"{low}".replace(" ", "")
    sanitized_high = f"{high}".replace(" ", "")
    filename = f"seed{seed}_low{sanitized_low}_high{sanitized_high}.npy"
    return data_dir / "random_instances" / f"n{size}" / filename


def _load_or_generate_random(
    problem_cfg: dict[str, Any],
    *,
    seed: int,
    data_dir: Path,
    persist: bool,
) -> np.ndarray:
    size = int(problem_cfg.get("size", 6))
    low = float(problem_cfg.get("low", -2.0))
    high = float(problem_cfg.get("high", 2.0))

    if persist:
        cache_path = _random_cache_path(size, low, high, seed, data_dir)
        if cache_path.exists():
            return np.load(cache_path, allow_pickle=False)

    matrix = qubo_basics.generate_random_qubo(size, low=low, high=high, seed=seed)
    if persist:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, matrix)
    return matrix


def _build_random_instance(
    problem_cfg: dict[str, Any],
    *,
    seed: int,
    data_dir: Path,
    persist_random: bool,
) -> QuboInstance:
    matrix = _load_or_generate_random(
        problem_cfg,
        seed=seed,
        data_dir=data_dir,
        persist=persist_random,
    )
    metadata = {
        "type": "random",
        "size": int(problem_cfg.get("size", matrix.shape[0])),
        "seed": seed,
        "low": float(problem_cfg.get("low", -2.0)),
        "high": float(problem_cfg.get("high", 2.0)),
    }
    description = f"random_n{metadata['size']}_seed{seed}"
    return QuboInstance(matrix=matrix, constant=0.0, description=description, metadata=metadata)


def _build_graph_instance(problem_cfg: dict[str, Any]) -> QuboInstance:
    graph_cfg = problem_cfg.get("graph", {})
    num_nodes = int(graph_cfg.get("num_nodes", 4))
    num_colors = int(graph_cfg.get("num_colors", 3))
    edges: Iterable[Iterable[int]] = graph_cfg.get("edges", [])
    penalty = float(problem_cfg.get("penalty", graph_cfg.get("penalty", 4.0)))

    matrix, constant = qubo_basics.generate_graph_coloring_qubo(
        num_nodes,
        num_colors,
        edges,
        penalty=penalty,
    )
    metadata = {
        "type": "graph-coloring",
        "num_nodes": num_nodes,
        "num_colors": num_colors,
        "edges": [list(edge) for edge in edges],
        "penalty": penalty,
    }
    description = f"graph-coloring_n{num_nodes}_k{num_colors}"
    return QuboInstance(matrix=matrix, constant=constant, description=description, metadata=metadata)


def _graph_from_config(graph_cfg: dict[str, Any], *, seed: int) -> nx.Graph:
    num_nodes = graph_cfg.get("num_nodes")
    edges = graph_cfg.get("edges")
    edge_prob = graph_cfg.get("edge_prob", graph_cfg.get("probability", 0.5))

    if edges:
        inferred_nodes = {int(idx) for edge in edges for idx in edge}
        if num_nodes is None:
            num_nodes = (max(inferred_nodes) + 1) if inferred_nodes else 0
        graph = nx.Graph()
        graph.add_nodes_from(range(int(num_nodes)))
        graph.add_edges_from((int(u), int(v)) for u, v in edges)
        return graph

    if num_nodes is None:
        num_nodes = 4
    prob = float(edge_prob)
    rng_seed = int(seed)
    return nx.erdos_renyi_graph(int(num_nodes), prob, seed=rng_seed)


def graph_to_matrix_maxcut(graph: nx.Graph) -> np.ndarray:
    adjacency = nx.to_numpy_array(graph, dtype=float)
    degrees = np.array([graph.degree[node] for node in graph.nodes()], dtype=float)
    np.fill_diagonal(adjacency, -degrees)
    return adjacency


def _build_maxcut_instance(
    problem_cfg: dict[str, Any],
    *,
    seed: int,
) -> QuboInstance:
    graph_cfg = problem_cfg.get("graph", {})
    graph = _graph_from_config(graph_cfg, seed=seed)
    matrix = graph_to_matrix_maxcut(graph)
    metadata = {
        "type": "maxcut",
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "edge_prob": float(graph_cfg.get("edge_prob", graph_cfg.get("probability", 0.5))),
        "edges": [list(edge) for edge in graph.edges()],
    }
    description = f"maxcut_n{graph.number_of_nodes()}_m{graph.number_of_edges()}"
    return QuboInstance(matrix=matrix, constant=0.0, description=description, metadata=metadata)


def build_problem(
    problem_cfg: dict[str, Any],
    *,
    seed: int = 0,
    persist_random: bool = True,
    data_dir: Path | None = None,
) -> QuboInstance:
    """Create a QUBO instance from config (random or graph-coloring)."""

    problem_type = (problem_cfg.get("type") or "random").lower()
    if data_dir is None:
        data_dir = Path.cwd() / "data"

    if problem_type == "random":
        return _build_random_instance(
            problem_cfg,
            seed=seed,
            data_dir=data_dir,
            persist_random=persist_random,
        )
    if problem_type in {"graph", "graph-coloring", "graph_coloring"}:
        return _build_graph_instance(problem_cfg)
    if problem_type == "maxcut":
        return _build_maxcut_instance(problem_cfg, seed=seed)

    raise ValueError(f"Unknown problem type: {problem_type}")


def solve_problem(
    instance: QuboInstance,
    solver_cfg: dict[str, Any],
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, float, str]:
    """Run the requested classical solver and return (solution, value, method)."""

    method = (solver_cfg.get("method") or "exhaustive").lower()

    if method == "exhaustive":
        solution, value = qubo_basics.brute_force_solve(
            instance.matrix,
            instance.constant,
            show_progress=bool(solver_cfg.get("progress", False)),
        )
        return solution, float(value), method

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
        return solution, float(value), method

    raise ValueError(f"Unknown solver method: {method}")


def qubo_objective(matrix: np.ndarray, solution: np.ndarray, constant: float = 0.0) -> float:
    return qubo_basics.qubo_objective(matrix, solution, constant)


__all__ = [
    "QuboInstance",
    "build_problem",
    "solve_problem",
    "qubo_objective",
    "graph_to_matrix_maxcut",
]
