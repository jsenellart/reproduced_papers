import math
from itertools import product
from typing import Iterable

import numpy as np
from tqdm import tqdm

from . import save_data
from . import save_data_graph


def _symmetrize(Q: np.ndarray) -> np.ndarray:
    upper = np.triu(Q)
    sym = upper + np.triu(upper, 1).T
    np.fill_diagonal(sym, 0.0)
    return sym


def generate_random_qubo(size: int, low: float = -2.0, high: float = 2.0, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q = rng.uniform(low, high, (size, size))
    return _symmetrize(Q)


def generate_graph_coloring_qubo(
    num_nodes: int,
    num_colors: int,
    edges: Iterable[tuple[int, int]] | Iterable[list[int]],
    penalty: float = 4.0,
) -> tuple[np.ndarray, float]:
    Q, constant = save_data_graph.generate_qubo_matrix(num_nodes, num_colors, edges, penalty)
    return Q, constant


def qubo_objective(Q: np.ndarray, x: np.ndarray, constant: float = 0.0) -> float:
    x_vec = np.asarray(x, dtype=float)
    return float(x_vec @ Q @ x_vec + constant)


def brute_force_solve(Q: np.ndarray, constant: float = 0.0, *, show_progress: bool = False) -> tuple[np.ndarray, float]:
    best_solution: np.ndarray | None = None
    best_value = math.inf
    total = 2 ** Q.shape[0]
    iterator = product([0, 1], repeat=Q.shape[0])
    if show_progress and total <= 2**20:
        iterator = tqdm(iterator, total=total, desc="Brute-force QUBO", leave=False)
    for combination in iterator:
        x = np.fromiter(combination, dtype=int, count=Q.shape[0])
        value = qubo_objective(Q, x, constant)
        if value < best_value:
            best_value = value
            best_solution = x
    assert best_solution is not None
    return best_solution, best_value


def _accept(delta: float, temperature: float, rng: np.random.Generator) -> bool:
    if delta <= 0:
        return True
    return rng.random() < math.exp(-delta / max(temperature, 1e-9))


def simulated_annealing(
    Q: np.ndarray,
    constant: float = 0.0,
    *,
    max_iter: int = 2000,
    temperature: float = 1.0,
    cooling: float = 0.995,
    restarts: int = 5,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    size = Q.shape[0]

    best_solution: np.ndarray | None = None
    best_value = math.inf

    for _ in range(restarts):
        state = rng.integers(0, 2, size=size, dtype=int)
        value = qubo_objective(Q, state, constant)
        temp = temperature

        for _ in range(max_iter):
            idx = rng.integers(0, size)
            candidate = state.copy()
            candidate[idx] ^= 1
            cand_val = qubo_objective(Q, candidate, constant)
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
