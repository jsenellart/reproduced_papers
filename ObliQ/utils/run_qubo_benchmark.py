from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
BENCH_DIR = PROJECT_DIR / "qubo-benchmark" / "benchmark"
INST_DIR = PROJECT_DIR / "qubo-benchmark" / "instances"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "qubo-benchmark" / "results_obliq"

# Add benchmark code to path so we can reuse its loader without modifying it
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

from main import load_problem  # type: ignore  # noqa: E402

from lib.qubo import QuboInstance, qubo_objective, solve_problem
from lib.quantum import maybe_run_quantum


def collect_instances(instances: Iterable[str]) -> list[Path]:
    problems: list[Path] = []
    for entry in instances:
        base = INST_DIR / entry
        if base.suffix == ".npz":
            problems.append(base)
            continue
        if base.is_dir():
            problems.extend(base.rglob("*.npz"))
    return sorted(problems)


def run_solver_on_problem(
    Q: np.ndarray,
    solver_cfg: dict,
    quantum_cfg: dict,
    seed: int,
    *,
    graph_val: int = 0,
) -> tuple[np.ndarray, float, str]:
    quantum_methods = {"obliq-static", "vqc"}
    method = solver_cfg.get("method", "annealing").lower()

    instance = QuboInstance(matrix=Q, constant=0.0, description="benchmark")

    if method in quantum_methods:
        quantum_cfg = dict(quantum_cfg)
        quantum_cfg["enabled"] = True
        quantum_cfg["method"] = method
        q_res = maybe_run_quantum(
            instance.matrix,
            quantum_cfg,
            constant=instance.constant,
            graph_val=graph_val,
        )
        assert q_res is not None
        return np.array(q_res.solution), float(q_res.objective), f"obliq_{q_res.solver}"

    solution, value, method = solve_problem(instance, solver_cfg, seed=seed)

    if quantum_cfg.get("enabled"):
        q_res = maybe_run_quantum(
            instance.matrix,
            quantum_cfg,
            constant=instance.constant,
            graph_val=graph_val,
        )
        if q_res:
            return np.array(q_res.solution), float(q_res.objective), f"obliq_{q_res.solver}"

    return solution, float(value), f"obliq_{method}"


def run_benchmark(
    instances: list[str],
    solver_cfg: dict,
    quantum_cfg: dict,
    results_dir: Path,
    solver_label: str,
    seed: int,
) -> None:
    problems = collect_instances(instances)
    results_dir.mkdir(parents=True, exist_ok=True)

    for prob in problems:
        Q = load_problem(str(prob))
        start = time.time()
        solution, loss, method = run_solver_on_problem(
            Q,
            solver_cfg,
            quantum_cfg,
            solver_label,
            seed,
            graph_val=int(quantum_cfg.get("graph_val", 0)),
        )
        elapsed = time.time() - start

        rel = prob.relative_to(INST_DIR).with_suffix("")
        res_path = results_dir / rel.parent
        res_path.mkdir(parents=True, exist_ok=True)
        res_file = res_path / f"{rel.name}_{solver_label}.pkl"

        G = nx.from_numpy_array(Q)
        payload = {
            "task": rel.name,
            "solver": solver_label,
            "loss": float(loss),
            "time": elapsed,
            "x": solution,
            "success": bool(np.all(solution >= 0)),
            "bipartite": nx.algorithms.bipartite.is_bipartite(G),
            "planar": nx.check_planarity(G)[0],
            "method": method,
        }

        with open(res_file, "wb") as fh:
            pickle.dump(payload, fh)

        print(f"[obliq] {rel.name} -> {res_file} (loss={loss:.4f}, time={elapsed:.2f}s)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ObliQ solver on qubo-benchmark instances without modifying the benchmark.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BENCH_DIR / "config.yml"),
        help="Path to qubo-benchmark config.yml",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Where to store pickled results",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="annealing",
        choices=["exhaustive", "annealing", "obliq-static", "vqc"],
        help="Solver backend from ObliQ (classical or photonic)",
    )
    parser.add_argument(
        "--quantum",
        action="store_true",
        help="(Optional) also run quantum ObliQ when classical solver is chosen",
    )
    parser.add_argument(
        "--quantum-method",
        type=str,
        default="obliq-static",
        choices=["obliq-static", "vqc"],
        help="Quantum solver choice",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--anneal-iters", type=int, default=5000, help="Annealing iterations")
    parser.add_argument("--anneal-restarts", type=int, default=5, help="Annealing restarts")
    parser.add_argument("--quantum-restarts", type=int, default=3, help="Random restarts for VQC parameters")
    parser.add_argument("--nsamples", type=int, default=1000, help="Number of photonic samples/shots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as fh:
        configs = yaml.safe_load(fh)  # type: ignore[name-defined]

    solver_cfg = {
        "method": args.solver,
        "max_iter": args.anneal_iters,
        "restarts": args.anneal_restarts,
    }

    quantum_cfg = {
        "enabled": args.quantum or args.solver in {"obliq-static", "vqc"},
        "method": args.quantum_method,
        "restarts": args.quantum_restarts,
        "nsamples": args.nsamples,
        "seed": args.seed,
    }

    solver_label = f"obliq_{args.solver}"
    run_benchmark(
        instances=configs.get("instances", []),
        solver_cfg=solver_cfg,
        quantum_cfg=quantum_cfg,
        results_dir=Path(args.results_dir),
        solver_label=solver_label,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
