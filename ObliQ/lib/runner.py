from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .qubo import QuboInstance, build_problem, qubo_objective, solve_problem
from .quantum import maybe_run_quantum


def _serialize_edges(metadata: dict) -> list[list[int]]:
    edges = metadata.get("edges", [])
    return [list(edge) for edge in edges]


def _save_arrays(run_dir: Path, instance: QuboInstance, solution: np.ndarray) -> None:
    np.save(run_dir / "qubo.npy", instance.matrix)
    np.save(run_dir / "solution.npy", solution)


def train_and_evaluate(cfg: dict, run_dir: Path) -> None:
    """Generate a QUBO, run a classical solver, and persist artifacts."""
    logger = logging.getLogger(__name__)
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))

    problem_cfg = cfg.get("problem", {})
    solver_cfg = cfg.get("solver", {})

    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"

    instance = build_problem(
        problem_cfg,
        seed=seed,
        persist_random=True,
        data_dir=data_dir,
    )

    quantum_methods = {"obliq", "obliq-static", "vqc"}
    classical_method = solver_cfg.get("method", "exhaustive").lower()

    quantum_cfg = dict(cfg.get("quantum", {}))
    quantum_cfg.setdefault("seed", seed)

    results: dict[str, object] = {
        "description": instance.description,
        "constant": instance.constant,
        "num_variables": int(instance.matrix.shape[0]),
        "metadata": {**instance.metadata, "edges": _serialize_edges(instance.metadata)},
    }

    if classical_method in quantum_methods:
        quantum_cfg["enabled"] = True
        quantum_cfg["method"] = classical_method
        q_result = maybe_run_quantum(
            instance.matrix,
            quantum_cfg,
            constant=instance.constant,
            graph_val=int(quantum_cfg.get("graph_val", 0)),
        )
        if q_result:
            _save_arrays(run_dir, instance, np.array(q_result.solution, dtype=int))
            results["solver"] = q_result.solver
            results["objective"] = q_result.objective
            results["solution"] = q_result.solution
            results["objective_check"] = qubo_objective(
                instance.matrix, np.array(q_result.solution), instance.constant
            )
        else:
            logger.warning("Quantum solver unavailable; falling back to classical solver.")
            fallback_cfg = dict(solver_cfg)
            fallback_cfg["method"] = fallback_cfg.get("fallback_method", "exhaustive")
            solution, value, method = solve_problem(instance, fallback_cfg, seed=seed)
            _save_arrays(run_dir, instance, solution)
            results.update(
                {
                    "solver": f"fallback-{method}",
                    "objective": value,
                    "solution": solution.tolist(),
                }
            )
            if np.any(solution):
                results["objective_check"] = qubo_objective(instance.matrix, solution, instance.constant)
    else:
        solution, value, method = solve_problem(instance, solver_cfg, seed=seed)
        _save_arrays(run_dir, instance, solution)

        results.update(
            {
                "solver": method,
                "objective": value,
                "solution": solution.tolist(),
            }
        )
        if np.any(solution):
            results["objective_check"] = qubo_objective(instance.matrix, solution, instance.constant)

        if quantum_cfg.get("enabled"):
            q_result = maybe_run_quantum(
                instance.matrix,
                quantum_cfg,
                constant=instance.constant,
                graph_val=int(quantum_cfg.get("graph_val", 0)),
            )
            if q_result:
                results["quantum"] = {
                    "solver": q_result.solver,
                    "solution": q_result.solution,
                    "objective": q_result.objective,
                }

    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    obj_val = results.get("objective")
    if obj_val is not None:
        logger.info(
            "Completed %s solver: objective=%.6f, vars=%d",
            results["solver"],
            float(obj_val),
            instance.matrix.shape[0],
        )
    else:
        logger.info("Completed %s solver: vars=%d", results["solver"], instance.matrix.shape[0])
